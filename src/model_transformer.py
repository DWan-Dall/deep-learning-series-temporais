"""
Variacao extra sugerida pelo professor: um Transformer encoder (mesma ideia
do Problema 4 do enunciado e do exemplo classico de classificacao de series
temporais com Transformer) comparado contra a LSTM (E2/E3), no mesmo
pipeline de dados, janela e protocolo de calibracao de limiar.

Arquitetura: projecao linear das features -> embedding posicional aprendido
-> blocos de self-attention (TransformerEncoder) -> global average pooling
no tempo -> camada linear -> logit. E um encoder-only, sem decoder, que e o
padrao usual para classificacao (nao geracao) de sequencias.
"""
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from model_rnn import carregar, to_tensors, treinar, avaliar, DEVICE, SEED
from calibrar_limiar import melhor_limiar_f1, metrics_com_limiar

OUT_DIR = "/home/claude/tf_dl_projeto/outputs"

torch.manual_seed(SEED)
np.random.seed(SEED)


class TransformerClassifier(nn.Module):
    def __init__(self, n_features, seq_len=12, d_model=32, nhead=4, num_layers=2,
                 dim_feedforward=64, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation="relu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        h = self.input_proj(x) + self.pos_embedding
        h = self.encoder(h)
        pooled = h.mean(dim=1)  # global average pooling no tempo
        pooled = self.drop(pooled)
        return self.fc(pooled).squeeze(-1)


def treinar_transformer(horizonte):
    X, y, train, val, test = carregar(horizonte)
    Xtr, ytr = to_tensors(X, y, train)
    Xval, yval = to_tensors(X, y, val)
    Xte, yte = to_tensors(X, y, test)

    torch.manual_seed(SEED)
    model = TransformerClassifier(X.shape[-1], seq_len=X.shape[1]).to(DEVICE)
    model, best_epoch, best_auc = treinar(model, Xtr, ytr, Xval, yval, epochs=60, patience=8, log=False)

    model.eval()
    with torch.no_grad():
        prob_val = torch.sigmoid(model(Xval)).cpu().numpy()
        prob_test = torch.sigmoid(model(Xte)).cpu().numpy()
    y_val_np, y_test_np = yval.cpu().numpy(), yte.cpu().numpy()

    thr, f1_val = melhor_limiar_f1(y_val_np, prob_val)
    r = metrics_com_limiar(y_test_np, prob_test, thr)
    n_params = sum(p.numel() for p in model.parameters())
    r.update({
        "experimento": "Transformer (extra)",
        "horizonte": "Nowcast" if horizonte == "now" else "Forecast t+1",
        "melhor_epoca_val": best_epoch,
        "n_parametros": n_params,
    })
    torch.save(model.state_dict(), f"{OUT_DIR}/modelo_transformer_{horizonte}.pt")
    return r


if __name__ == "__main__":
    t0 = time.time()
    linhas = []
    print("=== Transformer encoder — nowcast ===")
    linhas.append(treinar_transformer("now"))
    print("=== Transformer encoder — forecast t+1 ===")
    linhas.append(treinar_transformer("fore"))

    tabela = pd.DataFrame(linhas)[
        ["experimento", "horizonte", "n_parametros", "melhor_epoca_val",
         "limiar", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    ]
    tabela.to_csv(f"{OUT_DIR}/tabela_transformer.csv", index=False)
    print("\n" + tabela.to_string(index=False))
    print(f"\nTempo total: {time.time()-t0:.1f}s")
