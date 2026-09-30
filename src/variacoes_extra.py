"""
Material extra de discussao (nao exigido, mas mencionado no plano): duas
variacoes de arquitetura/janela sobre o mesmo problema (nowcast e forecast),
comparadas contra o E2/E3 originais (janela=12, 1 camada), com o mesmo
criterio de calibracao de limiar (F1 maximo na validacao).

Variacoes:
  - Janela maior: 24 meses de historico em vez de 12 (mesma 1 camada).
  - Mais camadas: 2 camadas LSTM empilhadas (mesma janela de 12 meses).
"""
import sys
import time
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "/home/claude/tf_dl_projeto/src")
from windows_and_split import build_windows, split_mask
from model_rnn import RNNClassifier, treinar, DEVICE, SEED
from calibrar_limiar import melhor_limiar_f1, metrics_com_limiar

DATA_DIR = "/home/claude/tf_dl_projeto/data"
OUT_DIR = "/home/claude/tf_dl_projeto/outputs"

torch.manual_seed(SEED)
np.random.seed(SEED)

painel = pd.read_parquet(f"{DATA_DIR}/painel_modelagem.parquet")


def preparar(window):
    X, y_now, y_fore, meta = build_windows(painel, window=window)
    train, val, test = split_mask(meta)
    return X, y_now, y_fore, train, val, test


def normalizar(X, train_mask):
    mu = X[train_mask].reshape(-1, X.shape[-1]).mean(axis=0)
    sd = X[train_mask].reshape(-1, X.shape[-1]).std(axis=0)
    sd[sd == 0] = 1.0
    return ((X - mu) / sd).astype("float32")


def to_tensors(X, y, mask):
    return (
        torch.from_numpy(X[mask]).to(DEVICE),
        torch.from_numpy(y[mask].astype("float32")).to(DEVICE),
    )


def treinar_variante(nome, window, num_layers, horizonte):
    X, y_now, y_fore, train, val, test = preparar(window)
    y = y_now if horizonte == "now" else y_fore
    ok = ~np.isnan(y)
    Xo, yo = X[ok], y[ok]
    trn, vl, te = train[ok], val[ok], test[ok]

    Xn = normalizar(Xo, trn)
    Xtr, ytr = to_tensors(Xn, yo, trn)
    Xv, yv = to_tensors(Xn, yo, vl)
    Xte, yte = to_tensors(Xn, yo, te)

    torch.manual_seed(SEED)
    model = RNNClassifier(Xn.shape[-1], hidden_size=32, num_layers=num_layers, cell="LSTM").to(DEVICE)
    model, best_epoch, best_auc = treinar(model, Xtr, ytr, Xv, yv, epochs=60, patience=8, log=False)

    model.eval()
    with torch.no_grad():
        prob_val = torch.sigmoid(model(Xv)).cpu().numpy()
        prob_test = torch.sigmoid(model(Xte)).cpu().numpy()
    y_val_np = yv.cpu().numpy()
    y_test_np = yte.cpu().numpy()

    thr, f1_val = melhor_limiar_f1(y_val_np, prob_val)
    r = metrics_com_limiar(y_test_np, prob_test, thr)
    r.update({
        "experimento": nome,
        "horizonte": "Nowcast" if horizonte == "now" else "Forecast t+1",
        "janela": window,
        "camadas": num_layers,
        "melhor_epoca_val": best_epoch,
    })
    return r


if __name__ == "__main__":
    t0 = time.time()
    linhas = []

    print("=== Recalculando E2/E3 originais (janela=12, 1 camada) como referencia ===")
    linhas.append(treinar_variante("Original (referencia)", window=12, num_layers=1, horizonte="now"))
    linhas.append(treinar_variante("Original (referencia)", window=12, num_layers=1, horizonte="fore"))

    print("=== Variacao: janela maior (24 meses, 1 camada) ===")
    linhas.append(treinar_variante("Janela maior (24 meses)", window=24, num_layers=1, horizonte="now"))
    linhas.append(treinar_variante("Janela maior (24 meses)", window=24, num_layers=1, horizonte="fore"))

    print("=== Variacao: mais camadas (janela=12, 2 camadas LSTM) ===")
    linhas.append(treinar_variante("Mais camadas (2x LSTM)", window=12, num_layers=2, horizonte="now"))
    linhas.append(treinar_variante("Mais camadas (2x LSTM)", window=12, num_layers=2, horizonte="fore"))

    tabela = pd.DataFrame(linhas)[
        ["experimento", "horizonte", "janela", "camadas", "melhor_epoca_val",
         "limiar", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    ]
    tabela.to_csv(f"{OUT_DIR}/tabela_variacoes_extra.csv", index=False)
    print("\n" + tabela.to_string(index=False))
    print(f"\nTempo total: {time.time()-t0:.1f}s")
