"""
E2 (nowcast) e E3 (forecast t+1) - RNN/LSTM/GRU para o Problema 3
Trabalho Final de Aprendizado Profundo - PPGCA/UNIVALI

Antes de treinar os modelos finais, compara rapidamente RNN simples vs LSTM vs GRU
no nowcast (mesma janela/hiperparametros) para justificar a escolha de arquitetura
(pergunta do enunciado: "por que esta arquitetura foi escolhida?").
"""
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score, average_precision_score,
)

DATA_DIR = "/home/fedora-lema/Documentos/Pessoal/Mestrado/deep-learning-series-temporais/data"
OUT_DIR = "/home/fedora-lema/Documentos/Pessoal/Mestrado/deep-learning-series-temporais/outputs"
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------
def carregar(horizonte="now"):
    d = np.load(f"{DATA_DIR}/janelas.npz", allow_pickle=True)
    X = d["X"]
    y = d["y_now"] if horizonte == "now" else d["y_fore"]
    train, val, test = d["train"], d["val"], d["test"]

    ok = ~np.isnan(y)  # forecast: ultimo mes de cada municipio nao tem y_{t+1}
    X, y = X[ok], y[ok]
    train, val, test = train[ok], val[ok], test[ok]

    # normalizacao: media/desvio calculados so no treino, aplicados a tudo
    mu = X[train].reshape(-1, X.shape[-1]).mean(axis=0)
    sd = X[train].reshape(-1, X.shape[-1]).std(axis=0)
    sd[sd == 0] = 1.0
    Xn = (X - mu) / sd

    return Xn.astype("float32"), y.astype("float32"), train, val, test


def to_tensors(X, y, mask):
    return (
        torch.from_numpy(X[mask]).to(DEVICE),
        torch.from_numpy(y[mask]).to(DEVICE),
    )


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------
class RNNClassifier(nn.Module):
    def __init__(self, n_features, hidden_size=32, num_layers=1, cell="LSTM", dropout=0.2):
        super().__init__()
        cell_cls = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}[cell]
        self.rnn = cell_cls(
            input_size=n_features, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        last = out[:, -1, :]          # estado oculto do ultimo passo da janela
        last = self.drop(last)
        return self.fc(last).squeeze(-1)  # logit


def treinar(model, Xtr, ytr, Xval, yval, epochs=60, lr=1e-3, patience=8, batch_size=128, log=True):
    pos_weight = torch.tensor([(ytr == 0).sum() / max((ytr == 1).sum(), 1)], device=DEVICE)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    n = Xtr.shape[0]
    best_auc, best_state, best_epoch, wait = -1, None, 0, 0

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        total_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = Xtr[idx], ytr[idx]
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)

        model.eval()
        with torch.no_grad():
            val_logits = model(Xval)
            val_prob = torch.sigmoid(val_logits).cpu().numpy()
        val_auc = roc_auc_score(yval.cpu().numpy(), val_prob) if yval.sum() > 0 else 0.5

        if log and epoch % 5 == 0:
            print(f"  epoch {epoch:3d} | loss {total_loss/n:.4f} | val ROC-AUC {val_auc:.4f}")

        if val_auc > best_auc:
            best_auc, best_state, best_epoch, wait = val_auc, {k: v.clone() for k, v in model.state_dict().items()}, epoch, 0
        else:
            wait += 1
            if wait >= patience:
                if log:
                    print(f"  early stopping na epoca {epoch} (melhor: {best_epoch}, val ROC-AUC {best_auc:.4f})")
                break

    model.load_state_dict(best_state)
    return model, best_epoch, best_auc


def avaliar(model, X, y, thr=None):
    model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(model(X)).cpu().numpy()
    y_np = y.cpu().numpy()
    if thr is None:
        thr = y_np.mean()  # limiar calibrado pela prevalencia, nao 0.5 (ver nota no plano experimental)
    pred = (prob >= thr).astype(int)
    return {
        "n": len(y_np),
        "taxa_observada": round(float(y_np.mean()), 4),
        "limiar": round(float(thr), 4),
        "precision": round(precision_score(y_np, pred, zero_division=0), 4),
        "recall": round(recall_score(y_np, pred, zero_division=0), 4),
        "f1": round(f1_score(y_np, pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_np, prob), 4) if y_np.sum() > 0 else np.nan,
        "pr_auc": round(average_precision_score(y_np, prob), 4) if y_np.sum() > 0 else np.nan,
    }


# ---------------------------------------------------------------------------
# 1) Comparacao rapida de arquitetura no nowcast (RNN vs LSTM vs GRU)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()

    X, y, train, val, test = carregar("now")
    n_features = X.shape[-1]
    Xtr, ytr = to_tensors(X, y, train)
    Xval, yval = to_tensors(X, y, val)
    Xte, yte = to_tensors(X, y, test)

    print(f"torch {torch.__version__} | device={DEVICE} | seed={SEED}")
    print(f"treino={Xtr.shape}, val={Xval.shape}, teste={Xte.shape}\n")

    print("=== Comparacao de arquitetura (nowcast, hidden=32, 1 camada, 30 epocas) ===")
    resultados_arq = []
    for cell in ["RNN", "LSTM", "GRU"]:
        torch.manual_seed(SEED)
        print(f"\n-- {cell} --")
        model = RNNClassifier(n_features, hidden_size=32, num_layers=1, cell=cell).to(DEVICE)
        model, best_epoch, best_auc = treinar(model, Xtr, ytr, Xval, yval, epochs=30, patience=6, log=False)
        m_val = avaliar(model, Xval, yval)
        m_val["arquitetura"] = cell
        m_val["melhor_epoca"] = best_epoch
        resultados_arq.append(m_val)
        print(f"  melhor epoca {best_epoch} | val: {m_val}")

    df_arq = pd.DataFrame(resultados_arq).set_index("arquitetura")
    df_arq.to_csv(f"{OUT_DIR}/comparacao_arquiteturas_nowcast.csv")
    print("\nResumo da comparacao (validacao):")
    print(df_arq[["roc_auc", "pr_auc", "f1", "melhor_epoca"]])

    melhor_cell = df_arq["roc_auc"].idxmax()
    print(f"\nArquitetura escolhida para E2/E3: {melhor_cell} (maior ROC-AUC na validacao)")

    # -----------------------------------------------------------------
    # 2) E2 - nowcast final, com a arquitetura escolhida
    # -----------------------------------------------------------------
    print(f"\n=== E2 - nowcast final ({melhor_cell}, hidden=32, 1 camada) ===")
    torch.manual_seed(SEED)
    model_e2 = RNNClassifier(n_features, hidden_size=32, num_layers=1, cell=melhor_cell).to(DEVICE)
    model_e2, ep2, auc2 = treinar(model_e2, Xtr, ytr, Xval, yval, epochs=60, patience=8)
    r_e2_val = avaliar(model_e2, Xval, yval)
    r_e2_test = avaliar(model_e2, Xte, yte)
    print("E2 val :", r_e2_val)
    print("E2 test:", r_e2_test)
    torch.save(model_e2.state_dict(), f"{OUT_DIR}/modelo_E2_nowcast_{melhor_cell}.pt")

    # -----------------------------------------------------------------
    # 3) E3 - forecast t+1 final, mesma arquitetura
    # -----------------------------------------------------------------
    Xf, yf, trainf, valf, testf = carregar("fore")
    Xtrf, ytrf = to_tensors(Xf, yf, trainf)
    Xvalf, yvalf = to_tensors(Xf, yf, valf)
    Xtef, ytef = to_tensors(Xf, yf, testf)

    print(f"\n=== E3 - forecast t+1 final ({melhor_cell}, hidden=32, 1 camada) ===")
    torch.manual_seed(SEED)
    model_e3 = RNNClassifier(n_features, hidden_size=32, num_layers=1, cell=melhor_cell).to(DEVICE)
    model_e3, ep3, auc3 = treinar(model_e3, Xtrf, ytrf, Xvalf, yvalf, epochs=60, patience=8)
    r_e3_val = avaliar(model_e3, Xvalf, yvalf)
    r_e3_test = avaliar(model_e3, Xtef, ytef)
    print("E3 val :", r_e3_val)
    print("E3 test:", r_e3_test)
    torch.save(model_e3.state_dict(), f"{OUT_DIR}/modelo_E3_forecast_{melhor_cell}.pt")

    # -----------------------------------------------------------------
    # 4) Tabela final comparando E1 (baseline, ja calculado em windows_and_split.py) / E2 / E3
    # -----------------------------------------------------------------
    tabela = pd.DataFrame([
        {"experimento": "E1 baseline (taxa historica)", "horizonte": "nowcast", "conjunto": "teste",
         "roc_auc": 0.5453, "pr_auc": 0.1915, "f1": 0.0},
        {"experimento": "E1 baseline (taxa historica)", "horizonte": "forecast t+1", "conjunto": "teste",
         "roc_auc": 0.5428, "pr_auc": 0.1899, "f1": 0.0},
        {"experimento": "extra: persistencia", "horizonte": "forecast t+1", "conjunto": "teste",
         "roc_auc": 0.5784, "pr_auc": 0.2057, "f1": 0.2977},
        {"experimento": f"E2 {melhor_cell}", "horizonte": "nowcast", "conjunto": "teste",
         "roc_auc": r_e2_test["roc_auc"], "pr_auc": r_e2_test["pr_auc"], "f1": r_e2_test["f1"]},
        {"experimento": f"E3 {melhor_cell}", "horizonte": "forecast t+1", "conjunto": "teste",
         "roc_auc": r_e3_test["roc_auc"], "pr_auc": r_e3_test["pr_auc"], "f1": r_e3_test["f1"]},
    ])
    tabela.to_csv(f"{OUT_DIR}/tabela_comparativa_E1_E2_E3.csv", index=False)
    print("\n=== TABELA COMPARATIVA FINAL (teste) ===")
    print(tabela.to_string(index=False))

    with open(f"{OUT_DIR}/config_execucao.json", "w") as f:
        json.dump({
            "seed": SEED, "torch_version": torch.__version__, "device": DEVICE,
            "arquitetura_escolhida": melhor_cell, "window": 12,
            "epocas_E2": ep2, "epocas_E3": ep3,
            "tempo_total_s": round(time.time() - t0, 1),
        }, f, indent=2)

    print(f"\nTempo total: {time.time()-t0:.1f}s")
