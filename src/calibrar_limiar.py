"""
Calibracao de limiar - unifica o criterio entre E1 (baseline), E2 (nowcast)
e E3 (forecast t+1): para cada modelo, o limiar de decisao e escolhido
maximizando F1 SO na validacao, e depois aplicado (congelado) no teste.
Isso evita o problema anterior (limiar = prevalencia, que degenerava em
"prever quase tudo positivo") e evita usar o teste pra escolher o limiar
(vazamento).
"""
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score, average_precision_score,
)

from model_rnn import RNNClassifier, carregar, to_tensors, DEVICE

DATA_DIR = "/home/claude/tf_dl_projeto/data"
OUT_DIR = "/home/claude/tf_dl_projeto/outputs"
MELHOR_CELL = "LSTM"


def melhor_limiar_f1(y_val, prob_val):
    """Varre limiares candidatos (os proprios valores previstos) e devolve
    o que maximiza F1 na validacao."""
    candidatos = np.unique(np.round(prob_val, 4))
    melhor_thr, melhor_f1 = 0.5, -1
    for thr in candidatos:
        pred = (prob_val >= thr).astype(int)
        f1 = f1_score(y_val, pred, zero_division=0)
        if f1 > melhor_f1:
            melhor_f1, melhor_thr = f1, thr
    return melhor_thr, melhor_f1


def metrics_com_limiar(y_true, prob, thr):
    pred = (prob >= thr).astype(int)
    return {
        "limiar": round(float(thr), 4),
        "precision": round(precision_score(y_true, pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, prob), 4) if y_true.sum() > 0 else np.nan,
        "pr_auc": round(average_precision_score(y_true, prob), 4) if y_true.sum() > 0 else np.nan,
    }


def probs_baseline(horizonte):
    d = np.load(f"{DATA_DIR}/janelas.npz", allow_pickle=True)
    y = d["y_now"] if horizonte == "now" else d["y_fore"]
    train, val, test = d["train"], d["val"], d["test"]
    geocodigo = d["geocodigo"]
    ok = ~np.isnan(y)
    y, train, val, test, geocodigo = y[ok], train[ok], val[ok], test[ok], geocodigo[ok]

    taxa_treino = pd.Series(y[train]).groupby(geocodigo[train]).mean()
    taxa_global = y[train].mean()
    prob = pd.Series(geocodigo).map(taxa_treino).fillna(taxa_global).values
    return y, prob, val, test


def probs_rede(horizonte, arquivo_pesos):
    X, y, train, val, test = carregar(horizonte)
    n_features = X.shape[-1]
    model = RNNClassifier(n_features, hidden_size=32, num_layers=1, cell=MELHOR_CELL).to(DEVICE)
    model.load_state_dict(torch.load(arquivo_pesos, map_location=DEVICE))
    model.eval()
    Xval, yval = to_tensors(X, y, val)
    Xte, yte = to_tensors(X, y, test)
    with torch.no_grad():
        prob_val = torch.sigmoid(model(Xval)).cpu().numpy()
        prob_test = torch.sigmoid(model(Xte)).cpu().numpy()
    return yval.cpu().numpy(), prob_val, yte.cpu().numpy(), prob_test


if __name__ == "__main__":
    linhas = []

    # --- E1 baseline (nowcast e forecast) ---
    for horizonte, nome_h in [("now", "Nowcast"), ("fore", "Forecast t+1")]:
        y, prob, val, test = probs_baseline(horizonte)
        thr, f1_val = melhor_limiar_f1(y[val], prob[val])
        r = metrics_com_limiar(y[test], prob[test], thr)
        r.update({"experimento": "E1 baseline (taxa historica)", "horizonte": nome_h})
        linhas.append(r)

    # --- extra: persistencia (so faz sentido no forecast; ja e binario 0/1) ---
    d = np.load(f"{DATA_DIR}/janelas.npz", allow_pickle=True)
    y_now, y_fore, test = d["y_now"], d["y_fore"], d["test"]
    ok = ~np.isnan(y_fore)
    r = metrics_com_limiar(y_fore[ok & test], y_now[ok & test].astype(float), thr=0.5)
    r.update({"experimento": "extra: persistencia", "horizonte": "Forecast t+1"})
    linhas.append(r)

    # --- E2 nowcast (LSTM) ---
    yval, pval, yte, pte = probs_rede("now", f"{OUT_DIR}/modelo_E2_nowcast_{MELHOR_CELL}.pt")
    thr, f1_val = melhor_limiar_f1(yval, pval)
    r = metrics_com_limiar(yte, pte, thr)
    r.update({"experimento": f"E2 {MELHOR_CELL} nowcast", "horizonte": "Nowcast"})
    linhas.append(r)

    # --- E3 forecast (LSTM) ---
    yval, pval, yte, pte = probs_rede("fore", f"{OUT_DIR}/modelo_E3_forecast_{MELHOR_CELL}.pt")
    thr, f1_val = melhor_limiar_f1(yval, pval)
    r = metrics_com_limiar(yte, pte, thr)
    r.update({"experimento": f"E3 {MELHOR_CELL} forecast", "horizonte": "Forecast t+1"})
    linhas.append(r)

    tabela = pd.DataFrame(linhas)[
        ["experimento", "horizonte", "limiar", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    ]
    tabela.to_csv(f"{OUT_DIR}/tabela_comparativa_E1_E2_E3_calibrada.csv", index=False)
    print("Limiar escolhido SEMPRE pela validacao (maximiza F1), aplicado congelado no teste.\n")
    print(tabela.to_string(index=False))
