"""
Gera o historico de treinamento (perda de treino e ROC-AUC de validacao por
epoca) para E2 (nowcast) e E3 (forecast), exigido no checklist do enunciado
("Historico de treinamento apresentado"). Retreina do zero so para capturar
o log epoca a epoca (os pesos finais salvos em outputs/ nao mudam).
"""
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from model_rnn import RNNClassifier, carregar, to_tensors, DEVICE, SEED

OUT_DIR = "/home/claude/tf_dl_projeto/outputs"


def treinar_com_log(model, Xtr, ytr, Xval, yval, epochs=30, lr=1e-3, batch_size=128):
    pos_weight = torch.tensor([(ytr == 0).sum() / max((ytr == 1).sum(), 1)], device=DEVICE)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = Xtr.shape[0]
    hist = {"epoch": [], "loss_treino": [], "val_roc_auc": []}
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
            val_prob = torch.sigmoid(model(Xval)).cpu().numpy()
        val_auc = roc_auc_score(yval.cpu().numpy(), val_prob)
        hist["epoch"].append(epoch)
        hist["loss_treino"].append(total_loss / n)
        hist["val_roc_auc"].append(val_auc)
    return hist


fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
cores = {"now": "#3f7a6e", "fore": "#c1622d"}
nomes = {"now": "E2 — LSTM nowcast", "fore": "E3 — LSTM forecast t+1"}

for horizonte in ["now", "fore"]:
    X, y, train, val, test = carregar(horizonte)
    Xtr, ytr = to_tensors(X, y, train)
    Xval, yval = to_tensors(X, y, val)
    torch.manual_seed(SEED)
    model = RNNClassifier(X.shape[-1], hidden_size=32, num_layers=1, cell="LSTM").to(DEVICE)
    hist = treinar_com_log(model, Xtr, ytr, Xval, yval, epochs=30)

    axes[0].plot(hist["epoch"], hist["loss_treino"], label=nomes[horizonte], color=cores[horizonte], linewidth=2)
    axes[1].plot(hist["epoch"], hist["val_roc_auc"], label=nomes[horizonte], color=cores[horizonte], linewidth=2)
    melhor_ep = int(np.argmax(hist["val_roc_auc"])) + 1
    axes[1].scatter([melhor_ep], [hist["val_roc_auc"][melhor_ep - 1]], color=cores[horizonte], zorder=5, s=50)

axes[0].set_xlabel("Época")
axes[0].set_ylabel("Perda de treino (BCE)")
axes[0].set_title("Perda de treino por época")
axes[0].legend(fontsize=8)
axes[0].grid(alpha=0.25)

axes[1].set_xlabel("Época")
axes[1].set_ylabel("ROC-AUC (validação)")
axes[1].set_title("ROC-AUC de validação por época\n(ponto = melhor época, usada por early stopping)")
axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.25)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/historico_treinamento.png", dpi=150)
print(f"Salvo em {OUT_DIR}/historico_treinamento.png")
