"""
Monta as janelas deslizantes (nowcast e forecast) e o split temporal
treino/validacao/teste, e calcula o baseline (E1) para os dois horizontes.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix,
)

DATA_DIR = "/home/fedora-lema/Documentos/Pessoal/Mestrado/deep-learning-series-temporais/data"
WINDOW = 12

# Split temporal (ver justificativa no relatorio: teste inteiramente no
# regime de registro pos-2012, para nao "premiar" o modelo por aprender
# um artefato de sub-registro do periodo antigo)
TRAIN_END = 2014
VAL_END = 2018   # val: 2015-2018
# test: 2019-2024

FEATURES = [
    "era5_t2m_C", "era5_tp_mm_mes", "era5_tp_max", "era5_cp_mm_mes",
    "era5_tcc_pct", "era5_vento_vel_m_s", "era5_e_mm_dia", "era5_pev_mm_dia",
    "era5_ssr_W_m2", "era5_slhf_W_m2", "era5_tp_dp",
    "mapb_cob_n2_area_urbanizada_pct", "mapb_cob_n1_floresta_pct",
    "mapb_cob_n2_campo_alagado_e_area_pantanosa_pct", "mapb_urb_pct_area",
    "mapb_risco_total_ha",
]


def build_windows(df, window=WINDOW):
    """Para cada municipio, gera janelas [t-window+1 .. t] -> (y_t, y_t+1)."""
    X_list, y_now_list, y_fore_list, meta = [], [], [], []
    for geocodigo, g in df.groupby("geocodigo"):
        g = g.sort_values("data").reset_index(drop=True)
        vals = g[FEATURES].values.astype("float32")
        y = g["y"].values.astype("float32")
        datas = g["data"].values
        n = len(g)
        for t in range(window - 1, n):
            X_list.append(vals[t - window + 1: t + 1])
            y_now_list.append(y[t])
            if t + 1 < n:
                y_fore_list.append(y[t + 1])
            else:
                y_fore_list.append(np.nan)
            meta.append((geocodigo, datas[t]))
    X = np.stack(X_list)
    y_now = np.array(y_now_list)
    y_fore = np.array(y_fore_list)
    meta = pd.DataFrame(meta, columns=["geocodigo", "data"])
    meta["ano"] = pd.to_datetime(meta["data"]).dt.year
    return X, y_now, y_fore, meta


def split_mask(meta):
    ano = meta["ano"].values
    train = ano <= TRAIN_END
    val = (ano > TRAIN_END) & (ano <= VAL_END)
    test = ano > VAL_END
    return train, val, test


def metrics_bin(y_true, y_prob, thr=0.5):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    ok = ~np.isnan(y_true)
    y_true, y_prob = y_true[ok], y_prob[ok]
    y_pred = (y_prob >= thr).astype(int)
    out = {
        "n": len(y_true),
        "taxa_observada": y_true.mean(),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    try:
        out["roc_auc"] = roc_auc_score(y_true, y_prob)
        out["pr_auc"] = average_precision_score(y_true, y_prob)
    except ValueError:
        out["roc_auc"] = np.nan
        out["pr_auc"] = np.nan
    return out


if __name__ == "__main__":
    painel = pd.read_parquet(f"{DATA_DIR}/painel_modelagem.parquet")
    X, y_now, y_fore, meta = build_windows(painel)
    print(f"Janelas construidas: X={X.shape}, y_now={y_now.shape}, y_fore={y_fore.shape}")

    train, val, test = split_mask(meta)
    for nome, m in [("treino", train), ("validacao", val), ("teste", test)]:
        anos = meta.loc[m, "ano"]
        print(f"{nome}: {m.sum()} janelas | anos {anos.min()}-{anos.max()} "
              f"| taxa evento (nowcast) = {y_now[m].mean():.4f}")

    np.savez(
        f"{DATA_DIR}/janelas.npz",
        X=X, y_now=y_now, y_fore=y_fore,
        train=train, val=val, test=test,
        geocodigo=meta["geocodigo"].values, ano=meta["ano"].values,
    )
    print(f"\nJanelas e split salvos em {DATA_DIR}/janelas.npz")

    # -----------------------------------------------------------------
    # Baseline E1: taxa historica do municipio (calculada so com treino)
    # -----------------------------------------------------------------
    df_base = pd.DataFrame({"geocodigo": meta["geocodigo"].values})
    taxa_treino = pd.Series(y_now[train]).groupby(meta.loc[train, "geocodigo"].values).mean()
    taxa_global_treino = y_now[train].mean()
    prob_baseline = df_base["geocodigo"].map(taxa_treino).fillna(taxa_global_treino).values

    print("\n=== Baseline E1 - taxa historica do municipio (nowcast) ===")
    for nome, m in [("validacao", val), ("teste", test)]:
        r = metrics_bin(y_now[m], prob_baseline[m])
        print(nome, {k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()})

    print("\n=== Baseline E1 - taxa historica do municipio (forecast t+1) ===")
    for nome, m in [("validacao", val), ("teste", test)]:
        r = metrics_bin(y_fore[m], prob_baseline[m])
        print(nome, {k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()})

    # Baseline extra so p/ forecast: persistencia (repete o estado do mes atual)
    print("\n=== Baseline extra - persistencia (forecast t+1 = y_t) ===")
    for nome, m in [("validacao", val), ("teste", test)]:
        r = metrics_bin(y_fore[m], y_now[m])
        print(nome, {k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()})
