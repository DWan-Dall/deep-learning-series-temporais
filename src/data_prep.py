"""
Pipeline de dados - Trabalho Final de Aprendizado Profundo (Problema 3: séries temporais)
Daiane Wan-Dall Splitter da Silva - PPGCA/UNIVALI

Le a base Consolidado (municipio x mes), constroi a variavel resposta binaria
(ocorreu algum desastre no municipio-mes), seleciona as features climaticas
(ERA5) e de uso da terra (MapBiomas) e monta as janelas deslizantes por
municipio para alimentar RNN/LSTM/GRU, em dois horizontes:
  - nowcast:  janela [t-W+1 .. t]      -> y_t
  - forecast: janela [t-W+1 .. t]      -> y_{t+1}

Tambem define o split temporal treino/validacao/teste.
"""
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent  # raiz do projeto (pasta acima de src/)
DATA_DIR = BASE_DIR / "data"
WINDOW = 12  # meses de historico usados como entrada da rede

# ---------------------------------------------------------------------------
# 1) Carregar dados
# ---------------------------------------------------------------------------
mes = pd.read_parquet(f"{DATA_DIR}/Consolidado_municipio_mes.parquet")
cobertura = pd.read_parquet(f"{DATA_DIR}/Consolidado_diagnostico_cobertura.parquet")

# Municipios com as 3 fontes (exclui Camboriu e Paulo Lopes, que nao tem
# Atlas nem MapBiomas - ver AED_dados_recebidos_trabalho_AprendizadoProfundo.md)
municipios_validos = cobertura.loc[cobertura["meses_tres_fontes"] > 0, "geocodigo"].tolist()
mes = mes[mes["geocodigo"].isin(municipios_validos)].copy()

# Janela util: onde as 3 fontes coexistem (1991-2024)
mes = mes[(mes["tem_era5"]) & (mes["tem_atlas"]) & (mes["tem_mapbiomas"])].copy()
mes = mes.sort_values(["geocodigo", "data"]).reset_index(drop=True)

print(f"Municipios validos: {len(municipios_validos)}")
print(f"Linhas apos restringir a janela com as 3 fontes: {len(mes)}")
print(f"Periodo: {mes['data'].min()} a {mes['data'].max()}")

# ---------------------------------------------------------------------------
# 2) Variavel resposta: ocorreu algum desastre no municipio-mes?
# ---------------------------------------------------------------------------
grupo_cols = [
    "atlas_n_eventos_climatologico",
    "atlas_n_eventos_hidrologico",
    "atlas_n_eventos_meteorologico",
    "atlas_n_eventos_outros",
]
mes["evento_total"] = mes[grupo_cols].fillna(0).sum(axis=1)
mes["y"] = (mes["evento_total"] > 0).astype(int)

taxa = mes["y"].mean()
print(f"Proporcao de municipio-meses com evento: {taxa:.4f} ({mes['y'].sum()} de {len(mes)})")

# ---------------------------------------------------------------------------
# 3) Selecao de features
#    - ERA5 mensal: variaveis dinamicas relevantes (exclui campos estaticos e
#      as colunas 100% vazias era5_tvh_dp / era5_tvl_dp - ver dicionario)
#    - MapBiomas anual (repetido nos 12 meses do ano): exposicao urbana e
#      cobertura vegetal. Exclui o bloco mapb_agua_* (~80% ausente - variaveis
#      de tendencia, so calculadas p/ 25 dos 37 municipios).
# ---------------------------------------------------------------------------
features_era5 = [
    "era5_t2m_C",           # temperatura media do ar
    "era5_tp_mm_mes",       # precipitacao total do mes
    "era5_tp_max",          # maior valor diario/pixel no mes (proxy de extremo)
    "era5_cp_mm_mes",       # precipitacao convectiva
    "era5_tcc_pct",         # cobertura de nuvens
    "era5_vento_vel_m_s",   # velocidade do vento
    "era5_e_mm_dia",        # evaporacao
    "era5_pev_mm_dia",      # evaporacao potencial
    "era5_ssr_W_m2",        # radiacao solar liquida
    "era5_slhf_W_m2",       # fluxo de calor latente
    "era5_tp_dp",           # heterogeneidade espacial da chuva no municipio
]

features_mapb = [
    "mapb_cob_n2_area_urbanizada_pct",
    "mapb_cob_n1_floresta_pct",
    "mapb_cob_n2_campo_alagado_e_area_pantanosa_pct",
    "mapb_urb_pct_area",
    "mapb_risco_total_ha",
]

features = features_era5 + features_mapb
faltantes = [c for c in features if c not in mes.columns]
if faltantes:
    raise ValueError(f"Colunas nao encontradas no Consolidado: {faltantes}")

na_pct = mes[features].isna().mean().sort_values(ascending=False)
print("\n% de ausencia por feature (na janela util):")
print((na_pct * 100).round(2))

# risco_total_ha pode ter NA quando nao ha area de risco mapeada -> tratar como 0
mes["mapb_risco_total_ha"] = mes["mapb_risco_total_ha"].fillna(0)

# demais NAs residuais: preencher com a mediana do proprio municipio (fallback: mediana global)
for col in features:
    mes[col] = mes.groupby("geocodigo")[col].transform(lambda s: s.fillna(s.median()))
    mes[col] = mes[col].fillna(mes[col].median())

print("\nNAs restantes apos imputacao:", mes[features].isna().sum().sum())

mes.to_parquet(f"{DATA_DIR}/painel_modelagem.parquet", index=False)
print(f"\nPainel salvo em {DATA_DIR}/painel_modelagem.parquet -> shape {mes.shape}")
