# Trabalho Final - Aprendizado Profundo (PPGCA/UNIVALI)
Problema 3: predicao/classificacao de series temporais (RNN/LSTM/GRU + Transformer)

Reaproveitamento da base da dissertacao (Atlas Digital de Desastres + ERA5/Copernicus + MapBiomas),
consolidada em `data/Consolidado_municipio_mes.parquet` e `data/Consolidado_municipio_ano.parquet`
(35-37 municipios costeiros de SC, 1940-2026).

## Formulacao do problema
- Grao: mensal, 35 municipios x 408 meses (1991-2024) = 14.280 municipio-meses
- Resposta: binaria - ocorreu algum desastre (qualquer grupo) no municipio-mes (taxa geral 6,98%)
- E1: baseline (taxa historica do municipio)
- E2: RNN/LSTM/GRU nowcast - janela de 12 meses ate t -> classifica evento em t
- E3: RNN/LSTM/GRU forecast - mesma janela -> classifica evento em t+1
- Extra: variacoes de janela (24 meses) e de profundidade (2 camadas LSTM), e um Transformer encoder
  proprio (sugestao do professor Felipe Viel numa consultoria), comparado com a LSTM em E2/E3

Detalhes completos das decisoes (split temporal, features, viés de registro pos-2012 etc.),
os resultados finais e a leitura critica de cada experimento estao documentados no projeto Claude
("Mestrado - Modelagem estatistica"), docs `AED_dados_recebidos_trabalho_AprendizadoProfundo.md`
e `Plano_experimental_AprendizadoProfundo.md`.

## Resultados finais (teste, 2019-2024, limiar calibrado por F1 na validacao)

| Experimento | Horizonte | ROC-AUC | PR-AUC |
|---|---|---|---|
| E1 baseline | Nowcast | 0,545 | 0,192 |
| E1 baseline | Forecast t+1 | 0,543 | 0,190 |
| Persistencia (extra) | Forecast t+1 | 0,578 | 0,206 |
| **E2 - LSTM nowcast** | Nowcast | **0,590** | **0,239** |
| **E3 - LSTM forecast** | Forecast t+1 | 0,511 | 0,188 |
| Transformer (extra) | Nowcast | 0,600 | 0,263 |
| Transformer (extra) | Forecast t+1 | 0,553 | 0,189 |

O nowcast (E2) supera o baseline com folga; o forecast (E3) fica proximo do acaso - consequencia
direta de a correlacao clima-desastre estar concentrada no mesmo mes (lag 0), achado da AED. O
Transformer supera a LSTM nas duas tarefas, mais notavelmente no forecast, sem resolver essa
limitacao estrutural.

## Como rodar

```bash
python3 -m venv .venv && source .venv/bin/activate   # opcional
pip install -r requirements.txt

python3 src/data_prep.py          # gera data/painel_modelagem.parquet
python3 src/windows_and_split.py  # gera data/janelas.npz e imprime o baseline (E1)
python3 src/model_rnn.py          # treina E2 e E3 (RNN/LSTM/GRU), gera tabela comparativa
python3 src/calibrar_limiar.py    # calibra o limiar de decisao (F1 maximo na validacao) para E1/E2/E3
python3 src/variacoes_extra.py    # variacoes extra: janela=24 meses e 2 camadas LSTM
python3 src/plot_treinamento.py   # historico de treinamento (perda x ROC-AUC por epoca), 30 epocas sem early stopping
python3 src/model_transformer.py  # Transformer encoder (nowcast e forecast), comparado com a LSTM
```

Ambiente: Python 3.11, PyTorch 2.14 (CPU). TensorFlow era o framework originalmente previsto, mas
sua instalacao expirou por timeout no ambiente usado; o pipeline foi implementado em PyTorch.

## Estrutura

```
data/    - parquets originais do Consolidado + artefatos gerados pelos scripts (janelas, painel)
src/     - pipeline (data_prep, windows_and_split, model_rnn, calibrar_limiar, variacoes_extra,
           plot_treinamento, model_transformer)
outputs/ - resultados, tabelas, figuras e pesos de modelo gerados pelos experimentos
relatorio_tecnico_problema3.md - relatorio tecnico completo (estrutura exigida pelo enunciado)
```

## Status
- [x] Pipeline de dados (rotulo + features + limpeza)
- [x] Janelas deslizantes + split temporal + baseline E1
- [x] E2 - RNN/LSTM/GRU nowcast
- [x] E3 - RNN/LSTM/GRU forecast
- [x] Calibracao do limiar de decisao (F1 maximo na validacao, congelado no teste)
- [x] Variacoes extra: janela maior, mais camadas, Transformer encoder
- [x] Avaliacao comparativa e analise critica
- [x] Relatorio e apresentacao
