# Plano experimental - Trabalho Final de Aprendizado Profundo (Problema 3 e 4)

Para o levantamento dos dados, ver `AED_dados_recebidos_trabalho_AprendizadoProfundo.md`.

## Formulação do problema

- **Grão temporal:** mensal - 35 municípios × 408 meses (1991–2024) = 14.280 município-meses (exclui Camboriú e Paulo Lopes, sem Atlas/MapBiomas).
- **Resposta:** classificação binária - ocorreu algum desastre (qualquer um dos 4 grupos) no município-mês. Taxa geral: 6,98% (997 de 14.280).
- **Duas variações exigidas pelo enunciado (E2 e E3), sobre a mesma arquitetura recorrente:**
  - **E2 - nowcast:** janela de 12 meses terminando em *t* → classifica o evento em *t*.
  - **E3 - forecast:** a mesma janela terminando em *t* → classifica o evento em *t+1* (previsão real, um passo à frente).
- **E1 - baseline:** taxa histórica de evento do município (calculada só com o treino), na linha do que a dissertação já usava como referência no Objetivo_02 (porém de forma anual).

## Features selecionadas (16 no total)

ERA5 mensal (11): `era5_t2m_C`, `era5_tp_mm_mes`, `era5_tp_max`, `era5_cp_mm_mes`, `era5_tcc_pct`, `era5_vento_vel_m_s`, `era5_e_mm_dia`, `era5_pev_mm_dia`, `era5_ssr_W_m2`, `era5_slhf_W_m2`, `era5_tp_dp` (heterogeneidade espacial da chuva no município).

MapBiomas anual, repetido nos 12 meses (5): `mapb_cob_n2_area_urbanizada_pct`, `mapb_cob_n1_floresta_pct`, `mapb_cob_n2_campo_alagado_e_area_pantanosa_pct`, `mapb_urb_pct_area`, `mapb_risco_total_ha`.

Excluídos deliberadamente: bloco `mapb_agua_*` (~80% ausente) e `era5_tvh_dp`/`era5_tvl_dp` (100% vazias).

## Split temporal (decidido, com justificativa)

Treino: 1991–2014 (9.695 janelas) · Validação: 2015–2018 (1.680) · Teste: 2019–2024 (2.520). Desenhado de propósito para o teste cair inteiramente no regime pós-2012. A taxa observada de evento sobe de 4,25% (treino) para 16,79% (teste) - (o próprio split expõe o viés de registro.)

## Resultados finais - E1/E2/E3 (teste, limiar calibrado)

Limiar de decisão escolhido maximizando F1 **apenas na validação** e aplicado congelado no teste - mesmo critério para os três experimentos (corrige o problema anterior de limiar por prevalência, que degenerava em "prever quase tudo como positivo").

| Experimento | Horizonte | Limiar | Precisão | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|
| E1 baseline (taxa histórica) | Nowcast | 0,058 | 0,222 | 0,189 | 0,204 | 0,545 | 0,192 |
| E1 baseline (taxa histórica) | Forecast t+1 | 0,058 | 0,220 | 0,188 | 0,203 | 0,543 | 0,190 |
| Extra: persistência | Forecast t+1 | 0,500 | 0,297 | 0,299 | 0,298 | 0,578 | 0,206 |
| **E2 - LSTM nowcast** | Nowcast | 0,515 | 0,232 | 0,418 | **0,298** | **0,590** | **0,239** |
| **E3 - LSTM forecast t+1** | Forecast t+1 | 0,499 | 0,176 | 0,354 | 0,235 | 0,511 | 0,188 |

**Comparação de arquitetura (nowcast, hidden=32, 1 camada, mesmos hiperparâmetros):** RNN simples 0,596 / **LSTM 0,599** / GRU 0,574 de ROC-AUC na validação - diferenças pequenas entre as três, LSTM ligeiramente à frente. Modelo final: **6.433 parâmetros treináveis**.

**Leitura do resultado mais importante:** E2 (nowcast) supera o baseline em ROC-AUC (0,590 vs 0,545) e principalmente em PR-AUC (0,239 vs 0,192, quase +25%). Já **E3 (forecast t+1) despenca para 0,511 de ROC-AUC - quase o acaso, e pior que o baseline de persistência (0,578)**. Isso confirma diretamente a hipótese levantada na AED: a correlação cruzada entre clima e desastres tem pico em lag 0 (mesmo mês). É um resultado cientificamente coerente e defensável - a queda de nowcast para forecast não é um bug, é a confirmação empírica de um achado que já vinha da AED.

**Acurácia (calculada à parte, não usada como métrica principal):** E1 baseline ~75%, E2 LSTM nowcast ~67%, E3 LSTM forecast ~61% - todos abaixo dos ~83% que se consegue só prevendo "sem evento" sempre (taxa de evento no teste ~17%). Esperado: o limiar foi calibrado para maximizar F1 (pega mais casos reais), o que sacrifica acurácia de propósito. Documentado como ponto de discussão, não nas tabelas principais, para não sugerir que acurácia seria a métrica certa aqui.

**Sobreajuste:** re-treinando E2/E3 por 30 épocas completas (sem early stopping) e plotando perda de treino × ROC-AUC de validação por época (`outputs/historico_treinamento.png`), fica visível o padrão clássico de sobreajuste - perda de treino cai suavemente, ROC-AUC de validação estabiliza cedo (época 1-3) e depois só oscila sem ganho líquido. Justifica diretamente o uso de early stopping e serve de resposta pronta para a pergunta oficial de reflexão nº5 do enunciado ("Houve sobreajuste? Como foi identificado?").

## Variações extras exploradas (material de discussão)

Testadas duas variações sobre o E2/E3 (LSTM), com o mesmo protocolo de calibração de limiar:

| Variação | Horizonte | Janela | Camadas | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Original (referência) | Nowcast | 12 | 1 | 0,298 | 0,590 | 0,239 |
| Original (referência) | Forecast t+1 | 12 | 1 | 0,235 | 0,511 | 0,188 |
| Janela maior (24 meses) | Nowcast | 24 | 1 | 0,310 | 0,594 | 0,244 |
| Janela maior (24 meses) | Forecast t+1 | 24 | 1 | 0,238 | 0,505 | 0,185 |
| Mais camadas (2× LSTM) | Nowcast | 12 | 2 | 0,275 | 0,555 | 0,205 |
| Mais camadas (2× LSTM) | Forecast t+1 | 12 | 2 | 0,248 | 0,513 | 0,187 |

Janela maior ajuda um pouco no nowcast (ROC-AUC 0,594, PR-AUC 0,244) mas não resgata o forecast (0,505, ainda no acaso) - reforça o achado de lag 0. Mais camadas piora o nowcast (0,555) sem ganho real no forecast - consistente com o sinal fraco já identificado. Conclusão: a arquitetura simples (1 camada, janela=12) continua sendo a escolha mais defensável entre as recorrentes.

## Transformer encoder - sugestão do professor Felipe Viel (numa consultoria)

Através de uma consultoria com o professor sobre o trabalho; ele comentou as métricas de ROC-AUC (mencionando que outro aluno com o Problema 3 teve resultado parecido) e sugeriu testar Transformer, citando o exemplo clássico de classificação de séries temporais com essa arquitetura (keras.io/examples/timeseries/timeseries_classification_transformer). Foi implementado um Transformer encoder próprio em PyTorch (não copiado do exemplo - arquitetura padrão: projeção linear → embedding posicional aprendido → 2 blocos de auto-atenção (4 cabeças, dim_feedforward=64) → global average pooling → classificador), 18.049 parâmetros, mesmo pipeline de dados/janelas/calibração de limiar.

| Modelo | Horizonte | Parâmetros | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| LSTM (referência) | Nowcast | 6.433 | 0,298 | 0,590 | 0,239 |
| **Transformer** | Nowcast | 18.049 | **0,300** | **0,600** | **0,263** |
| LSTM (referência) | Forecast t+1 | 6.433 | 0,235 | 0,511 | 0,188 |
| **Transformer** | Forecast t+1 | 18.049 | **0,281** | **0,553** | 0,189 |

**Resultado real e positivo:** o Transformer superou a LSTM nas duas tarefas - modesto no nowcast, mais notável no forecast (ROC-AUC 0,553 vs 0,511, quase saindo do nível de acaso). A auto-atenção parece capturar um pouco mais de sinal por poder ponderar livremente qualquer mês da janela, sem a restrição da recorrência sequencial. Mas 0,553 ainda é um desempenho modesto - o ganho é real, não resolve a limitação estrutural do lag 0. Incorporado ao relatório (seção 10, com ressalva explícita para não superinterpretar), à apresentação (slide extra antes da conclusão, citando explicitamente que veio da consultoria) e ao roteiro de fala.

## Estado da implementação

- `src/data_prep.py` - carga do Consolidado, rótulo, seleção/imputação de features.
- `src/windows_and_split.py` - janelas deslizantes, split temporal, baseline E1.
- `src/model_rnn.py` - comparação RNN/LSTM/GRU, treino de E2 e E3 com early stopping (PyTorch - TensorFlow não instalou por timeout no ambiente), tabela comparativa final em `outputs/tabela_comparativa_E1_E2_E3.csv`.
- `src/calibrar_limiar.py` - unifica o critério de limiar (F1 máximo na validação) entre E1/E2/E3, salva `outputs/tabela_comparativa_E1_E2_E3_calibrada.csv` (tabela final acima).
- `src/variacoes_extra.py` - janela maior (24 meses) e mais camadas (2× LSTM), mesmo protocolo de calibração, salva `outputs/tabela_variacoes_extra.csv` (tabela acima).
- `src/plot_treinamento.py` - histórico de treinamento (perda × ROC-AUC de validação por época), salva `outputs/historico_treinamento.png`.
- `src/model_transformer.py` - Transformer encoder (nowcast e forecast), salva `outputs/tabela_transformer.csv` (tabela acima).