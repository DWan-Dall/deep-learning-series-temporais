# Relatório Técnico — Trabalho Final de Aprendizado Profundo

## Problema 3: Predição/Classificação de Séries Temporais com RNN/LSTM/GRU

**Autora:** Daiane Wan-Dall
**Disciplina:** Aprendizado Profundo — PPGCA/UNIVALI — Prof. Felipe Viel — 2026/2

---

## 1. Introdução

Este trabalho aborda o Problema 3 do enunciado — predição/classificação de séries temporais com redes neurais recorrentes (RNN/LSTM/GRU) — reaproveitando a infraestrutura de dados construída para a dissertação de mestrado da autora, cujo tema é a predição de vulnerabilidade a desastres climáticos em municípios costeiros de Santa Catarina.

O objetivo é prever, mês a mês e município a município, se ocorrerá algum evento de desastre (climatológico, hidrológico, meteorológico ou outro), usando como entrada o histórico climático recente e características de uso do solo. O problema é formulado em duas variações, ambas sobre a mesma arquitetura recorrente de base, conforme exigido pelo enunciado:

- **Nowcast (E2):** dado o histórico dos últimos 12 meses até o mês *t*, classificar se houve evento **em t**.
- **Forecast (E3):** dado o mesmo histórico até *t*, classificar se haverá evento **em t+1** — uma predição real, um passo à frente.

Um baseline simples (E1) e um baseline de persistência são usados como referência de comparação.

## 2. Contextualização

Municípios costeiros de Santa Catarina estão sujeitos a eventos climáticos extremos (chuvas intensas, vendavais, ciclones) que geram desastres registrados oficialmente. Antecipar a ocorrência desses eventos — mesmo que de forma aproximada — é relevante para planejamento de defesa civil e para a própria linha de pesquisa da dissertação, que trata de vulnerabilidade climática nesses municípios.

Este trabalho também se conecta a um resultado anterior da dissertação (Objetivo_02, Tabela 22 — comparação de desempenho preditivo fora da amostra entre dez abordagens): a "Rede neural" testada ali (família "Aprendizado profundo", um modelo feedforward simples treinado no grão município-ano, sem estrutura sequencial) teve pseudo-R² negativo nos dois esquemas de validação (-0,121 em blocos de anos; -0,107 em blocos de municípios), pior que Random Forest (+0,065 e +0,038) e Gradient Boosting (+0,014 e +0,057) nesses mesmos esquemas. Isso motiva a pergunta central deste trabalho: **a estrutura sequencial explícita (RNN/LSTM/GRU) melhora a predição em relação a essa tentativa tabular anterior?** Como os grãos temporais e as tarefas não são idênticos (contagem anual vs. classificação mensal), essa comparação é qualificada, não direta — mas serve de referência.

## 3. Dataset

Os dados vêm de três fontes, já consolidadas em uma base única (`Consolidado_municipio_mes.parquet`) construída para a dissertação:

- **ERA5/Copernicus** — reanálise climática mensal (temperatura, precipitação, vento, radiação, evaporação etc.), agregada por município a partir dos pixels que cobrem cada área. Grade mestra do painel: 37 municípios × 1.039 meses (1940–2026).
- **Atlas Digital de Desastres no Brasil** — registros de desastres por município, com contagem mensal de eventos em 4 grupos (climatológico, hidrológico, meteorológico, outros), disponível entre 1991–2025.
- **MapBiomas** — uso e cobertura do solo, anual, repetido nos 12 meses de cada ano, disponível entre 1985–2024.

As três fontes coexistem simultaneamente em 35 dos 37 municípios (Camboriú e Paulo Lopes não têm registros de Atlas nem de MapBiomas e foram excluídos), na janela 1991–2024. Isso resulta em **14.280 município-meses** (35 municípios × 408 meses) usados neste trabalho.

## 4. Análise Exploratória (AED) — achados que orientaram as decisões

A análise exploratória, feita antes da modelagem, revelou pontos centrais para a formulação do problema:

- **Viés de registro.** A correlação entre número de eventos registrados e o ano é forte (0,815), mas a correlação entre número de eventos e extremos climáticos observados é fraca (0,208), e a frequência desses extremos não aumenta ao longo do tempo (correlação com o ano de apenas 0,024). O registro de desastres praticamente triplica entre 1991–2011 e 2012–2025, sugerindo uma mudança na forma como os desastres passaram a ser reportados (provavelmente ligada ao sistema S2ID, criado por volta de 2012) — não um aumento real de eventos climáticos extremos. Isso teve implicação direta na escolha do split temporal (seção 5).
- **Estrutura temporal.** A série mensal de desastres não é estacionária em nível bruto, mas se torna estacionária após remover tendência e sazonalidade, restando ainda autocorrelação residual — ou seja, há memória temporal que uma RNN/LSTM pode, em tese, capturar.
- **Correlação cruzada clima–desastre concentrada em lag 0.** A correlação entre variáveis climáticas e ocorrência de desastre atinge o pico no mesmo mês (precipitação: 0,366; vento: 0,127), caindo nos meses seguintes. Esse achado é a peça-chave para interpretar por que o forecast (E3) tem desempenho muito inferior ao nowcast (E2) — ver seção 9.
- **Resposta rara e desbalanceada.** Apenas 6,98% dos município-meses têm algum evento registrado — desbalanceamento de classes relevante para a escolha de função de perda e de métricas.
- **Rede neural tabular testada na dissertação teve desempenho fraco** (pseudo-R² negativo, ver seção 2), servindo de referência qualificada.

## 5. Pré-processamento

**Seleção de features (16 no total):**

- ERA5 mensal (11): `era5_t2m_C`, `era5_tp_mm_mes`, `era5_tp_max`, `era5_cp_mm_mes`, `era5_tcc_pct`, `era5_vento_vel_m_s`, `era5_e_mm_dia`, `era5_pev_mm_dia`, `era5_ssr_W_m2`, `era5_slhf_W_m2`, `era5_tp_dp` (heterogeneidade espacial da chuva dentro do município).
- MapBiomas anual, repetido nos 12 meses (5): `mapb_cob_n2_area_urbanizada_pct`, `mapb_cob_n1_floresta_pct`, `mapb_cob_n2_campo_alagado_e_area_pantanosa_pct`, `mapb_urb_pct_area`, `mapb_risco_total_ha`.

Foram deliberadamente excluídas as variáveis `mapb_agua_*` (bloco de tendência de superfície de água, calculado por desenho apenas para 25 dos 37 municípios — por isso ~80% de ausência estrutural, não um problema de qualidade) e `era5_tvh_dp`/`era5_tvl_dp` (100% vazias, campos estáticos sem variação).

**Tratamento de ausências:** `mapb_risco_total_ha` ausente foi tratado como 0 (ausência de área de risco mapeada); ausências residuais foram preenchidas pela mediana do próprio município e, quando ainda ausentes, pela mediana global.

**Variável resposta:** binária — `y = 1` se `evento_total > 0` (soma dos 4 grupos de desastre no município-mês), `y = 0` caso contrário.

**Janelas deslizantes:** para cada município, uma janela de 12 meses de histórico (`[t-11, ..., t]`) gera dois rótulos possíveis: `y_t` (nowcast) e `y_{t+1}` (forecast, quando existe mês seguinte disponível).

**Normalização:** z-score, com média e desvio-padrão calculados **apenas no conjunto de treino** e aplicados a treino/validação/teste, evitando vazamento de informação.

**Split temporal:** treino 1991–2014 (9.695 janelas), validação 2015–2018 (1.680), teste 2019–2024 (2.520). O corte foi desenhado deliberadamente para que o teste caia inteiramente no regime de registro pós-2012 identificado na AED — a taxa observada de evento sobe de 4,25% no treino para 16,79% no teste, o que expõe explicitamente o viés de registro em vez de escondê-lo atrás de uma divisão aleatória.

## 6. Arquitetura

A rede é composta por uma única camada recorrente (RNN, LSTM ou GRU, testadas comparativamente), `hidden_size=32`, seguida de dropout (0,2) e uma camada linear que produz um logit a partir do último passo temporal da janela:

```
entrada (12 meses × 16 features) → camada recorrente (hidden=32) → dropout(0,2) → linear(32→1) → logit
```

**Comparação de arquitetura (nowcast, mesmos hiperparâmetros, 30 épocas, early stopping):**

| Arquitetura | ROC-AUC (validação) | PR-AUC | Melhor época |
|---|---|---|---|
| RNN | 0,596 | 0,124 | 1 |
| **LSTM** | **0,599** | **0,128** | 1 |
| GRU | 0,574 | 0,123 | 1 |

A LSTM foi escolhida por ter o maior ROC-AUC na validação, embora a diferença entre as três células seja pequena. Um achado notável, que se repete também no treinamento final de E2 e E3: **em todos os casos a melhor época de validação foi a primeira**, e o treinamento não melhora depois disso. Isso indica que o sinal preditivo disponível nos dados é fraco e é capturado quase inteiramente de uma vez, não construído progressivamente ao longo de várias épocas — consistente com o que a AED já apontava sobre a força moderada da relação clima–desastre.

**Treinamento:** `BCEWithLogitsLoss` com `pos_weight` (compensa o desbalanceamento de classes sem precisar de reamostragem), otimizador Adam (`lr=1e-3`), *early stopping* com paciência de 8 épocas sobre o ROC-AUC de validação, lotes de 128 exemplos. Semente fixa (`seed=42`) para reprodutibilidade. Implementado em PyTorch 2.14.0 (CPU). O modelo final (LSTM, `hidden_size=32`, 1 camada) tem **6.433 parâmetros treináveis**.

**Histórico de treinamento e sobreajuste.** Para visualizar o comportamento do treinamento além do ponto de corte do early stopping, os modelos E2 e E3 foram re-treinados por 30 épocas completas, sem interromper nem restaurar pesos, registrando a perda de treino e o ROC-AUC de validação a cada época:

![Histórico de treinamento — perda e ROC-AUC de validação por época](../outputs/historico_treinamento.png)

O padrão confirma sobreajuste após a convergência inicial: a perda de treino cai de forma monotônica e suave ao longo das 30 épocas (o modelo continua se ajustando aos dados de treino), enquanto o ROC-AUC de validação estabiliza já por volta da época 1–3 (~0,55–0,60) e passa a oscilar de forma ruidosa sem ganho líquido consistente daí em diante. Esse é exatamente o padrão clássico que justifica o uso de *early stopping*: mais treinamento reduz a perda de treino sem melhorar a generalização. Por isso os modelos finais (E2 e E3) usam os pesos da época de melhor ROC-AUC de validação dentro da janela de paciência (8 épocas sem melhora), não os da última época — no caso do E2, isso coincidiu com a primeira época; no do E3, também.

## 7. Experimentos

| Experimento | Descrição |
|---|---|
| **E1 — baseline** | Taxa histórica de evento do município (calculada só com o treino), aplicada como probabilidade constante por município. |
| **Extra — persistência** | Só no forecast: assume `y_{t+1} = y_t` (o mês seguinte repete o estado do mês atual). |
| **E2 — LSTM nowcast** | Janela de 12 meses → classifica o evento em *t*. |
| **E3 — LSTM forecast** | Mesma janela → classifica o evento em *t+1*. |

**Calibração do limiar de decisão:** inicialmente o limiar foi fixado pela prevalência da classe positiva, o que fez os modelos de rede degenerarem em "prever quase tudo como positivo" (recall≈1,0), produzindo um F1 artificialmente inflado e sem valor discriminativo real. Isso foi corrigido adotando um critério único para todos os experimentos: o limiar é escolhido maximizando o F1 **apenas no conjunto de validação**, e então aplicado congelado no teste — evitando tanto o problema da prevalência quanto o vazamento de informação do teste para a escolha do limiar.

## 8. Resultados

Métricas no conjunto de teste (2019–2024), com limiar calibrado na validação:

| Experimento | Horizonte | Limiar | Precisão | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|
| E1 baseline (taxa histórica) | Nowcast | 0,058 | 0,222 | 0,189 | 0,204 | 0,545 | 0,192 |
| E1 baseline (taxa histórica) | Forecast t+1 | 0,058 | 0,220 | 0,188 | 0,203 | 0,543 | 0,190 |
| Extra: persistência | Forecast t+1 | 0,500 | 0,297 | 0,299 | 0,298 | 0,578 | 0,206 |
| **E2 — LSTM nowcast** | Nowcast | 0,515 | 0,232 | 0,418 | **0,298** | **0,590** | **0,239** |
| **E3 — LSTM forecast** | Forecast t+1 | 0,499 | 0,176 | 0,354 | 0,235 | 0,511 | 0,188 |

## 9. Discussão

O resultado mais importante deste trabalho é o contraste entre E2 e E3. **O nowcast (E2) supera o baseline em todas as métricas** — ROC-AUC de 0,590 contra 0,545, e um ganho relativo de quase 25% em PR-AUC (0,239 contra 0,192), a métrica mais informativa neste cenário de forte desbalanceamento de classes. Já **o forecast (E3) despenca para 0,511 de ROC-AUC — praticamente equivalente ao acaso, e pior até que o baseline de persistência (0,578)**.

Essa queda de nowcast para forecast não é um problema de implementação: ela confirma diretamente o achado da AED de que a correlação cruzada entre clima e desastre tem pico em lag 0 (mesmo mês). Em outras palavras, o histórico climático recente ajuda a identificar se **está acontecendo** um evento no mês corrente, mas quase não contém informação sobre se um evento ocorrerá **no mês seguinte** — a dependência temporal disponível nos dados é praticamente instantânea, não preditiva a um passo. Esse é um resultado cientificamente coerente e defensável, não uma falha do modelo.

Em relação à rede neural tabular testada na dissertação (Objetivo_02, Tabela 22, grão município-ano, tarefa de contagem, pseudo-R² negativo de -0,121/-0,107 nos dois esquemas de validação): embora a comparação não seja direta — grão temporal e tarefa diferem —, os dois exercícios convergem para uma mesma leitura qualitativa: com os preditores climáticos e de uso do solo disponíveis, o sinal preditivo para desastres nesta base é moderado a fraco, e a estrutura sequencial (RNN/LSTM) ajuda claramente na tarefa de mesmo mês (nowcast), mas não resolve a tarefa de predição genuína a um passo adiante.

## 10. Variações adicionais (material extra de discussão)

Além dos experimentos principais, foram testadas duas variações sobre o E2/E3 (LSTM), como material extra de discussão: uma janela de histórico maior (24 meses, o dobro da original) e uma arquitetura mais profunda (2 camadas LSTM empilhadas), mantendo o mesmo protocolo de calibração de limiar (F1 máximo na validação, congelado no teste).

| Variação | Horizonte | Janela | Camadas | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Original (referência) | Nowcast | 12 | 1 | 0,298 | 0,590 | 0,239 |
| Original (referência) | Forecast t+1 | 12 | 1 | 0,235 | 0,511 | 0,188 |
| Janela maior (24 meses) | Nowcast | 24 | 1 | 0,310 | 0,594 | 0,244 |
| Janela maior (24 meses) | Forecast t+1 | 24 | 1 | 0,238 | 0,505 | 0,185 |
| Mais camadas (2× LSTM) | Nowcast | 12 | 2 | 0,275 | 0,555 | 0,205 |
| Mais camadas (2× LSTM) | Forecast t+1 | 12 | 2 | 0,248 | 0,513 | 0,187 |

Duas leituras se destacam. Primeiro, ampliar a janela de histórico para 24 meses traz um ganho pequeno, mas consistente, no nowcast (ROC-AUC 0,594 e PR-AUC 0,244, ambos acima do original) — mais contexto climático ajuda marginalmente a identificar o evento no mês corrente. No forecast, porém, o ganho não se sustenta (ROC-AUC 0,505, ainda ao nível do acaso): mesmo com o dobro de histórico, a informação sobre o mês seguinte continua praticamente ausente, reforçando o achado de que a relação clima–desastre nesta base é de curtíssimo prazo (lag 0), não uma dependência que se acumula ao longo do tempo.

Segundo, empilhar uma segunda camada LSTM piora o desempenho no nowcast (ROC-AUC cai de 0,590 para 0,555) e não traz ganho real no forecast. Isso é coerente com o achado já discutido na comparação de arquitetura (seção 6): o sinal disponível é fraco e é capturado quase todo na primeira época de treinamento — uma rede mais profunda tem mais parâmetros para ajustar com o mesmo sinal fraco, o que tende a dificultar a otimização (com paciência de 8 épocas) em vez de ajudar. Na prática, a arquitetura mais simples (1 camada) continua sendo a escolha mais defensável entre as variações recorrentes testadas.

### Uma arquitetura diferente: Transformer encoder

Além das variações sobre a LSTM, foi testada uma arquitetura de natureza distinta — um Transformer encoder (auto-atenção, sem recorrência), no espírito do Problema 4 do enunciado e de exemplos consagrados de classificação de séries temporais com essa arquitetura. A rede projeta as 16 features em um espaço de 32 dimensões, soma um embedding posicional aprendido, passa por 2 blocos de auto-atenção (4 cabeças, `dim_feedforward=64`, dropout 0,2) e agrega a sequência por *global average pooling* antes da camada de classificação — 18.049 parâmetros treináveis (quase 3× a LSTM), mesmo pipeline de janelas, normalização e calibração de limiar.

| Experimento | Horizonte | Parâmetros | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| E2 — LSTM nowcast (referência) | Nowcast | 6.433 | 0,298 | 0,590 | 0,239 |
| **Transformer — nowcast** | Nowcast | 18.049 | **0,300** | **0,600** | **0,263** |
| E3 — LSTM forecast (referência) | Forecast t+1 | 6.433 | 0,235 | 0,511 | 0,188 |
| **Transformer — forecast t+1** | Forecast t+1 | 18.049 | **0,281** | **0,553** | 0,189 |

O Transformer superou a LSTM nas duas tarefas — modestamente no nowcast (ROC-AUC 0,600 vs 0,590; PR-AUC 0,263 vs 0,239, um ganho relativo de ~10%), e de forma mais notável no forecast (ROC-AUC 0,553 vs 0,511, F1 0,281 vs 0,235). O mecanismo de auto-atenção parece conseguir extrair um pouco mais de sinal preditivo para o mês seguinte do que a recorrência sequencial da LSTM — plausivelmente porque a atenção pondera livremente qualquer mês da janela, sem a restrição de processar a sequência passo a passo. Ainda assim, é importante não superinterpretar: 0,553 de ROC-AUC continua sendo um desempenho modesto (bem distante de um classificador forte), e o padrão geral do trabalho se mantém — prever o mês seguinte continua sendo uma tarefa muito mais difícil do que prever o mês corrente, com qualquer arquitetura testada. O ganho do Transformer é real, mas não muda a conclusão central sobre o lag 0; é antes um refinamento sobre ela.

## 11. Limitações

- **Escassez de eventos positivos.** Mesmo compensando via `pos_weight`, apenas 6,98% dos município-meses têm evento — há pouco sinal positivo para o modelo aprender padrões finos, o que limita o teto de desempenho de qualquer modelo nesta formulação.
- **Viés de registro não totalmente removível.** Parte do que os modelos capturam pode refletir mudanças na forma de reportar desastres (efeito S2ID pós-2012), não necessariamente mudança climática real. O split temporal escolhido expõe esse viés em vez de escondê-lo, mas não o elimina — é uma limitação estrutural da fonte de dados (Atlas), não do modelo.
- **Ganho do Transformer ainda modesto em termos absolutos.** Mesmo sendo a melhor configuração testada, o Transformer no forecast (ROC-AUC 0,553) está longe de um classificador forte — o ganho sobre a LSTM é real, mas não resolve a dificuldade estrutural da tarefa (lag 0).
- **Split temporal deliberadamente conservador.** O corte treino/val/teste foi desenhado para colocar o teste inteiramente no regime pós-2012 (mais desafiador). Um split aleatório provavelmente produziria métricas mais otimistas, mas misturaria os dois regimes de registro e mascararia o viés identificado na AED — a escolha atual prioriza validade científica sobre número bonito.
- **Escala do dataset.** 14.280 município-meses (35 municípios) é um volume moderado para redes recorrentes; a generalização entre municípios com poucos eventos históricos é limitada.

## 12. Conclusão

Este trabalho formulou e resolveu o Problema 3 (séries temporais com RNN/LSTM/GRU) sobre a base de dados da dissertação de mestrado, comparando um baseline de taxa histórica com uma LSTM em duas variações: nowcast (classificar o evento no mês corrente) e forecast (prever o mês seguinte). O nowcast supera o baseline de forma consistente em todas as métricas relevantes, enquanto o forecast fica próximo do acaso — um resultado que não é uma falha, mas a confirmação empírica de um achado já identificado na análise exploratória: a relação entre clima e desastre nesta base é quase instantânea (lag 0), com pouca informação preditiva a um passo adiante. Um Transformer encoder, testado como variação adicional de arquitetura, confirma essa leitura e ao mesmo tempo a refina: supera a LSTM nas duas tarefas (mais notavelmente no forecast), mas sem deixar de ser um desempenho modesto no forecast — o ganho vem do mecanismo de atenção extrair um pouco mais de sinal, não de resolver a limitação estrutural do problema. Como contribuição adicional, o pipeline de dados e modelagem construído aqui é diretamente reaproveitável para trabalhos futuros que usem a mesma base consolidada.

## 13. Referências

- HOCHREITER, S.; SCHMIDHUBER, J. Long Short-Term Memory. *Neural Computation*, v. 9, n. 8, p. 1735–1780, 1997.
- CHO, K. et al. Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation. *EMNLP*, 2014. (arquitetura GRU)
- VASWANI, A. et al. Attention Is All You Need. *NeurIPS*, 2017. (arquitetura Transformer, base do encoder usado na seção 10)
- HERSBACH, H. et al. The ERA5 global reanalysis. *Quarterly Journal of the Royal Meteorological Society*, v. 146, n. 730, p. 1999–2049, 2020. Dados: Copernicus Climate Change Service (C3S), ERA5 monthly averaged data.
- SOUZA, C. M. et al. Reconstructing Three Decades of Land Use and Land Cover Changes in Brazilian Biomes with Landsat Archive and Earth Engine. *Remote Sensing*, v. 12, n. 17, 2758, 2020. Dados: Coleção MapBiomas.
- BRASIL. Ministério da Integração e do Desenvolvimento Regional. Sistema Integrado de Informações sobre Desastres (S2ID) e Atlas Digital de Desastres no Brasil. Disponível em: https://s2id.mi.gov.br/.
- PASZKE, A. et al. PyTorch: An Imperative Style, High-Performance Deep Learning Library. *NeurIPS*, 2019.
- Dados e pipeline reaproveitados da dissertação de mestrado da autora (PPGCA/UNIVALI), sobre vulnerabilidade a desastres climáticos em municípios costeiros de Santa Catarina — ver `Plano_experimental_AprendizadoProfundo.md` e `AED_dados_recebidos_trabalho_AprendizadoProfundo.md` no material de apoio do projeto.

---

## Anexo — reprodutibilidade

- Código-fonte: `data_prep.py`, `windows_and_split.py`, `model_rnn.py`, `calibrar_limiar.py`, `variacoes_extra.py`, `plot_treinamento.py`, `model_transformer.py` (repositório git local, ver README do projeto).
- Semente aleatória: 42.
- **Linguagem:** Python 3.11.15.
- **Bibliotecas principais (modelagem):** PyTorch 2.14.0 (redes neurais e treinamento, execução em CPU), scikit-learn 1.8.0 (métricas: ROC-AUC, PR-AUC, F1, precisão, recall), pandas 3.0.2 e NumPy 2.4.4 (manipulação dos dados e das janelas), pyarrow 25.0.1 (leitura dos arquivos `.parquet` do Consolidado), matplotlib 3.10.9 (gráfico de histórico de treinamento).
- **Outras ferramentas usadas na etapa de preparação dos dados** (antes da modelagem, fora do pipeline final): `pyreadr` (ler os arquivos `.RData` originais da dissertação, já que não há R instalado no ambiente) e `dbfread` (ler as tabelas de atributos dos shapefiles). Git para controle de versão do código.
- **Nota:** a intenção inicial era usar TensorFlow/Keras (como no exemplo de Transformer sugerido pelo professor), mas a instalação travou por timeout no ambiente de execução; PyTorch foi usado como substituto equivalente, aceito pela disciplina.
- Arquitetura final escolhida (E2/E3): LSTM, `hidden_size=32`, 1 camada, dropout 0,2, janela de 12 meses, 6.433 parâmetros treináveis. Transformer extra (seção 10): 18.049 parâmetros.
- Tempo total de treinamento (comparação de arquitetura + E2 + E3): ~25 segundos.
