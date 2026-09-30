# Análise exploratória dos dados recebidos - Trabalho Final de Aprendizado Profundo

**Contexto:** disciplina de Aprendizado Profundo (PPGCA, prof. Felipe Viel, 2026/2). Foi reaproveitada a base de dados da dissertação, optando pelo **Problema 3 do trabalho - Predição/classificação de séries temporais (RNN/LSTM/GRU)** com acrescimo de **Problema 4 do trabalho - Classificação ou processamento de sequências com Transformers**.

**Situação dos dados: completa.** Os arquivos `Consolidado_municipio_mes.parquet` (38.443 linhas × 216 colunas) e `Consolidado_municipio_ano.parquet` (3.219 linhas × 234 colunas), junto com `Consolidado_municipios.parquet` (perfil dos 37 municípios), os dois dicionários de variáveis e as duas tabelas de diagnóstico, formam a base **Consolidado inteira, com os 37 municípios**, num único par de arquivos por grão temporal.

## Decisão registrada: pasta Copernicus_Era5 (dados brutos) não é necessária

Como este trabalho é da disciplina de Aprendizado Profundo - não é a dissertação, serve só para dar um norte - e o Problema 3 e 4 escolhido trabalha no grão município×mês, os dados brutos do ERA5 por pixel (antes da agregação municipal) não trazem nada que o Consolidado já não tenha. O Consolidado traz o ERA5 agregado por município-mês (média, mínimo, máximo, desvio-padrão entre pixels), que é exatamente o grão que a RNN/LSTM vai consumir.

## O que está em dados, de forma definitiva (em data e data-extra) 
### (Para lembrar!)

- `Compilacao_01.zip`, `Shapefiles.zip`, `R_Work.zip` - material de apoio da dissertação (planilhas brutas, malha municipal, scripts R).
- `AED_Atlas_results.tar.gz`, `AED_ERA5_results.tar.gz`, `AED_MapBiomas_results.tar.gz`, `AED_Completo_results.tar.gz`, `Objetivo_02_results.tar.gz` - resultados já calculados pela dissertação (tabelas, figuras, modelos GLM/GLMM Poisson, GAM, Elastic Net, Random Forest, Gradient Boosting e uma rede neural - ver Tabela 22 do Objetivo_02).
- `Atlas_Digital_Outputs_RS.tar.gz`, `MapBiomas_Outputs_RS.tar.gz` - parquets municipais já processados de cada fonte isolada.
- **Consolidado completo (37 municípios)** em `Consolidado_municipio_mes.parquet` + `Consolidado_municipio_ano.parquet`, com dicionário de variáveis e diagnósticos de cobertura próprios - este é o dataset que vai alimentar o modelo de Deep Learning.

## Estrutura confirmada do Consolidado (agora com os 37 municípios, direto do arquivo oficial)

- **`municipio_mes`**: 38.443 linhas (37 municípios × 1.039 meses, 1940-01 a 2026-07) × 216 colunas. Grade mestra do ERA5; Atlas e MapBiomas encaixados por *left join* (NA fora da janela deles).
- **`municipio_ano`**: 3.219 linhas (37 × 87 anos) × 234 colunas. Traz, além de tudo da mensal, as 16 colunas `atlas_n_eventos_tip_*` (tipologia fina de desastre) e um bloco só de água anual.
- **Cobertura por município (`diagnostico_cobertura`):** confirmado que **Camboriú (4203204) e Paulo Lopes (4212304) têm 0 meses de Atlas e 0 de MapBiomas** - ficam só com ERA5 nos 37; os outros 35 têm as três fontes simultâneas em 408 dos 1.039 meses (39,27%), que correspondem exatamente à janela 1991–2024.
- **Cobertura por ano (`diagnostico_temporal`):** ERA5 sempre 37 municípios (1940–2026); Atlas 35 municípios só entre 1991–2025; MapBiomas 35 municípios só entre 1985–2024.

**Sobre as variáveis `mapb_agua_*` - explicação oficial encontrada no dicionário, mais precisa do que a hipótese anterior:** essas colunas não são um problema de qualidade aleatório; são **faixas de tendência** (leve/médio/forte crescimo ou decréscimo da superfície de água), calculadas pelo MapBiomas só para **25 dos 37 municípios** - por isso 80,2% de ausência exata (confirmado pelo próprio dicionário, coerente com os ~81% que tínhamos estimado na amostra parcial). A ressalva do dicionário é explícita: "não é estado da paisagem" e "não fecha a área do município". Continua valendo *de tratar esse bloco com cautela (excluir ou modelar a ausência como informação).*

Duas colunas do ERA5 (`era5_tvh_dp`, `era5_tvl_dp`) estão 100% vazias e como campo estático sem variação, vazio desde a origem, podendo ser descartadas sem perda.

## Achados da AED já relevantes para o novo trabalho

**Viés de registro é o ponto mais crítico a tratar.** A correlação entre nº de eventos e o ano é 0,815 (forte), mas a correlação entre nº de eventos e extremos climáticos observados é de apenas 0,208 no agregado anual - e a frequência de extremos não cresce ao longo do tempo (correlação com o ano de só 0,024). O registro de desastres praticamente triplica entre 1991–2011 e 2012–2025, apontando para uma mudança na forma de registro (provavelmente relacionada ao sistema S2ID, criado por volta de 2012) e não para um aumento real de eventos climáticos extremos. *Isso tem implicação direta em como dividir treino/validação/teste por tempo.*

**Estrutura temporal da resposta.** A série mensal de desastres do litoral é não estacionária em nível bruto, mas passa a estacionária depois de remover tendência e sazonalidade (ainda resta autocorrelação residual, ou seja, há memória temporal que um RNN/LSTM pode em tese capturar). A correlação cruzada entre clima e desastres pica em **lag 0** para precipitação (0,366) e vento (0,127) - a resposta ao clima é praticamente no mesmo mês.

**A resposta é uma contagem rara e superdispersa.** Em nível anual (Objetivo_02, hidrológico apenas): 26,9% dos município-anos com pelo menos um evento. Contando qualquer grupo de desastre: 44,7% (achado da amostra de 32 municípios). Em nível mensal: 7,2% dos município-meses têm evento. Desbalanceamento de classes é algo a tratar explicitamente em qualquer formulação.

**Uma rede neural tabular simples já foi testada e teve desempenho fraco.** Fonte exata: `Objetivo_02_results.tar.gz` → `Tabelas/Tab_22_desempenho_preditivo.csv` (e `Objetivo_02_relatorio_sintese.txt`, seção 5) - comparação de dez abordagens na validação fora da amostra. 
*Esses arquivos também doram adicionadas na pasta `data-mestrado` para melhor verificação.*
A linha "Rede neural" (família "Aprendizado profundo", modelo feedforward simples, grão município-ano, sem estrutura sequencial) teve pseudo-R² negativo nos dois esquemas de validação: -0,121 (blocos de anos) e -0,107 (blocos de municípios) - pior que Random Forest (+0,065 e +0,038) e Gradient Boosting (+0,014 e +0,057) nesses mesmos esquemas. Gancho natural para o trabalho novo: testar se a estrutura sequencial (RNN/LSTM/GRU - Transformers) melhora sobre essa tentativa tabular anterior, *porém nesse trabalho os dados serão tradados como mensal e não anual como no projeto de dissertação.*

**Seleção de preditores já foi feita na dissertação** (VIF > 5, redundância |ρ| > 0,80), resultando em ~14 variáveis mantidas (clima, uso da terra, anomalia climática anual). Bom ponto de partida, mas pensado para modelo anual/tabular.