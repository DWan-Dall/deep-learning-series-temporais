Arquivos originais do Consolidado:

- Consolidado_municipio_mes.parquet
- Consolidado_municipio_ano.parquet
- Consolidado_municipios.parquet
- Consolidado_dicionario_mensal.parquet
- Consolidado_diagnostico_cobertura.parquet

Depois rode `python3 src/data_prep.py` (verificar se necessário troca do nome dos locais de arquivos) e `python3 src/windows_and_split.py` (pip install -U scikit-learn - se não instalado na máquina) (nessa ordem)
para gerar `painel_modelagem.parquet` e `janelas.npz` aqui mesmo.
