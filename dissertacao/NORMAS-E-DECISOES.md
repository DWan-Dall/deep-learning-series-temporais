# Normas adotadas e decisões de formatação

Este documento registra **quais normas a dissertação segue**, **onde ela se afasta
do padrão estrito** e **por quê**, para responder a eventuais questionamentos da
banca, da secretaria do PPG ou da Biblioteca da UNIVALI.

**Regra geral:** o trabalho segue a ABNT em tudo. A única exceção é estética e
está concentrada em um único ponto (a caixinha de abertura de capítulo), que pode
ser desligada com uma linha no `thesis.tex`, sem nenhuma outra alteração no texto.

---

## 1. Base: template oficial UNIVALI

O projeto é uma cópia do template LaTeX UNIVALI
(<https://github.com/fockingf/template-latex-univali>, licença LPPL 1.3), baseado
no abnTeX2. Todas as regras dele foram mantidas:

| Norma | Assunto |
|---|---|
| ABNT NBR 14724:2024 | Apresentação de trabalhos acadêmicos (margens, fonte, espaçamento, elementos pré/pós-textuais) |
| ABNT NBR 6024 | Numeração progressiva das seções |
| ABNT NBR 6027 | Sumário |
| ABNT NBR 6023:2025 | Referências |
| ABNT NBR 10520:2023 | Citações |
| Guia UNIVALI | *Livro digital para elaboração de trabalhos acadêmicos* (Biblioteca/PPG) |

---

## 2. Decisões de formatação

### 2.1 Caixinha "CAPÍTULO n" (única personalização)

**O que é:** no início de cada capítulo numerado aparece, à esquerda, uma caixa
com faixa preta escrita "CAPÍTULO" e o número grande embaixo, seguida de uma linha
horizontal. Em apêndices e anexos, a caixa mostra "APÊNDICE"/"ANEXO" e a letra.
Veio do modelo MCA/IEEE usado anteriormente.

**Onde está:** `packages/ppg.cls`, comando `\printchaptertitle`, entre os
comentários `--- Caixinha decorativa` e `--- fim da caixinha ---`.

**Situação normativa:** zona cinzenta, sem proibição explícita.
- A NBR 14724 e a NBR 6024 não tratam de elementos gráficos decorativos em
  títulos: não os exigem nem os proíbem.
- O número dentro da caixa está **alinhado à esquerda**, como a NBR 6024 pede
  para o indicativo numérico.
- O tamanho grande do número na caixa (cerca de 1,1 cm) é um **elemento gráfico**,
  não texto. Todo o **texto** do trabalho, títulos incluídos, continua em 12 pt
  (NBR 14724:2024, item 5.1). Esta é a única exceção de tamanho, e é decorativa.

### 2.2 Número do capítulo só na caixinha (sem repetir no título)

**Configuração atual:** `\numeronotitulofalse`. O título sai como `INTRODUÇÃO`,
e o "1" aparece na caixinha logo acima, alinhado à esquerda.

**Por quê:** "1" na caixa e "1 INTRODUÇÃO" logo abaixo ficaria redundante.

**Ponto que pode ser questionado:** a NBR 6024 prevê que o indicativo numérico
**preceda o título, alinhado à esquerda, separado por um espaço**, ou seja, na
mesma linha. Aqui o número está alinhado à esquerda, mas numa linha própria,
acima do título.

**Argumentos de defesa:**
1. O indicativo numérico existe, é visível, está alinhado à esquerda e precede o
   título (vem imediatamente antes dele, na abertura do capítulo).
2. O **Sumário**, que é o elemento regulado pela NBR 6027, continua no padrão
   estrito, com o número antes do título (`1 INTRODUÇÃO ... 17`).
3. A numeração progressiva das seções internas (1.1, 1.2.1...) não muda e segue a
   NBR 6024 normalmente.
4. A remissão no texto ("ver Capítulo 2") e os links do PDF continuam funcionando.

**Se pedirem para corrigir:** troque no `thesis.tex`

```latex
\numeronotitulofalse   % atual: "INTRODUÇÃO"
```

por

```latex
\numeronotitulotrue    % padrão estrito: "1 INTRODUÇÃO" (a caixinha continua)
```

E se pedirem para **tirar a caixinha** também:

```latex
\caixacapitulofalse    % sem caixinha; o título volta a "1 INTRODUÇÃO" automaticamente
```

Com `\caixacapitulofalse`, o layout fica **idêntico** ao template UNIVALI original,
qualquer que seja o valor de `\numeronotitulo`.

### 2.3 Apêndices e anexos: sempre no padrão

Independentemente das chaves, o título de apêndices e anexos sempre mantém a
identificação por letra antes do título (`APÊNDICE A  TÍTULO` / `ANEXO A  TÍTULO`),
exatamente como no template UNIVALI. A caixinha só se soma a ele.

> Observação: a NBR 14724 mostra travessão entre a letra e o título
> ("APÊNDICE A – Título"). O template UNIVALI usa um espaço. Isso é do template
> original e não foi alterado aqui. Se a Biblioteca pedir, dá pra ajustar.

### 2.4 Títulos sem número (Resumo, Sumário, Referências...)

Sem caixinha e centralizados, como manda a NBR 14724. Nada foi alterado aqui.

### 2.5 Tamanho de fonte: 12 pt em todos os títulos

O modelo MCA/IEEE anterior usava títulos de capítulo grandes (cerca de 17 pt).
**Isso não foi trazido**: a NBR 14724:2024 (5.1) pede fonte 12 para todo o texto,
e títulos não estão entre as exceções (que são para tamanho **menor**: citações
longas, notas, legendas, paginação etc.). Fica tudo como no template UNIVALI.

### 2.6 Família de fonte: Times New Roman

Corpo e títulos em Times New Roman, como no template UNIVALI. O modelo anterior
usava títulos sem serifa (tipo Arial/Helvetica). Optamos por **não** trazer isso,
para manter uma única família tipográfica e evitar questionamento.

### 2.7 Destaque gradativo das seções (negrito × itálico)

A mistura de negrito, normal e itálico nos títulos de seção **é intencional e
segue a NBR 6024**, que pede destaque **gradativo** e distinto para cada nível,
com o mesmo destaque repetido no Sumário:

| Nível | Exemplo | Destaque |
|---|---|---|
| Primária (capítulo) | 1 INTRODUÇÃO | MAIÚSCULAS + negrito |
| Secundária | 1.1 Problema de pesquisa | negrito |
| Terciária | 1.2.1 Objetivo geral | normal (redondo) |
| Quaternária | 1.2.1.1 Título | itálico |
| Quinária | 1.2.1.1.1 Título | normal |

O que seria erro é o **mesmo** nível aparecer com estilos diferentes, ou o Sumário
não acompanhar o texto. O template garante isso automaticamente.

---

## 3. Resumo para responder a um questionamento

> A formatação segue integralmente o template LaTeX oficial UNIVALI (NBR 14724:2024,
> 6024, 6027, 6023:2025 e 10520:2023). A única diferença é uma caixa gráfica na
> abertura dos capítulos, com o indicativo numérico alinhado à esquerda. Todo o
> texto, títulos incluídos, está em 12 pt, e o Sumário e a numeração progressiva
> seguem o padrão estrito. Se a Biblioteca preferir o número na mesma linha do
> título, ou a remoção da caixa, a mudança é de uma linha e não afeta o conteúdo.

---

## 4. Histórico de decisões

| Data | Decisão |
|---|---|
| 2026-11-14 | Adotado o template UNIVALI como base, com a caixinha "CAPÍTULO n" do modelo MCA/IEEE |
| 2026-11-14 | Caixinha movida para a esquerda, com o número só nela (`\numeronotitulofalse`); chaves liga/desliga criadas |
| 2026-11-14 | Mantidos: 12 pt em todos os títulos, Times New Roman e destaque gradativo da NBR 6024 |

> **Atenção:** antes do depósito, confira a versão vigente do guia da Biblioteca
> UNIVALI. Este documento descreve as normas como entendidas na data acima.
