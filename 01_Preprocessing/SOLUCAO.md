# Solução de Pré-processamento para o Projeto de Diabetes

## 1. Objetivo

O objetivo deste projeto é melhorar o desempenho do classificador de diabetes sem alterar a parte de criação do modelo. O algoritmo continua sendo o mesmo do enunciado:

- `KNeighborsClassifier(n_neighbors=3)`

Logo, toda a otimização foi concentrada em pré-processamento e validação. A solução foi construída para:

- manter os CSVs originais intactos;
- não alterar o valor de `k`;
- não trocar o algoritmo;
- aplicar o pré-processamento em memória;
- permitir validação local antes de cada envio ao servidor;
- documentar de forma honesta o que funcionou localmente e o que aconteceu no teste real.

## 2. Diagnóstico Inicial da Base

O arquivo `diabetes_dataset.csv` possui problemas reais de completude nas features usadas pelo modelo:

- `Glucose`: 5 ausentes
- `BloodPressure`: 35 ausentes
- `SkinThickness`: 227 ausentes
- `Insulin`: 374 ausentes
- `BMI`: 11 ausentes

Esses problemas são especialmente críticos para `k-NN`, porque:

- o algoritmo não trabalha com `NaN`;
- a distância entre pontos é sensível a escala;
- outliers podem distorcer fortemente os vizinhos mais próximos;
- imputação ruim gera pacientes artificiais e prejudica a noção de similaridade.

Também foi considerado o comportamento clássico da base Pima: valores `0` em colunas biomédicas como glicose, pressão, espessura da pele, insulina e BMI costumam representar erro de coleta ou dado não informado. Por isso, o projeto trata `0` como ausente apenas nessas colunas:

- `Glucose`
- `BloodPressure`
- `SkinThickness`
- `Insulin`
- `BMI`

## 3. Estratégia Inicial e o que Aconteceu no Servidor

A primeira versão da solução adotou este pipeline padrão:

1. coerção para numérico;
2. transformação de vazios em ausentes;
3. tratamento de `0` como ausente nas colunas biomédicas problemáticas;
4. imputação por mediana;
5. clipping `1%-99%`;
6. padronização z-score.

Na validação offline inicial, essa estratégia parecia boa e superava alternativas mais simples. Porém, o resultado do teste real no servidor foi:

- `accuracy = 0.6122448979591837`

Esse resultado foi decisivo. A partir dele, a solução deixou de tratar a validação offline como prova suficiente e passou a usar o servidor como fonte de verdade principal.

## 4. Recalibração da Solução

Depois do resultado real do servidor, foi feita uma busca local mais ampla comparando:

- imputação por zero;
- imputação por mediana;
- imputação por média;
- padronização z-score;
- min-max scaling;
- clipping `1%-99%`;
- clipping `5%-95%`;
- versões sem clipping.

O melhor sinal local, dentro das regras da atividade e sem alterar o modelo, foi:

- imputação por média;
- clipping `1%-99%`;
- padronização z-score.

Por isso, esse passou a ser o novo pipeline padrão recomendado para a próxima tentativa.

## 5. Pipeline Atual Recomendado

O pipeline atual implementado no projeto é:

1. converter as colunas de entrada para numérico;
2. transformar strings vazias em ausentes;
3. marcar `0` como ausente apenas em:
   - `Glucose`
   - `BloodPressure`
   - `SkinThickness`
   - `Insulin`
   - `BMI`
4. imputar ausentes com a média calculada no conjunto de treino;
5. aplicar clipping por quantis `1%` e `99%`;
6. padronizar as features com z-score;
7. aplicar a mesma transformação ao `diabetes_app.csv` usando somente os artefatos aprendidos no treino.

Esse pipeline mantém:

- as mesmas 8 features;
- a mesma ordem de colunas;
- o mesmo número de linhas no treino;
- o mesmo número de linhas no arquivo de aplicação;
- o mesmo modelo.

## 6. Por Que Trocar Mediana por Média

A mediana é robusta, mas a busca local mais ampla mostrou que, nesta base específica e com este `k-NN`, a média ficou melhor quando combinada com:

- clipping `1%-99%`;
- padronização z-score.

A interpretação mais provável é que:

- o clipping já reduz parte do efeito dos extremos;
- depois disso, a média passa a representar melhor o centro efetivo das features do que a mediana para esse modelo;
- isso produz uma geometria de distâncias mais favorável ao `k-NN`.

Em outras palavras, a mediana não estava errada, mas deixou de ser a melhor escolha depois que o teste real no servidor mostrou que o pipeline anterior não generalizou tão bem quanto parecia offline.

## 7. Por Que Manter Clipping e Padronização

### Padronização

Padronização continua sendo obrigatória para `k-NN`, porque as features têm escalas muito diferentes.

Sem isso, variáveis com faixa maior dominam a distância euclidiana.

### Clipping

O clipping foi mantido porque:

- preserva todas as linhas;
- reduz o impacto de extremos;
- é compatível com o fato de o arquivo de aplicação precisar manter todas as amostras.

Mesmo assim, o projeto registra que clipping não é uma verdade absoluta. A documentação deixa isso claro e mostra que a escolha final depende de evidência empírica.

## 8. O que Não Foi Mudado

Algumas possibilidades apareceram na análise, mas não entraram na recomendação atual porque o objetivo é fazer apenas uma mudança bem justificada por tentativa.

Foram mantidos sem alteração:

- o algoritmo `k-NN`;
- `n_neighbors=3`;
- as 8 features originais;
- o fluxo principal do script.

Também não foi adotada remoção de colunas nesta rodada. Em especial, surgiu a hipótese de que remover `BloodPressure` poderia ajudar, mas isso foi deixado apenas como hipótese futura, porque já seria uma mudança mais estrutural do que simplesmente ajustar o pré-processamento.

## 9. Arquivos do Projeto

### `preprocessing.py`

Esse arquivo centraliza o pipeline.

Funções principais:

- `get_feature_columns()`
  - devolve a lista oficial de features.

- `coerce_and_mark_missing(df, feature_cols)`
  - converte para numérico;
  - transforma vazios em `NaN`;
  - marca zeros suspeitos como ausentes nas colunas biomédicas.

- `fit_preprocessing(df, feature_cols, imputation_strategy=..., lower_q=..., upper_q=..., ...)`
  - calcula os artefatos usando somente a base de treino;
  - aceita imputação por `mean`, `median` ou `zero`.

- `transform_with_artifacts(df, artifacts, feature_cols)`
  - aplica imputação, clipping e padronização com os artefatos do treino.

- `preprocess_train_and_app(train_df, app_df, feature_cols=None, imputation_strategy="mean", ...)`
  - executa o pipeline principal;
  - retorna treino transformado, aplicação transformada e artefatos.

Funções auxiliares para comparação:

- `fill_missing_with_zero(...)`
- `clip_by_quantiles(...)`
- `filter_rows_by_quantiles(...)`
- `count_missing_by_column(...)`

### `diabetes_csv.py`

Esse é o fluxo principal usado para treinar e prever.

Hoje ele:

- lê `diabetes_dataset.csv`;
- lê `diabetes_app.csv`;
- aplica o pipeline com imputação por média;
- cria `X` e `y`;
- treina o `k-NN` com `k=3`;
- gera previsões;
- envia para o servidor.

### `validate_local_model.py`

Esse arquivo foi atualizado para ser um comparador real de estratégias.

Ele agora compara explicitamente:

- baseline antigo: mediana + padronização + clipping `1%-99%`;
- candidata nova: média + padronização + clipping `1%-99%`;
- candidata nova: média + padronização sem clipping;
- referência com mediana + padronização sem clipping;
- zero-fill como referência fraca.

Também foi adicionada uma verificação de estabilidade com múltiplas sementes, para reduzir o risco de escolher uma estratégia que parece boa apenas por causa de uma única partição aleatória.

### `validate_external_dataset.py`

Esse script tenta validar o pipeline em uma base pública equivalente.

Ele:

- tenta baixar a base;
- detecta corretamente quando o CSV externo não tem cabeçalho;
- normaliza nomes de colunas;
- falha de forma amigável sem internet.

## 10. Resultado Local Mais Relevante

Na busca local mais ampla, a melhor combinação encontrada entre imputação, escala e clipping foi:

- `mean + standard scaling + clipping 1%-99%`

Ela ficou acima do pipeline anterior com mediana na validação local ampla.

Esse foi o principal motivo para trocar o default do projeto.

## 11. Resultado Real no Servidor

O resultado real já obtido e registrado foi:

- pipeline anterior com mediana;
- `accuracy = 0.6122448979591837`

Esse valor é importante porque mudou a interpretação de toda a solução:

- a validação local é útil;
- mas não é suficiente sozinha;
- toda recomendação agora precisa ser explicada à luz do servidor.

## 12. Como Executar

### Fluxo principal

```bash
python3 diabetes_csv.py
```

Esse comando:

- lê treino e aplicação;
- aplica o pipeline padrão atual;
- treina o modelo;
- gera predições;
- envia ao servidor.

### Validação local

```bash
python3 validate_local_model.py
```

Esse comando:

- diagnostica a base;
- confere ausência de `NaN` após o pipeline;
- compara baseline antigo e nova estratégia;
- imprime acurácia, precisão, recall, F1, matriz de confusão e estabilidade.

### Validação externa opcional

```bash
python3 validate_external_dataset.py
```

Ou:

```bash
python3 validate_external_dataset.py --url "https://sua-url-publica/dataset.csv"
```

## 13. Critérios de Aceitação

Antes de uma nova tentativa no servidor, o pipeline precisa satisfazer:

- nenhum `NaN` restante nas features de treino;
- nenhum `NaN` restante nas features de aplicação;
- mesmas 8 colunas do modelo original;
- mesma ordem das colunas;
- mesma quantidade de linhas;
- melhora local consistente sobre o baseline antigo;
- estabilidade razoável em múltiplas sementes.

## 14. Próxima Tentativa Recomendada

A próxima tentativa recomendada é:

- mesmas 8 features;
- imputação por média;
- clipping `1%-99%`;
- padronização z-score;
- `KNeighborsClassifier(n_neighbors=3)` inalterado.

Essa recomendação foi escolhida porque:

- foi a melhor combinação encontrada na busca local mais ampla;
- melhora o baseline antigo na análise offline;
- respeita integralmente a restrição de não mudar o modelo;
- faz apenas uma mudança conceitualmente importante por rodada: trocar a imputação padrão.

## 15. Hipótese Secundária para Rodada Futura

Se a nova tentativa ainda não produzir melhora convincente no servidor, a próxima hipótese controlada será:

- testar a remoção de `BloodPressure`

Mas isso não entra nesta rodada principal, porque:

- é uma mudança mais estrutural;
- mistura seleção de atributos com pré-processamento;
- seria melhor testar separadamente para não confundir a interpretação do resultado.

## 16. Limitações

Mesmo com a solução atual, ainda existem limites importantes:

- o conjunto de aplicação não possui `Outcome` visível;
- a validação offline não reproduz perfeitamente o servidor;
- a base é relativamente pequena;
- há muitas ausências em colunas importantes, principalmente `Insulin` e `SkinThickness`.

## 17. Conclusão

A solução evoluiu em duas fases:

1. uma fase inicial em que a mediana parecia a melhor escolha local;
2. uma fase de recalibração depois do servidor mostrar que essa decisão não foi suficiente.

Hoje, o projeto está ajustado para a próxima tentativa com:

- imputação por média;
- clipping `1%-99%`;
- padronização z-score.

Essa é a recomendação atual mais bem sustentada por evidência local sem mudar o modelo e sem quebrar as regras da atividade.
