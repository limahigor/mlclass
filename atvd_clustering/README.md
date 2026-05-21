# Atividade: Eye Clustering

## Sobre a base utilizada

A atividade foi desenvolvida a partir do arquivo `barrettII_eyes_clustering.xlsx`,
que reúne medidas biométricas de olhos. Para formar os grupos, foram usadas
apenas cinco variáveis numéricas: `AL`, `ACD`, `WTW`, `K1` e `K2`.

Essas variáveis descrevem características importantes da anatomia ocular. Em
termos práticos, elas ajudam a representar tamanho do olho, profundidade da
câmara anterior, largura horizontal e curvatura da córnea.

A coluna `Correto` também está presente na base, mas ela não foi usada para
montar os grupos. A intenção foi deixar a formação dos clusters baseada apenas
nas medidas biométricas. Depois que os grupos ficaram prontos, essa coluna foi
usada somente para complementar a leitura dos resultados.

## Desenvolvimento da solução

A solução foi construída em etapas para garantir que a análise fosse executada
sobre dados consistentes e que o resultado final pudesse ser interpretado com
clareza.

Primeiro, a base foi carregada e passou por uma verificação das colunas
necessárias para o clustering. Esse passo evita seguir com a execução quando
falta alguma informação importante nas variáveis que realmente definem os
grupos. A coluna `Correto`, por ser apenas complementar, não é exigida para a
análise principal.

Depois disso, as variáveis `AL`, `ACD`, `WTW`, `K1` e `K2` foram convertidas para
formato numérico. Essa conversão foi feita porque a análise depende de cálculos
de distância entre os registros. Se houver valores inválidos, textos misturados
ou células vazias nessas colunas, o agrupamento pode falhar ou produzir um
resultado distorcido. Por isso, os registros com problemas nessas variáveis
foram removidos antes da etapa principal.

Em seguida, foi feita uma verificação de linhas duplicadas com base nessas cinco
medidas. Os registros duplicados não foram apagados automaticamente, mas essa
checagem foi mantida para sinalizar quando a base possui repetições exatas nas
variáveis usadas para formar os grupos.

Com os dados limpos, foi aplicada a padronização das variáveis. Esse passo foi
necessário porque cada medida usa uma escala própria. Sem esse ajuste, uma
variável com valores numericamente maiores poderia ter mais peso no cálculo das
distâncias, mesmo sem ser mais importante que as outras. A padronização colocou
as cinco medidas em uma base comparável.

Só depois dessa preparação o algoritmo K-Means foi executado. Como a quantidade
de grupos não era conhecida de antemão, não foi feita uma escolha direta de um
único valor de `k`. O script testou várias possibilidades, de 2 até 6 grupos,
e registrou o desempenho de cada uma.

Para decidir qual solução era mais adequada, foi usado principalmente o índice
de silhouette. Esse indicador ajuda a verificar se os registros ficaram bem
agrupados dentro do seu próprio grupo e bem separados dos demais. Junto com ele,
a inertia também foi registrada como apoio, porque mostra como o agrupamento vai
se comportando à medida que o número de grupos aumenta.

Depois da comparação entre as alternativas testadas, o melhor resultado foi
obtido com **2 grupos**. A partir dessa escolha, o script gerou a classificação
final de cada registro e organizou um resumo estatístico dos grupos, com
quantidade de casos, médias, medianas, desvio padrão, mínimos, máximos e
quartis.

## Análise dos resultados

O resultado mostrou que a base se organiza melhor em dois perfis principais.
Essa divisão é suficiente para separar comportamentos diferentes das medidas
oculares sem fragmentar a base em grupos pequenos demais.

O primeiro grupo reúne olhos com menor comprimento axial, menor profundidade de
câmara anterior, menor medida branco a branco e curvatura corneana mais alta.
Em termos diretos, esse grupo representa olhos mais curtos e mais curvos.

O segundo grupo apresenta o padrão inverso. Nele, os olhos tendem a ter maior
comprimento axial, câmara anterior mais profunda, maior largura branco a branco
e curvatura corneana menor. Na prática, esse grupo representa olhos mais longos
e mais planos.

Essas diferenças aparecem de forma consistente nas estatísticas resumidas e dão
suporte à separação encontrada pelo algoritmo. Os grupos não ficaram definidos
por uma única variável isolada, mas pelo conjunto das medidas consideradas ao
mesmo tempo.

A coluna `Correto` foi analisada apenas depois da formação dos grupos. Ela não
interferiu na segmentação, mas ajudou a observar como os valores `S` e `N` se
distribuem dentro de cada perfil encontrado.

## Conclusão

A atividade permitiu identificar dois perfis oculares bem definidos a partir das
variáveis biométricas da base. O processo foi desenvolvido com leitura dos
dados, validação das colunas, limpeza de registros inválidos, padronização das
medidas, teste de diferentes quantidades de grupos e escolha da melhor solução
com base em um critério objetivo.

Como resultado, a base pôde ser resumida de forma clara em dois perfis: um com
olhos mais curtos e curvos e outro com olhos mais longos e planos.
