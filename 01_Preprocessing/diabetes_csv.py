#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Atividade para trabalhar o pré-processamento dos dados.

Criação de modelo preditivo para diabetes e envio para verificação de peformance
no servidor.

@author: Aydano Machado <aydano.machado@gmail.com>
"""

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import requests

from preprocessing import get_feature_columns, preprocess_train_and_app

print("\n - Lendo o arquivo com o dataset sobre diabetes")
data = pd.read_csv("diabetes_dataset.csv")
data_app = pd.read_csv("diabetes_app.csv")

# Criando X and y par ao algorítmo de aprendizagem de máquina.\
print(
    " - Criando X e y para o algoritmo de aprendizagem a partir do arquivo diabetes_dataset"
)
# Caso queira modificar as colunas consideradas basta algera o array a seguir.
feature_cols = get_feature_columns()

print(" - Aplicando pré processamento")
data, data_app, preprocessing_artifacts = preprocess_train_and_app(
    data,
    data_app,
    feature_cols=feature_cols,
    imputation_strategy="median",
)

X = data[feature_cols]
y = data.Outcome

# Ciando o modelo preditivo para a base trabalhada
print(" - Criando modelo preditivo")
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X, y)

# realizando previsões com o arquivo de
print(" - Aplicando modelo e enviando para o servidor")
data_app = data_app[feature_cols]
y_pred = neigh.predict(data_app)

# Enviando previsões realizadas com o modelo para o servidor
URL = "https://aydanomachado.com/mlclass/01_Preprocessing.php"

# TODO Substituir pela sua chave aqui
DEV_KEY = "Laplace"

# json para ser enviado para o servidor
data = {"dev_key": DEV_KEY, "predictions": pd.Series(y_pred).to_json(orient="values")}

# # Enviando requisição e salvando o objeto resposta
r = requests.post(url=URL, data=data)
#
# # Extraindo e imprimindo o texto da resposta
pastebin_url = r.text
print(" - Resposta do servidor:\n", r.text, "\n")
