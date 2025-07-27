# Automação do Pipeline de Machine Learning
# Módulo de Deploy do Modelo

import os
import joblib

# Define a função para carregar o modelo, o scaler e as colunas do pré-processamento
def carrega_modelo(model_path, scaler_path, columns_path):

    # Tenta carregar o modelo, o scaler e as colunas salvas
    try:

        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        columns = joblib.load(columns_path)

        print(f"Modelo, scaler e colunas carregados de {model_path}")

        return model, scaler, columns

    # Trata possíveis exceções ao carregar os arquivos
    except Exception as e:

        print(f"Erro ao carregar o modelo ou pré-processamento: {e}")
        return None, None, None

# Define a função para realizar previsões com o modelo carregado
def faz_previsoes(model, X_new):

    predictions = model.predict(X_new)

    return predictions




