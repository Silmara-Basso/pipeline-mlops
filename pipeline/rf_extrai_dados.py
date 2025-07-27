# Automação do Pipeline de Machine Learning
# Módulo de Extração dos Dados

import pandas as pd

# Define a função para extrair os dados de um arquivo CSV
def extrai_dados(file_path):

    try:

        data = pd.read_csv(file_path)
        print("Dados extraídos com sucesso!")
        return data

    except Exception as e:

        print(f"Erro ao extrair dados: {e}")
        return None
