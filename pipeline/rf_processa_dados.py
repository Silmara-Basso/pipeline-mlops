# Automação do Pipeline de Machine Learning
# Módulo de Processamento dos Dados

import pandas as pd
from sklearn.preprocessing import StandardScaler

# Define a função para pré-processar os dados de um DataFrame
def preprocessa_dados(df, target_column, is_train=True, scaler=None, columns=None):

    print("Iniciando pré-processamento...")
    
    # Verifica e separa a coluna alvo (target) antes de modificar o DataFrame
    if target_column in df.columns:
        y = df[target_column]
        df = df.drop(columns=[target_column])

    else:

        # Define o target como None se a coluna alvo não existir
        y = None

    if is_train:

        # Aplica One-Hot Encoding em variáveis categóricas para os dados de treino
        df = pd.get_dummies(df, drop_first=True)
        columns = df.columns

    else:

        # Aplica One-Hot Encoding para os novos dados
        df = pd.get_dummies(df)

        missing_cols = set(columns) - set(df.columns)

        for col in missing_cols:
            df[col] = 0

        df = df[columns]

    if is_train:

        # Cria um objeto StandardScaler e ajusta aos dados de treino
        scaler = StandardScaler()

        # Aplica a normalização aos dados
        X_scaled = scaler.fit_transform(df)

    else:

        # Aplica a normalização aos novos dados usando o scaler treinado
        X_scaled = scaler.transform(df)

    print("Pré-processamento concluído.")

    return X_scaled, y, scaler, columns



