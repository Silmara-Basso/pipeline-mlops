# Automação do Pipeline de Machine Learning
# Módulo de Limpeza dos Dados

# Define a função para limpar os dados de um DataFrame
def limpa_dados(df):

    print("Iniciando limpeza de dados...")
    df = df.drop_duplicates()
    df = df.dropna()

    print("Limpeza de dados concluída.")
    return df
