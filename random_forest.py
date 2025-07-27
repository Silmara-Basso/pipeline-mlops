# Automação do Pipeline de Machine Learning
# Módulo de Automação

# Imports
import os
import joblib
from pipeline.rf_extrai_dados import extrai_dados
from pipeline.rf_limpa_dados import limpa_dados
from pipeline.rf_processa_dados import preprocessa_dados
from pipeline.rf_treina_modelo import treina_modelo
from pipeline.rf_avalia_modelo import avalia_modelo
from pipeline.rf_deploy import carrega_modelo, faz_previsoes
from sklearn.model_selection import train_test_split

# Função para executar o pipeline
def rf_executa_pipeline(data_path, target_column):
    
    # Diretórios para salvar o modelo e os objetos de pré-processamento
    model_path = "modelos/random_forest_model.pkl"
    scaler_path = "modelos/scaler.pkl"
    columns_path = "modelos/columns.pkl"
    
    ##### Etapa 1: Extração de Dados #####
    print("Executando a Etapa 1 - Extração e Carga dos Dados.\n")
    data = extrai_dados(data_path)
    
    # Verificação de existência da coluna target antes da limpeza
    if target_column not in data.columns:
        print(f"Erro: a coluna `{target_column}` não está presente nos dados extraídos.")
        return
    
    print(f"Coluna `{target_column}` presente nos dados antes da limpeza.")
    
    ##### Etapa 2: Limpeza de Dados #####
    print("\nExecutando a Etapa 2 - Limpeza dos Dados.\n")
    data_cleaned = limpa_dados(data)
    
    # Verificação de existência da coluna target após a limpeza
    if target_column not in data_cleaned.columns:
        print(f"Erro: a coluna `{target_column}` foi removida após a limpeza.")
        return
    
    print(f"Coluna `{target_column}` presente nos dados após a limpeza.")
    
    ##### Etapa 3: Pré-processamento #####

    print("\nExecutando a Etapa 3 - Pré-Processamento.\n")

    # Divisão do dataset em treino e teste antes do pré-processamento
    train_data, test_data = train_test_split(data_cleaned, test_size=0.2, random_state=42)
    
    # Verificação de existência da coluna target nos dados de treino e teste
    if target_column not in train_data.columns or target_column not in test_data.columns:
        print(f"Erro: a coluna `{target_column}` não está presente nos dados de treino ou teste após a divisão.")
        return
    
    print(f"Coluna `{target_column}` presente nos dados de treino e teste após a divisão.")

    # Executa a função
    X_train, y_train, scaler, columns = preprocessa_dados(train_data, target_column, is_train=True)
    
    # Verificação do tamanho de y_train
    if y_train is None or len(y_train) == 0:
        print("Erro: `y_train` está vazio após o pré-processamento.")
        return
    
    ##### Etapa 4: Treinamento do Modelo #####
    print("\nExecutando a Etapa 4 - Treinamento do Modelo.\n")
    model = treina_modelo(X_train, y_train)
    
    # Salvando scaler e colunas do pré-processamento
    joblib.dump(scaler, scaler_path)
    joblib.dump(columns, columns_path)
    
    ##### Etapa 5: Avaliação do Modelo #####
    # Usamos o conjunto de teste para avaliação
    print("\nExecutando a Etapa 5 - Avaliação do Modelo.\n")
    X_test, y_test, _, _ = preprocessa_dados(test_data, target_column, is_train=False, scaler=scaler, columns=columns)

    # Retorna a avaliação
    if y_test is not None and len(y_test) > 0:
        accuracy, report = avalia_modelo(model, X_test, y_test)
        print(f"Acurácia do modelo: {accuracy}")
        print(f"Relatório de classificação:\n{report}")
    else:
        print("Erro: `y_test` está vazio, não é possível avaliar o modelo.")

    ##### Etapa 6: Deploy (usando nova massa de dados) #####
    print("Executando a Etapa 6 - Deploy do Modelo.\n")
    model_deployed, scaler_deployed, columns_deployed = carrega_modelo(model_path, scaler_path, columns_path)
    
    # Usando novos dados para as previsões
    if model_deployed and scaler_deployed and columns_deployed is not None and not columns_deployed.empty:
        new_data_path = "dados/novos_dados.csv"  
        new_data = extrai_dados(new_data_path)
        new_data_cleaned = limpa_dados(new_data)
        
        # Pré-processando os novos dados com o mesmo scaler e colunas do treinamento
        X_new, _, _, _ = preprocessa_dados(new_data_cleaned, target_column, is_train=False, scaler=scaler_deployed, columns=columns_deployed)
        
        # Fazendo previsões com os novos dados
        predictions = faz_previsoes(model_deployed, X_new)
        print(f"Previsões para novos dados: {predictions}")
    else:
        print("Erro: O modelo ou o pré-processamento não foram carregados corretamente.")

# Bloco de execução
if __name__ == "__main__":

    print("\nIniciando a Execução do Pipeline...\n")

    # Caminho dos dados históricos
    data_path = "dados/dados_historicos.csv"  

    # Coluna definida como variável alvo (target)
    target_column = "target"      

    # Executa o pipeline 
    rf_executa_pipeline(data_path, target_column)

    print("\nExecução do Pipeline Concluída com Sucesso!\n")




