# Automação do Pipeline de Machine Learning
# Módulo de Avaliação do Modelo

# Importa accuracy_score para calcular a acurácia do modelo
# Importa classification_report para gerar o relatório de classificação
from sklearn.metrics import accuracy_score, classification_report

# Define a função para avaliar o desempenho do modelo
def avalia_modelo(model, X_test, y_test):
    print("Iniciando avaliação do modelo...")

    # Faz previsões com o modelo usando os dados de teste
    predictions = model.predict(X_test)

    # Calcula a acurácia das previsões
    accuracy = accuracy_score(y_test, predictions)

    report = classification_report(y_test, predictions)
    print(f"Acurácia: {accuracy}")
    print(f"Relatório de Classificação:\n{report}")

    return accuracy, report
