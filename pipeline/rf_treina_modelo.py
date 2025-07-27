# Automação do Pipeline de Machine Learning
# Módulo de Treinamento do Modelo

from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Define a função para treinar o modelo usando os dados de treino
def treina_modelo(X_train, y_train):

    print("Iniciando treinamento do modelo...")

    # Cria uma instância do RandomForestClassifier
    model = RandomForestClassifier()

    # Treina o modelo com os dados de treino fornecidos
    model.fit(X_train, y_train)

    model_path = os.path.join("modelos", "random_forest_model.pkl")

    joblib.dump(model, model_path)

    print(f"Modelo salvo em {model_path}")
    
    return model
