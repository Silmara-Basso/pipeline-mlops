# pipeline-mlops
Automação do Pipeline de Machine Learning
Vamos prever se um cliente vai ou não comprar o produto.

# Automação do Pipeline de Machine Learning

Cada parte do projeto será representada por um módulo Python e, por fim, um script principal (random_forest.py) irá orquestrar todo o pipeline. Aqui está uma visão geral dos scripts e diretórios:

````

pipeline/
│
├── dados/                         # Dados brutos e processados
├── modelos/                       # Modelos treinados
├── pipeline/
│   ├── __init__.py                # Inicializador do módulo pipeline
│   ├── rf_extrai_dados.py        # Extração de dados
│   ├── rf_limpa_dados.py         # Limpeza de dados
│   ├── rf_processa_dados.py      # Pré-processamento
│   ├── rf_treina_modelo.py       # Treinamento do modelo
│   ├── rf_avalia_modelo.py       # Avaliação do modelo
│   └── rf_deploy.py              # Deploy do modelo
│
└── random_forest.py                    # Script principal para execução do pipeline
````

## Passos para execução do projeto:

### Crie um ambiente virtual:
````
python -m venv rfvenv
````

### Ative o ambiente virtual:
```
source rfvenv/bin/activate
```

### Execute as instruções abaixo conforme demonstrado nas aulas:

````
pip install --upgrade pip

pip install -r requirements.txt

python random_forest.py
````

# Verifique os resultados.