# __init__.py
import logging

# Configuração global de logs
logging.basicConfig(level=logging.INFO)

# Importa os módulos
from .rf_extrai_dados import extrai_dados
from .rf_limpa_dados import limpa_dados
from .rf_processa_dados import preprocessa_dados
from .rf_treina_modelo import treina_modelo
from .rf_avalia_modelo import avalia_modelo
from .rf_deploy import carrega_modelo, faz_previsoes

# Estabelece a hierarquia
__all__ = [
    "extrai_dados",
    "limpa_dados",
    "preprocessa_dados",
    "treina_modelo",
    "avalia_modelo",
    "carrega_modelo",
    "faz_previsoes",
]
