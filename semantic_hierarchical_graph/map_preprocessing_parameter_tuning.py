import yaml
from sklearn.model_selection import ParameterGrid
from semantic_hierarchical_graph import map_preprocessing

# Laden der Konfigurationsdatei
with open("config\\map_preprocessing.yaml") as f:
    config = yaml.safe_load(f)

# Erstellen einer Liste von Parameterkombinationen
param_grid = {'param1': config['param1_values'],
              'param2': config['param2_values'],
              'param3': config['param3_values']}

params_list = list(ParameterGrid(param_grid))

# Iterieren durch jede Parameterkombination
for params in params_list:
    # Erstellen eines Parameterobjekts mit der aktuellen Parameterkombination
    param_obj = Parameter(params)

    # Führen Sie den Code mit der aktuellen Parameterkombination aus
    result = test_all_imgs(show_img=False)(param_obj)

    # Speichern Sie das Ergebnis für die aktuelle Parameterkombination
    save_result(result, params)
