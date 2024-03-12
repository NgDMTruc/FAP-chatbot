import os
import yaml

def load_configs():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(base_dir)
    config_path = os.path.join(parent_dir, 'configs/config.yaml')
    
    with open(config_path, "r") as config_file:
        file = yaml.safe_load(config_file)

    return file

