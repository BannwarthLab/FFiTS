from src.io.toml_parser import load_config
import os

def test_load_config_no_second_file():
    config = load_config()
    print(config) 
    assert config["system"]["charge"] == 0

def test_load_config_with_second_file():
    path = os.path.join(os.getcwd(), 'tests', 'examples', 'small_single_molecule', 'testconfig.toml')
    config = load_config(path)
    print(config) 
    assert config["system"]["charge"] == -2
    


    