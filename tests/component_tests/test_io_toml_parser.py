from src.io.toml_parser import load_config, load_calculation_data
import pytest
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
    
def test_load_calculation_data_no_second_file():
    config = load_calculation_data()
    print(config) 
    assert config.system.charge == 0
    assert config.postprocessing.relaxation == "None"
    
    
def test_load_calculation_data_with_second_file():
    path = os.path.join(os.getcwd(), 'tests', 'examples', 'small_single_molecule', 'testconfig.toml')
    config = load_calculation_data(path)
    print(config) 
    assert config.system.charge == -2
    assert config.postprocessing.relaxation == "None"
        
    
def test_load_calculation_data_valueerror():
    path = os.path.join(os.getcwd(), 'tests', 'examples', 'small_single_molecule', 'testconfig_wrong.toml')
    
    with pytest.raises(ValueError):
        config = load_calculation_data(path)
    