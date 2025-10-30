from src.io.toml_parser import load_config, load_calculation_data
import pytest
import os
from src.datatype.calculation_data import CalculationData
from src.io.toml_parser import load_calculation_data, overwrite_from_commandline
from src.io.commandline_parser import parse_args

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
    
def test_overwrite_from_commandline_overwrites(capsys):
    cd = CalculationData()
    # initially 0, 1
    overwrite_from_commandline(cd, multiplicity=2, charge=-1)
    captured = capsys.readouterr()
    assert cd.system.multiplicity == 2
    assert cd.system.charge == -1
    # check info messages printed
    assert "Multiplicity" in captured.out
    assert "Charge" in captured.out

def test_overwrite_from_commandline_no_change(capsys):
    cd = CalculationData()
    overwrite_from_commandline(cd, multiplicity=1, charge=0)
    captured = capsys.readouterr()
    # no message expected because nothing changed
    assert captured.out.strip() == ""

def test_overwrite_from_commandline_partial(capsys):
    cd = CalculationData()
    overwrite_from_commandline(cd, multiplicity=None, charge=-2)
    captured = capsys.readouterr()
    assert cd.system.multiplicity == 1  # unchanged
    assert cd.system.charge == -2
    assert "Charge" in captured.out
