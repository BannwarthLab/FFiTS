from src.datatype.structure_data import ForceField
from src.forcefield.setup import setup_ff


# pseudo code for complete generation 
def main():
    setup_ff(reactant)  # this setup could be done together with all the other readin of xyz, wbo calc and so on
    setup_ff(product)

    parameterize_ff(reactant)
    parameterize_ff(product)

    tsff: ForceField = create_tsff(reactant, product)

    ts_optimization(starting_structure: reactant, potential: tsff)
