#!/bin/python


from ffits.external.input_generator_crest import HeaderBlock 
from ffits.external.input_generator_crest import CalcBlock 
from ffits.external.input_generator_crest import SsffBlock
from ffits.external.input_generator_crest import TsffBlock 
from ffits.external.input_generator_crest import generate_input
from ffits.datatype.calculation_data import CalculationParameters
from ffits.datatype.structure_data import *


"""
In this file I am using crest-input_generator to construct the necessary inputs for separate usecases 
"""
def input_ts_search(starting_struc: str, c: CalculationParameters):
    return generate_input(header=HeaderBlock(input=starting_struc,
                                        runtype='ancopt', 
                                        threads=4),
                        calculation=CalcBlock(id=3, 
                                        maxcycle=219),
                        ssff1=SsffBlock(method='ssff', 
                                        refgeo=c.struc1.xyz_filename, 
                                        refhessian=c.struc1.hessian_filename, 
                                        refwbo=c.struc1.wbo_filename, 
                                        refff=c.struc1.ff_filename, 
                                        useff=True),
                        ssff2=SsffBlock(method='ssff', 
                                        refgeo=c.struc2.xyz_filename, 
                                        refhessian=c.struc2.hessian_filename, 
                                        refwbo=c.struc2.wbo_filename, 
                                        refff=c.struc2.ff_filename, 
                                        useff=True),
                        tsff=TsffBlock(method='tsff', 
                                        refff=c.ts.ff_filename, 
                                        id1=c.struc1.id, 
                                        id2=c.struc2.id))

def input_ff_optimization(starting_struc: str, useff: bool, struc: StructurePath):
    return generate_input(header=HeaderBlock(input=starting_struc, 
                                        runtype='ancopt', 
                                        threads=4),
                        calculation=CalcBlock(id=1, 
                                        maxcycle=219),
                        ssff1=SsffBlock(method='ssff', 
                                        refgeo=struc.xyz_filename, 
                                        refhessian=struc.hessian_filename, 
                                        refwbo=struc.wbo_filename, 
                                        refff=struc.ff_filename, 
                                        useff=useff))


