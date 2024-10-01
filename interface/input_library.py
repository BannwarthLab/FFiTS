#!/bin/python


from input_generator_crest import HeaderBlock 
from input_generator_crest import CalcBlock 
from input_generator_crest import SsffBlock
from input_generator_crest import TsffBlock 
from input_generator_crest import generate_input



"""
In this file I am using crest-input_generator to construct the necessary inputs for separate usecases 
"""
def input_ff_optimization(starting_struc: str):
    return generate_input(header=HeaderBlock(input='struc1.xyz', runtype='ancopt', threads=4),
                        calculation=CalcBlock(id=3, maxcycle=219),
                        ssff1=SsffBlock(
                            method='ssff', 
                            refgeo='struc2.xyz', 
                            refhessian='hess2', 
                            refwbo='wbo2', 
                            refff='ff2', 
                            useff=False),
                        ssff2=SsffBlock(
                            method='ssff', refgeo='struc2.xyz', refhessian='hess2', refwbo='wbo2', refff='ff2', useff=False),
                        tsff=TsffBlock(method='tsff', refff='tsff.txt', id1=1, id2=2))


# input_t = generate_input(header=HeaderBlock(input='struc2.xyz', runtype='ancopt', threads=4),
#                                     calculation=CalcBlock(id=3, maxcycle=219),
#                                     ssff1=SsffBlock(method='ssff', refgeo='struc1.xyz', refhessian='hess1', refwbo='wbo1', refff='ff1', useff=False),
#                                     ssff2=SsffBlock(method='ssff', refgeo='struc2.xyz', refhessian='hess2', refwbo='wbo2', refff='ff2', useff=False),
#                                     tsff=TsffBlock(method='tsff', refff='tsff.txt', id1=1, id2=2))
# input_ff_calc = '''#This is a CREST input file
# input='struc1.xyz'
# runtype='ancopt'

# #parallelization
# threads = 4
# #calculation data

# [calculation]
# type = 1   

# # the two hessians defining the potential is defined here
# [[calculation.level]]
# method = "hesspot"
# refgeo = "struc1.xyz"
# refhessian = "hess1"
# refwbo = "wbo1"
# refforcefield = "force_field1_or"
# useff = false

# [[calculation.level]]
# method = "hesspot"
# refgeo = "struc2.xyz"
# refhessian = "hess2"
# refwbo = "wbo2"
# refforcefield = "force_field2_or"
# useff = false

# '''

# input_avff_optimize = '''#This is a CREST input file
# input='start1.xyz'
# runtype='ancopt'

# #parallelization
# threads = 4
# #calculation data

# [calculation]
# type = 1   

# # the two hessians defining the potential is defined here
# [[calculation.level]]
# method = "hesspot"
# refgeo = "struc1.xyz"
# refhessian = "hess1"
# refwbo = "wbo1"
# refforcefield = "avff"
# useff = true
# '''

# input_ff_opt1 = '''#This is a CREST input file
# input='struc1.xyz'
# runtype='ancopt'

# #parallelization
# threads = 4
# #calculation data

# [calculation]
# type = 1   

# # the two hessians defining the potential is defined here
# [[calculation.level]]
# method = "hesspot"
# refgeo = "struc1.xyz"
# refhessian = "hess1"
# refwbo = "wbo1"
# refforcefield = "force_field1_mod"
# useff = true

# '''

# input_ff_opt2 = '''#This is a CREST input file
# input='struc2.xyz'
# runtype='ancopt'

# #parallelization
# threads = 4
# #calculation data

# [calculation]
# type = 1   

# # the two hessians defining the potential is defined here
# [[calculation.level]]
# method = "hesspot"
# refgeo = "struc2.xyz"
# refhessian = "hess2"
# refwbo = "wbo2"
# refforcefield = "force_field2_mod"
# useff = true

# '''

# input_avff1 = '''
# #This is a CREST input file
# input='struc1.xyz'
# runtype='ancopt'

# #parallelization
# threads = 4
# #calculation data

# [calculation]
# type = 3   # specify energy & gradient from [calculation.level] to be used
#             # -1 is for MECPs
# maxcycle=400
# eprint = true
# elog="energies.log"

# # the two hessians defining the potential is defined here
# [[calculation.level]]
# method = "hesspot"
# refgeo = "struc1.xyz"
# refhessian = "hess1"
# refwbo = "wbo1"
# refforcefield = "ff1"
# useff = false

# [[calculation.level]]
# method = "hesspot"
# refgeo = "struc2.xyz"
# refhessian = "hess2"
# refwbo = "wbo2"
# refforcefield = "ff2"
# useff = false

# [[calculation.level]]
# method = "averageff"
# refforcefield = "avff"
# '''

# input_avff2 = '''
# #This is a CREST input file
# input='struc2.xyz'
# runtype='ancopt'

# #parallelization
# threads = 4
# #calculation data

# [calculation]
# type = 3   # specify energy & gradient from [calculation.level] to be used
#             # -1 is for MECPs
# maxcycle=400
# eprint = true
# elog="energies.log"

# # the two hessians defining the potential is defined here
# [[calculation.level]]
# method = "hesspot"
# refgeo = "struc1.xyz"
# refhessian = "hess1"
# refwbo = "wbo1"
# refforcefield = "ff1"
# useff = false

# [[calculation.level]]
# method = "hesspot"
# refgeo = "struc2.xyz"
# refhessian = "hess2"
# refwbo = "wbo2"
# refforcefield = "ff2"
# useff = false

# [[calculation.level]]
# method = "averageff"
# refforcefield = "avff"
# '''

# input_start1 = '''
# #This is a CREST input file
# input='start1.xyz'
# runtype='ancopt'

# #parallelization
# threads = 4
# #calculation data

# [calculation]
# type = 3   # specify energy & gradient from [calculation.level] to be used
#             # -1 is for MECPs
# eprint = true
# elog="energies.log"

# # the two hessians defining the potential is defined here
# [[calculation.level]]
# method = "hesspot"
# refgeo = "struc1.xyz"
# refhessian = "hess1"
# refwbo = "wbo1"
# refforcefield = "ff1"
# useff = false

# [[calculation.level]]
# method = "hesspot"
# refgeo = "struc2.xyz"
# refhessian = "hess2"
# refwbo = "wbo2"
# refforcefield = "ff2"
# useff = false

# [[calculation.level]]
# method = "averageff"
# refforcefield = "avff"
# '''

# input_start2 = '''
# #This is a CREST input file
# input='start2.xyz'
# runtype='ancopt'

# #parallelization
# threads = 4
# #calculation data

# [calculation]
# type = 3   # specify energy & gradient from [calculation.level] to be used
#             # -1 is for MECPs
# eprint = true
# elog="energies.log"

# # the two hessians defining the potential is defined here
# [[calculation.level]]
# method = "hesspot"
# refgeo = "struc1.xyz"
# refhessian = "hess1"
# refwbo = "wbo1"
# refforcefield = "ff1"
# useff = false

# [[calculation.level]]
# method = "hesspot"
# refgeo = "struc2.xyz"
# refhessian = "hess2"
# refwbo = "wbo2"
# refforcefield = "ff2"
# useff = false

# [[calculation.level]]
# method = "averageff"
# refforcefield = "avff"
# '''
