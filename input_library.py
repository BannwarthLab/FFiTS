#!/bin/python

input_ff_calc = '''#This is a CREST input file
input='struc1.xyz'
runtype='ancopt'

#parallelization
threads = 4
#calculation data

[calculation]
type = 1   

# the two hessians defining the potential is defined here
[[calculation.level]]
method = "hesspot"
refgeo = "struc1.xyz"
refhessian = "hess1"
refwbo = "wbo1"
refforcefield = "force_field1_or"
useff = false

[[calculation.level]]
method = "hesspot"
refgeo = "struc2.xyz"
refhessian = "hess2"
refwbo = "wbo2"
refforcefield = "force_field2_or"
useff = false

'''

input_avff_optimize = '''#This is a CREST input file
input='start1.xyz'
runtype='ancopt'

#parallelization
threads = 4
#calculation data

[calculation]
type = 1   

# the two hessians defining the potential is defined here
[[calculation.level]]
method = "hesspot"
refgeo = "struc1.xyz"
refhessian = "hess1"
refwbo = "wbo1"
refforcefield = "avff"
useff = true
'''

input_ff_opt1 = '''#This is a CREST input file
input='struc1.xyz'
runtype='ancopt'

#parallelization
threads = 4
#calculation data

[calculation]
type = 1   

# the two hessians defining the potential is defined here
[[calculation.level]]
method = "hesspot"
refgeo = "struc1.xyz"
refhessian = "hess1"
refwbo = "wbo1"
refforcefield = "force_field1_mod"
useff = true

'''

input_ff_opt2 = '''#This is a CREST input file
input='struc2.xyz'
runtype='ancopt'

#parallelization
threads = 4
#calculation data

[calculation]
type = 1   

# the two hessians defining the potential is defined here
[[calculation.level]]
method = "hesspot"
refgeo = "struc2.xyz"
refhessian = "hess2"
refwbo = "wbo2"
refforcefield = "force_field2_mod"
useff = true

'''

input_avff1 = '''
#This is a CREST input file
input='struc1.xyz'
runtype='ancopt'

#parallelization
threads = 4
#calculation data

[calculation]
type = 3   # specify energy & gradient from [calculation.level] to be used
            # -1 is for MECPs
maxcycle=400
eprint = true
elog="energies.log"

# the two hessians defining the potential is defined here
[[calculation.level]]
method = "hesspot"
refgeo = "struc1.xyz"
refhessian = "hess1"
refwbo = "wbo1"
refforcefield = "ff1"
useff = false

[[calculation.level]]
method = "hesspot"
refgeo = "struc2.xyz"
refhessian = "hess2"
refwbo = "wbo2"
refforcefield = "ff2"
useff = false

[[calculation.level]]
method = "averageff"
refforcefield = "avff"
'''

input_avff2 = '''
#This is a CREST input file
input='struc2.xyz'
runtype='ancopt'

#parallelization
threads = 4
#calculation data

[calculation]
type = 3   # specify energy & gradient from [calculation.level] to be used
            # -1 is for MECPs
maxcycle=400
eprint = true
elog="energies.log"

# the two hessians defining the potential is defined here
[[calculation.level]]
method = "hesspot"
refgeo = "struc1.xyz"
refhessian = "hess1"
refwbo = "wbo1"
refforcefield = "ff1"
useff = false

[[calculation.level]]
method = "hesspot"
refgeo = "struc2.xyz"
refhessian = "hess2"
refwbo = "wbo2"
refforcefield = "ff2"
useff = false

[[calculation.level]]
method = "averageff"
refforcefield = "avff"
'''

input_start1 = '''
#This is a CREST input file
input='start1.xyz'
runtype='ancopt'

#parallelization
threads = 4
#calculation data

[calculation]
type = 3   # specify energy & gradient from [calculation.level] to be used
            # -1 is for MECPs
eprint = true
elog="energies.log"

# the two hessians defining the potential is defined here
[[calculation.level]]
method = "hesspot"
refgeo = "struc1.xyz"
refhessian = "hess1"
refwbo = "wbo1"
refforcefield = "ff1"
useff = false

[[calculation.level]]
method = "hesspot"
refgeo = "struc2.xyz"
refhessian = "hess2"
refwbo = "wbo2"
refforcefield = "ff2"
useff = false

[[calculation.level]]
method = "averageff"
refforcefield = "avff"
'''

input_start2 = '''
#This is a CREST input file
input='start2.xyz'
runtype='ancopt'

#parallelization
threads = 4
#calculation data

[calculation]
type = 3   # specify energy & gradient from [calculation.level] to be used
            # -1 is for MECPs
eprint = true
elog="energies.log"

# the two hessians defining the potential is defined here
[[calculation.level]]
method = "hesspot"
refgeo = "struc1.xyz"
refhessian = "hess1"
refwbo = "wbo1"
refforcefield = "ff1"
useff = false

[[calculation.level]]
method = "hesspot"
refgeo = "struc2.xyz"
refhessian = "hess2"
refwbo = "wbo2"
refforcefield = "ff2"
useff = false

[[calculation.level]]
method = "averageff"
refforcefield = "avff"
'''
