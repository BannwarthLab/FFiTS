# This file is part of crest.
# SPDX-Identifier: LGPL-3.0-or-later
#
# crest is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# crest is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with crest.  If not, see <https://www.gnu.org/licenses/>.

set(dir "${CMAKE_CURRENT_SOURCE_DIR}")

list(APPEND srcs
  "${dir}/fortran_helper.f90"
  "${dir}/a_angle_derivatives.f90"
  "${dir}/a_dihedral_derivatives.f90"
  "${dir}/b_angle_derivatives.f90"
  "${dir}/b_dihedral_derivatives.f90"
  "${dir}/c_angle_derivatives.f90"
  "${dir}/c_dihedral_derivatives.f90"
  "${dir}/d_dihedral_derivatives.f90"
  "${dir}/angle_derivatives.f90"
  "${dir}/bond_derivatives.f90"
  "${dir}/lj_derivatives.f90"
  "${dir}/dihedral_derivatives.f90"
  "${dir}/energy.f90"
  "${dir}/inv_rij_derivatives.f90"
  "${dir}/repulsion_derivatives.f90"
  "${dir}/analytical_derivatives.f90"
  "${dir}/ff_utility.f90"
  "${dir}/ff_interface.f90"
  "${dir}/grad_derivatives.f90"
  "${dir}/numerical_derivatives.f90"
)

set(srcs ${srcs} PARENT_SCOPE)






