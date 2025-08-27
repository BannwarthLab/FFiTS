cmake_minimum_required(VERSION 3.16.0)

project(pyTSguessF C Fortran)

option(SKBUILD "Should be ON of being build by skbuild, 
and OFF of being build by regular cmake" OFF)

if(SKBUILD)
  message(STATUS "The project is built using scikit-build")
endif()

if (NOT SKBUILD)
  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/../cmake/")
endif()

if (NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()

find_package(PythonExtensions REQUIRED)
find_package(NumPy)
find_package(F2PY REQUIRED)

message(STATUS "Cmake build directory: ${CMAKE_BINARY_DIR}$")
message(STATUS "F2PY included from: ${F2PY_INCLUDE_DIR}")

#
# Force field

# Build libary
add_library(fortran_helper STATIC "${CMAKE_SOURCE_DIR}/src/force_field/fortran_helper.f90")
set_property(TARGET fortran_helper PROPERTY POSITION_INDEPENDENT_CODE ON)
# build libary
add_library(a_angle_derivatives STATIC "${CMAKE_SOURCE_DIR}/src/force_field/a_angle_derivatives.f90")
set_property(TARGET a_angle_derivatives PROPERTY POSITION_INDEPENDENT_CODE ON)
# build libary
add_library(b_angle_derivatives STATIC "${CMAKE_SOURCE_DIR}/src/force_field/b_angle_derivatives.f90")
set_property(TARGET b_angle_derivatives PROPERTY POSITION_INDEPENDENT_CODE ON)
# build libary
add_library(c_angle_derivatives STATIC "${CMAKE_SOURCE_DIR}/src/force_field/c_angle_derivatives.f90")
set_property(TARGET c_angle_derivatives PROPERTY POSITION_INDEPENDENT_CODE ON)
# build libary
add_library(a_dihedral_derivatives STATIC "${CMAKE_SOURCE_DIR}/src/force_field/a_dihedral_derivatives.f90")
set_property(TARGET a_dihedral_derivatives PROPERTY POSITION_INDEPENDENT_CODE ON)
# build libary
add_library(b_dihedral_derivatives STATIC "${CMAKE_SOURCE_DIR}/src/force_field/b_dihedral_derivatives.f90")
set_property(TARGET b_dihedral_derivatives PROPERTY POSITION_INDEPENDENT_CODE ON)
# build libary
add_library(c_dihedral_derivatives STATIC "${CMAKE_SOURCE_DIR}/src/force_field/c_dihedral_derivatives.f90")
set_property(TARGET c_dihedral_derivatives PROPERTY POSITION_INDEPENDENT_CODE ON)
# build libary
add_library(d_dihedral_derivatives STATIC "${CMAKE_SOURCE_DIR}/src/force_field/d_dihedral_derivatives.f90")
set_property(TARGET d_dihedral_derivatives PROPERTY POSITION_INDEPENDENT_CODE ON)
# build libary
add_library(inv_rij_derivatives STATIC "${CMAKE_SOURCE_DIR}/src/force_field/inv_rij_derivatives.f90")
set_property(TARGET inv_rij_derivatives PROPERTY POSITION_INDEPENDENT_CODE ON)
# build libary
add_library(bond_derivatives STATIC "${CMAKE_SOURCE_DIR}/src/force_field/bond_derivatives.f90")
set_property(TARGET bond_derivatives PROPERTY POSITION_INDEPENDENT_CODE ON)
# build libary
add_library(angle_derivatives STATIC "${CMAKE_SOURCE_DIR}/src/force_field/angle_derivatives.f90")
set_property(TARGET angle_derivatives PROPERTY POSITION_INDEPENDENT_CODE ON)
# build libary
add_library(dihedral_derivatives STATIC "${CMAKE_SOURCE_DIR}/src/force_field/dihedral_derivatives.f90")
set_property(TARGET dihedral_derivatives PROPERTY POSITION_INDEPENDENT_CODE ON)
# build libary
add_library(repulsion_derivatives STATIC "${CMAKE_SOURCE_DIR}/src/force_field/repulsion_derivatives.f90")
set_property(TARGET repulsion_derivatives PROPERTY POSITION_INDEPENDENT_CODE ON)
#build libary
add_library(analytical_derivatives STATIC "${CMAKE_SOURCE_DIR}/src/force_field/analytical_derivatives.f90")
set_property(TARGET analytical_derivatives PROPERTY POSITION_INDEPENDENT_CODE ON)
#build libary
add_library(force_field_energy STATIC "${CMAKE_SOURCE_DIR}/src/force_field/energy.f90")
set_property(TARGET force_field_energy PROPERTY POSITION_INDEPENDENT_CODE ON)

add_library(lj_derivatives STATIC "${CMAKE_SOURCE_DIR}/src/force_field/lj_derivatives.f90")
set_property(TARGET force_field_energy PROPERTY POSITION_INDEPENDENT_CODE ON)


target_link_libraries(a_angle_derivatives fortran_helper)
target_link_libraries(b_angle_derivatives fortran_helper)
target_link_libraries(c_angle_derivatives fortran_helper b_angle_derivatives)
target_link_libraries(a_dihedral_derivatives fortran_helper)
target_link_libraries(b_dihedral_derivatives fortran_helper)
target_link_libraries(c_dihedral_derivatives fortran_helper b_dihedral_derivatives)
target_link_libraries(d_dihedral_derivatives fortran_helper)
target_link_libraries(inv_rij_derivatives fortran_helper)
target_link_libraries(bond_derivatives fortran_helper)
target_link_libraries(angle_derivatives fortran_helper a_angle_derivatives c_angle_derivatives b_angle_derivatives)
target_link_libraries(dihedral_derivatives fortran_helper a_dihedral_derivatives c_dihedral_derivatives b_dihedral_derivatives  d_dihedral_derivatives)
target_link_libraries(repulsion_derivatives fortran_helper inv_rij_derivatives)
target_link_libraries(angle_derivatives fortran_helper a_angle_derivatives c_angle_derivatives b_angle_derivatives)
target_link_libraries(bond_derivatives fortran_helper b_angle_derivatives)
target_link_libraries(lj_derivatives fortran_helper b_angle_derivatives)
target_link_libraries(dihedral_derivatives fortran_helper a_dihedral_derivatives c_dihedral_derivatives b_dihedral_derivatives  d_dihedral_derivatives)
target_link_libraries(repulsion_derivatives fortran_helper inv_rij_derivatives)
target_link_libraries(analytical_derivatives fortran_helper bond_derivatives angle_derivatives dihedral_derivatives repulsion_derivatives)
target_link_libraries(force_field_energy fortran_helper)

# Create the F2PY module directly from ancopt_optimizer.f90
set(f2py_module_name "fortran_forcefield")
# set(fortran_src_file "${CMAKE_SOURCE_DIR}/molbar/force_field/ancopt_optimizer.f90")
set(generated_module_file ${f2py_module_name}${PYTHON_EXTENSION_MODULE_SUFFIX})

add_custom_target(${f2py_module_name} ALL
  DEPENDS "${fortran_src_file}"
  )

add_custom_command(
  OUTPUT "${f2py_module_name}module.c" "${f2py_module_name}-f2pywrappers2.f90"
  COMMAND ${F2PY_EXECUTABLE}
    -m ${f2py_module_name}
    ${fortran_src_file}
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${fortran_src_file}
  )

add_library(${generated_module_file} MODULE
            "${f2py_module_name}module.c"
            "${F2PY_INCLUDE_DIR}/fortranobject.c"
            "${f2py_module_name}-f2pywrappers2.f90"
             ${fortran_src_file})

# Link against all the derivative libraries that ancopt_optimizer needs
# target_link_libraries(${generated_module_file} fortran_helper bond_derivatives angle_derivatives dihedral_derivatives repulsion_derivatives force_field_energy analytical_derivatives)

target_include_directories(${generated_module_file} PUBLIC
                           ${F2PY_INCLUDE_DIRS}
                           ${PYTHON_INCLUDE_DIRS})
set_target_properties(${generated_module_file} PROPERTIES SUFFIX "")
set_target_properties(${generated_module_file} PROPERTIES PREFIX "")


if (UNIX)
  if (APPLE)
    set_target_properties(${generated_module_file} PROPERTIES
    LINK_FLAGS  '-Wl,-dylib,-undefined,dynamic_lookup')
  else()
    set_target_properties(${generated_module_file} PROPERTIES
  LINK_FLAGS  '-Wl,--allow-shlib-undefined')
  endif()
endif()


if (SKBUILD)
  install(TARGETS ${generated_module_file} DESTINATION src/force_field/)
else()
  install(TARGETS ${generated_module_file} DESTINATION src/force_field/)
endif()

