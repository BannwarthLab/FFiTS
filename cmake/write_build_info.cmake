# Writes build_info.txt (build time, compilers, build type) next to the f2py module.
# Run at build time (POST_BUILD) so the timestamp belongs to the actual compilation;
# read by ffits/io/print/header.py and printed in the program header.
#
# Expected variables: OUT, FC_ID, FC_VERSION, C_ID, C_VERSION, BUILD_TYPE, FFLAGS
string(TIMESTAMP BUILD_TIME "%Y-%m-%d %H:%M")
file(WRITE "${OUT}"
  "build_date=${BUILD_TIME}\n"
  "fortran_compiler=${FC_ID} ${FC_VERSION}\n"
  "c_compiler=${C_ID} ${C_VERSION}\n"
  "build_type=${BUILD_TYPE}\n"
  "fortran_flags=${FFLAGS}\n"
)
