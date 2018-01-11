# - Try to find the chull library.
# Input variables:
#     CHULL_ROOT - chull installation root
#
# Output variables:
#     CHULL_INCLUDE_DIR - include directories
#     CHULL_FOUND - is 'YES' if library found, 'NO' otherwise

set(CHULL_FOUND YES)

find_path(
    CHULL_INCLUDE_DIR
    NAMES
    "chull/chull.h"
    HINTS
    "${CHULL_ROOT}"
    ENV
    "${chull_DIR}"
    PATH_SUFFIXES
    "include"
)

if(NOT CHULL_INCLUDE_DIR)
  set(CHULL_FOUND NO)
  message(STATUS "chull NOT found, please specify CHULL_ROOT.")
else()
    message(STATUS "Found chull: ${CHULL_INCLUDE_DIR}")
endif()

# No need to find other libraries yet.
