INCLUDE(FindPkgConfig)
PKG_CHECK_MODULES(PC_TEMPEST tempest)

FIND_PATH(
    TEMPEST_INCLUDE_DIRS
    NAMES tempest/api.h
    HINTS $ENV{TEMPEST_DIR}/include
        ${PC_TEMPEST_INCLUDEDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/include
          /usr/local/include
          /usr/include
)

FIND_LIBRARY(
    TEMPEST_LIBRARIES
    NAMES gnuradio-tempest
    HINTS $ENV{TEMPEST_DIR}/lib
        ${PC_TEMPEST_LIBDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/lib
          ${CMAKE_INSTALL_PREFIX}/lib64
          /usr/local/lib
          /usr/local/lib64
          /usr/lib
          /usr/lib64
          )

include("${CMAKE_CURRENT_LIST_DIR}/tempestTarget.cmake")

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(TEMPEST DEFAULT_MSG TEMPEST_LIBRARIES TEMPEST_INCLUDE_DIRS)
MARK_AS_ADVANCED(TEMPEST_LIBRARIES TEMPEST_INCLUDE_DIRS)
