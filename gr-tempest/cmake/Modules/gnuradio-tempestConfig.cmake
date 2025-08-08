find_package(PkgConfig)

PKG_CHECK_MODULES(PC_GR_TEMPEST gnuradio-tempest)

FIND_PATH(
    GR_TEMPEST_INCLUDE_DIRS
    NAMES gnuradio/tempest/api.h
    HINTS $ENV{TEMPEST_DIR}/include
        ${PC_TEMPEST_INCLUDEDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/include
          /usr/local/include
          /usr/include
)

FIND_LIBRARY(
    GR_TEMPEST_LIBRARIES
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

include("${CMAKE_CURRENT_LIST_DIR}/gnuradio-tempestTarget.cmake")

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(GR_TEMPEST DEFAULT_MSG GR_TEMPEST_LIBRARIES GR_TEMPEST_INCLUDE_DIRS)
MARK_AS_ADVANCED(GR_TEMPEST_LIBRARIES GR_TEMPEST_INCLUDE_DIRS)
