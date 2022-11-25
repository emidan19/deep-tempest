#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "gnuradio::gnuradio-tempest" for configuration "Release"
set_property(TARGET gnuradio::gnuradio-tempest APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(gnuradio::gnuradio-tempest PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/x86_64-linux-gnu/libgnuradio-tempest.so.17a754aa"
  IMPORTED_SONAME_RELEASE "libgnuradio-tempest.so.1.0.0git"
  )

list(APPEND _IMPORT_CHECK_TARGETS gnuradio::gnuradio-tempest )
list(APPEND _IMPORT_CHECK_FILES_FOR_gnuradio::gnuradio-tempest "${_IMPORT_PREFIX}/lib/x86_64-linux-gnu/libgnuradio-tempest.so.17a754aa" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
