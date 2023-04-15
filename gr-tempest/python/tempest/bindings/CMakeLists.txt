# Copyright 2020 Free Software Foundation, Inc.
#
# This file is part of GNU Radio
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

########################################################################
# Check if there is C++ code at all
########################################################################
if(NOT tempest_sources)
    MESSAGE(STATUS "No C++ sources... skipping python bindings")
    return()
endif(NOT tempest_sources)

########################################################################
# Check for pygccxml
########################################################################
GR_PYTHON_CHECK_MODULE_RAW(
    "pygccxml"
    "import pygccxml"
    PYGCCXML_FOUND
    )

include(GrPybind)

########################################################################
# Python Bindings
########################################################################
list(APPEND tempest_python_files
    Hsync_python.cc
    fft_peak_fine_sampling_sync_python.cc
    fine_sampling_synchronization_python.cc
    frame_drop_python.cc
    framing_python.cc
    infer_screen_resolution_python.cc
    normalize_flow_python.cc
    sampling_synchronization_python.cc
    ssamp_correction_python.cc
    sync_detector_python.cc python_bindings.cc)

GR_PYBIND_MAKE_OOT(tempest
   ../../..
   gr::tempest
   "${tempest_python_files}")

# copy in bindings .so file for use in QA test module
add_custom_target(
  copy_bindings_for_tests ALL
  COMMAND
    ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_BINARY_DIR}/*.so"
    ${CMAKE_BINARY_DIR}/test_modules/gnuradio/tempest/
  DEPENDS tempest_python)

install(TARGETS tempest_python DESTINATION ${GR_PYTHON_DIR}/gnuradio/tempest COMPONENT pythonapi)
