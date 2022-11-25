# Install script for directory: /home/emidan19/gr-tempest/include/tempest

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/tempest" TYPE FILE FILES
    "/home/emidan19/gr-tempest/include/tempest/api.h"
    "/home/emidan19/gr-tempest/include/tempest/sampling_synchronization.h"
    "/home/emidan19/gr-tempest/include/tempest/framing.h"
    "/home/emidan19/gr-tempest/include/tempest/Hsync.h"
    "/home/emidan19/gr-tempest/include/tempest/normalize_flow.h"
    "/home/emidan19/gr-tempest/include/tempest/fine_sampling_synchronization.h"
    "/home/emidan19/gr-tempest/include/tempest/sync_detector.h"
    "/home/emidan19/gr-tempest/include/tempest/frame_drop.h"
    "/home/emidan19/gr-tempest/include/tempest/fft_peak_fine_sampling_sync.h"
    "/home/emidan19/gr-tempest/include/tempest/infer_screen_resolution.h"
    "/home/emidan19/gr-tempest/include/tempest/ssamp_correction.h"
    )
endif()

