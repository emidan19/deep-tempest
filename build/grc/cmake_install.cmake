# Install script for directory: /home/emidan19/gr-tempest/grc

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
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/gnuradio/grc/blocks" TYPE FILE FILES
    "/home/emidan19/gr-tempest/grc/tempest_sampling_synchronization.block.yml"
    "/home/emidan19/gr-tempest/grc/tempest_framing.block.yml"
    "/home/emidan19/gr-tempest/grc/tempest_Hsync.block.yml"
    "/home/emidan19/gr-tempest/grc/tempest_image_source.block.yml"
    "/home/emidan19/gr-tempest/grc/tempest_normalize_flow.block.yml"
    "/home/emidan19/gr-tempest/grc/tempest_fine_sampling_synchronization.block.yml"
    "/home/emidan19/gr-tempest/grc/tempest_sync_detector.block.yml"
    "/home/emidan19/gr-tempest/grc/tempest_frame_drop.block.yml"
    "/home/emidan19/gr-tempest/grc/tempest_message_to_var.block.yml"
    "/home/emidan19/gr-tempest/grc/tempest_fft_peak_fine_sampling_sync.block.yml"
    "/home/emidan19/gr-tempest/grc/tempest_infer_screen_resolution.block.yml"
    "/home/emidan19/gr-tempest/grc/tempest_tempest_msgbtn.block.yml"
    "/home/emidan19/gr-tempest/grc/tempest_ssamp_correction.block.yml"
    "/home/emidan19/gr-tempest/grc/tempest_TMDS_image_source.block.yml"
    "/home/emidan19/gr-tempest/grc/tempest_binary_serializer.block.yml"
    "/home/emidan19/gr-tempest/grc/tempest_TMDS_decoder.block.yml"
    )
endif()

