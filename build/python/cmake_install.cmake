# Install script for directory: /home/emidan19/gr-tempest/python

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
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/python3/dist-packages/tempest" TYPE FILE FILES
    "/home/emidan19/gr-tempest/python/__init__.py"
    "/home/emidan19/gr-tempest/python/image_source.py"
    "/home/emidan19/gr-tempest/python/message_to_var.py"
    "/home/emidan19/gr-tempest/python/tempest_msgbtn.py"
    "/home/emidan19/gr-tempest/python/TMDS_image_source.py"
    "/home/emidan19/gr-tempest/python/binary_serializer.py"
    "/home/emidan19/gr-tempest/python/TMDS_decoder.py"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/python3/dist-packages/tempest" TYPE FILE FILES
    "/home/emidan19/gr-tempest/build/python/__init__.pyc"
    "/home/emidan19/gr-tempest/build/python/image_source.pyc"
    "/home/emidan19/gr-tempest/build/python/message_to_var.pyc"
    "/home/emidan19/gr-tempest/build/python/tempest_msgbtn.pyc"
    "/home/emidan19/gr-tempest/build/python/TMDS_image_source.pyc"
    "/home/emidan19/gr-tempest/build/python/binary_serializer.pyc"
    "/home/emidan19/gr-tempest/build/python/TMDS_decoder.pyc"
    "/home/emidan19/gr-tempest/build/python/__init__.pyo"
    "/home/emidan19/gr-tempest/build/python/image_source.pyo"
    "/home/emidan19/gr-tempest/build/python/message_to_var.pyo"
    "/home/emidan19/gr-tempest/build/python/tempest_msgbtn.pyo"
    "/home/emidan19/gr-tempest/build/python/TMDS_image_source.pyo"
    "/home/emidan19/gr-tempest/build/python/binary_serializer.pyo"
    "/home/emidan19/gr-tempest/build/python/TMDS_decoder.pyo"
    )
endif()

