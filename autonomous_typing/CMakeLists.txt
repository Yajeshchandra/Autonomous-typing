cmake_minimum_required(VERSION 3.0.2)
project(autonomous_typing)

#####################################
## Compile Options and Requirements #
#####################################

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  rospy
  trajectory_msgs
  sensor_msgs 
  std_msgs
)

## Ensure Python3 is available with necessary components
find_package(Python3 REQUIRED COMPONENTS Interpreter NumPy)

######################################
## Check and Display Python Packages #
######################################

## Check for installed Python dependencies and display them
execute_process(
    COMMAND ${Python3_EXECUTABLE} -m pip show cv_bridge numpy-quaternion pinocchio quadprog pynput pyqt5
    OUTPUT_VARIABLE PYTHON_PACKAGES
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "Python dependencies: ${PYTHON_PACKAGES}")

###################################
## Catkin Specific Configuration ##
###################################

## Declare the catkin package
catkin_package(
  CATKIN_DEPENDS rospy trajectory_msgs sensor_msgs std_msgs
)

###################################
## Build Configuration and Setup ##
###################################

## Include directories
include_directories(
  ${catkin_INCLUDE_DIRS}
)

###############
## Installation ##
###############

## Mark Python scripts for installation
catkin_install_python(PROGRAMS
  src/keyboard_typing.py
  src/yolo_seg.py
  src/controller.py
  src/multicontroller.py

  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Install launch files
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
  FILES_MATCHING PATTERN "*.launch"
)

## Install configuration files (e.g., YAML files)
install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
  FILES_MATCHING PATTERN "*.yaml"
)

## Install URDF files
install(DIRECTORY urdf/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/urdf
)

#################
## Testing (Optional) ##
#################

## Add tests if applicable
# catkin_add_gtest(${PROJECT_NAME}-test test/test_autonomous_typing.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${catkin_LIBRARIES})
# endif()
