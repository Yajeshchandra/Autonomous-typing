cmake_minimum_required(VERSION 3.0.2)
project(Arm_Urdf)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  urdf
  xacro
  geometry_msgs
  trajectory_msgs
)

## Ensure Python3 is available
find_package(Python3 REQUIRED COMPONENTS NumPy)

## Check and display Python dependencies
execute_process(
    COMMAND ${Python3_EXECUTABLE} -m pip show numpy-quaternion pinocchio quadprog pynput pyqt5
    OUTPUT_VARIABLE PYTHON_PACKAGES
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "Python dependencies: ${PYTHON_PACKAGES}")

###################################
## catkin specific configuration ##
###################################
catkin_package(
  CATKIN_DEPENDS roscpp rospy std_msgs urdf xacro geometry_msgs trajectory_msgs
)

###########
## Build ##
###########

include_directories(
  ${catkin_INCLUDE_DIRS}
)

#############
## Install ##
#############

## Mark Python scripts for installation
catkin_install_python(PROGRAMS
  scripts/Full_IK.py
  scripts/IK_gazebo.py
  scripts/IK_set_point.py
  scripts/IK_set_point_gazebo.py
  scripts/IK_tracking.py
  scripts/IK_velocity_constrained.py
  scripts/distance_calc.py
  scripts/test.ipynb
  scripts/multicontroller.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Install launch files
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
  FILES_MATCHING PATTERN "*.launch"
)

## Install config files
install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
  FILES_MATCHING PATTERN "*.yaml"
)

## Install URDF files
install(DIRECTORY urdf/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/urdf
)

## Install meshes
install(DIRECTORY meshes/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/meshes
)

#############
## Testing ##
#############

## Optional: Add tests if needed
