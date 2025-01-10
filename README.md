# Robotic Arm Teleoperation in Gazebo
This repository provides simulation and teleoperation tools for controlling a robotic arm in Gazebo using keyboard input and ROS controllers.

## Features
- **Keyboard Control**: Manual control of end-effector velocities.
- **Velocity Mapping**: Key-based directional velocity control.
- **Inverse Kinematics**: Joint velocities calculated using the Jacobian inverse.

## How to use this repo
### Prerequisites :
- #### Install Gazebo
  We will be launching our world in Gazebo so make sure to install it by using the command 
  ```
  curl -sSL http://get.gazebosim.org | sh
  ```
- #### Install ROS dependencies

  ```
  sudo apt-get install ros-noetic-gazebo-ros-pkgs ros-noetic-urdf ros-noetic-xacro ros-noetic-ros-control ros-noetic-ros-controllers
  ```
  For motion planning
  ```
  sudo apt-get install ros-noetic-moveit
  ```
  
> [!NOTE]
> All the installation commands are for rosdep noetic change noetic with <your_distro_name>

- #### Install ROS packages
  Make a workspace and create a directory 'src' where all the packages will be stored, clone this repo to get the packages and then build the catkin workspace.
  ```
  cd ~/robo_arm/src/
  git clone https://github.com/Team-Deimos-IIT-Mandi/Robotic_Arm.git
  cd ~/robo_arm && catkin build
  ```
  Source your workspace in .bashrc file by running the following command so that you don't have to source it in every terminal
  ```
  echo "source ~/robo_arm/devel/setup.bash" >> ~/.bashrc
  ```

  ## Control new arm

  ### To start the Gazebo simulation:
  
  ```
  roslaunch Arm_Urdf new.launch
  ```

  ### To control the robotic arm via keyboard:
  
  ```
  rosrun Arm_Urdf IK_gazebo.py
  ```

  ### To move arm to a point in space:

  ```
  rosrun Arm_Urdf IK_set_point_gazebo.py
  ```
  ### To move arm to a trajectory of points in space:
  
  ```
  rosrun Arm_Urdf IK_tracking.py
  ```
  
  ## Control old arm

  ### Launch Gazebo
  To start the Gazebo simulation:
  
  ```
  roslaunch moveit_pkg final.launch
  ```
  ### Run Teleoperation
  To control the robotic arm via keyboard:
  
  ```
  rosrun teleop_arm jaco_pt2.py
  ```

  ### Keyboard Controls
  - `w` - Move +X
  - `s` - Move -X
  - `a` - Move +Y
  - `d` - Move -Y
  - `q` - Move +Z
  - `e` - Move -Z

