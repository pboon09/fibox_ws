#!/usr/bin/python3

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    return LaunchDescription([       
        Node(
            package='joy',
            executable='joy_node',
            name='joy_node',
            output='screen',
            respawn=True,
        ),
        
        Node(
            package='final_mobile',
            executable='mobile_node.py',
            name='omni_robot_kinematics',
            output='screen',
            respawn=True,
        ),
        
        Node(
            package='final_mani',
            executable='mani_node.py',
            name='mani_shooter',
            output='screen',
            respawn=True,
        ),

        # ExecuteProcess(
        #     cmd=[
        #         'ros2', 'run', 'micro_ros_agent', 'micro_ros_agent',
        #         'serial', '--dev', '/dev/ttyACM0',
        #         '--baudrate', '115200'
        #     ],
        #     output='screen',
        # ),
        
        # ExecuteProcess(
        #     cmd=[
        #         'ros2', 'run', 'micro_ros_agent', 'micro_ros_agent',
        #         'serial', '--dev', '/dev/ttyACM2',
        #         '--baudrate', '115200'
        #     ],
        #     output='screen',
        # ),
    ])