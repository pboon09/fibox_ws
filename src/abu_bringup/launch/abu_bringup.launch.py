#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='joy',
            executable='joy_node',
            name='joy_node',
            parameters=[{
                'dev': '/dev/input/js0',  # Update this if your joystick is different
                'deadzone': 0.1,
                'autorepeat_rate': 20.0
            }]
        ),
        
        Node(
            package='abu_joy_converter',
            executable='abu_joy_converter.py',
            name='abu_joy_converter',
            parameters=[{
                'input_topic': '/joy',
                'output_topic': '/cmd_vel',
            }]
        ),
        
        Node(
            package='abu_kinematic',
            executable='abu_kinematic.py',
            name='abu_kinematic',
            parameters=[{
                'input_topic': '/cmd_vel',
                'output_topic': '/wheel_velocities',
            }]
        )
    ])