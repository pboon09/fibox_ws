#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray
import math

class AbuKinematic(Node):
    def __init__(self):
        super().__init__('abu_kinematic')
        
        self.declare_parameter('input_topic', '/cmd_vel')
        self.declare_parameter('output_topic', '/wheel_velocities')
        
        cmd_vel = self.get_parameter('input_topic').get_parameter_value().string_value
        wheel_velocities = self.get_parameter('output_topic').get_parameter_value().string_value
        
        # Robot parameter
        self.wheel_radius = 0.05  # meters
        self.robot_radius = 0.15  # meters (distance from center to wheel)
        self.wheel_angles = [math.pi/6, 5*math.pi/6, 3*math.pi/2]  # 30°, 150°, 270°
        
        self.wheel_vel_pub = self.create_publisher(
            Float32MultiArray,
            wheel_velocities,
            10
        )
        
        self.twist_sub = self.create_subscription(
            Twist,
            cmd_vel,
            self.twist_callback,
            10
        )
        
        self.get_logger().info(
            f"abu_joy_converabu_kinematicter node started\n"
            f"Subscribing to {cmd_vel}\n"
            f"Publishing to {wheel_velocities}")

    def twist_callback(self, msg):
        vx = msg.linear.x
        vy = msg.linear.y
        wz = msg.angular.z
        
    # Wheel 1: Front-right (typically at 30°)
    # Wheel 2: Front-left (typically at 150°)
    # Wheel 3: Rear-center (typically at 270°)
        wheel1 = (-math.sin(self.wheel_angles[0]) * vx + 
                   math.cos(self.wheel_angles[0]) * vy + 
                   self.robot_radius * wz) / self.wheel_radius
        
        wheel2 = (-math.sin(self.wheel_angles[1]) * vx + 
                   math.cos(self.wheel_angles[1]) * vy + 
                   self.robot_radius * wz) / self.wheel_radius
        
        wheel3 = (-math.sin(self.wheel_angles[2]) * vx + 
                   math.cos(self.wheel_angles[2]) * vy + 
                   self.robot_radius * wz) / self.wheel_radius
        
        wheel_vel_msg = Float32MultiArray()
        wheel_vel_msg.data = [wheel1, wheel2, wheel3]
        self.wheel_vel_pub.publish(wheel_vel_msg)
        
        # self.get_logger().info(f"Published wheel velocities: {wheel1:.2f}, {wheel2:.2f}, {wheel3:.2f}")

def main(args=None):
    rclpy.init(args=args)
    node = AbuKinematic()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()