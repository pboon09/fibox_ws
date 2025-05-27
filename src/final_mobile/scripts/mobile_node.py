#!/usr/bin/python3

import rclpy
from rclpy.node import Node
import math
import numpy as np
from sensor_msgs.msg import Joy
from std_msgs.msg import Float64, Float64MultiArray
from geometry_msgs.msg import Twist


class OmniRobotKinematics(Node):
    def __init__(self):
        super().__init__('omni_robot_kinematics')
        
        # Robot parameters
        self.wheel_radius = 0.06  
        self.robot_radius = 0.16  
        
        # Robot state
        self.current_heading = 0.0  # Current robot heading in radians
        self.target_heading = 0.0   # Target heading after L1/L2 press
        self.heading_locked = True  # Whether heading is locked
        
        # Control inputs
        self.vx = 0.0  # Linear velocity in x (forward/backward)
        self.vy = 0.0  # Linear velocity in y (left/right)
        self.vw = 0.0  # Angular velocity (rotation)
        
        # Joystick state
        self.joy_msg = None
        self.prev_l1 = False
        self.prev_l2 = False
        
        # Wheel configuration for 3-wheel omni (1 front, 2 back)
        # Angles: Front wheel at 90°, Back wheels at 210° and 330°
        self.wheel_angles = np.array([
            math.pi/2,      # Front wheel (90°)
            7*math.pi/6,    # Back left wheel (210°) 
            11*math.pi/6    # Back right wheel (330°)
        ])
        
        # Subscribers
        self.joy_sub = self.create_subscription(
            Joy,
            '/joy',
            self.joy_callback,
            10
        )
        
        self.imu_sub = self.create_subscription(
            Float64,
            '/imu_yaw',
            self.imu_callback,
            10
        )
        
        # Publishers
        self.wheel_speed_pub = self.create_publisher(
            Float64MultiArray,
            '/wheel_speed',
            10
        )
        
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )
        
        # Control loop timer (50Hz)
        self.control_timer = self.create_timer(0.02, self.control_loop)
        
        self.get_logger().info("Omni Robot Kinematics Node Started")

    def joy_callback(self, msg):
        self.joy_msg = msg
        
        if len(msg.axes) >= 4:
            self.vx = msg.axes[1] * 1.0  # Left stick Y-axis (forward/backward)
            self.vy = msg.axes[0] * 1.0  # Left stick X-axis (left/right)
        
        if len(msg.buttons) >= 6:
            l1_pressed = msg.buttons[4]  # L1 button
            l2_pressed = msg.buttons[5]  # L2 button
            
            if l1_pressed and not self.prev_l1:
                self.target_heading -= math.pi/2  # Rotate -90 degrees
                self.get_logger().info(f"L1 pressed: Target heading = {math.degrees(self.target_heading):.1f}°")
            
            if l2_pressed and not self.prev_l2:
                self.target_heading += math.pi/2  # Rotate +90 degrees
                self.get_logger().info(f"L2 pressed: Target heading = {math.degrees(self.target_heading):.1f}°")
            
            self.prev_l1 = l1_pressed
            self.prev_l2 = l2_pressed

    def imu_callback(self, msg):
        self.current_heading = msg.data

    def normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def compute_wheel_speeds(self, vx, vy, vw):
        wheel_speeds = []
        
        for i, angle in enumerate(self.wheel_angles):
            v_wheel = (vx * math.cos(angle) + 
                      vy * math.sin(angle) + 
                      vw * self.robot_radius)

            wheel_speed = v_wheel / self.wheel_radius * 255 / 10
            
            wheel_speed = max(-255, min(255, wheel_speed))
            wheel_speeds.append(wheel_speed)
        
        return wheel_speeds

    def control_loop(self):
        if self.joy_msg is None:
            return
        
        heading_error = self.normalize_angle(self.target_heading - self.current_heading)
        
        kp_heading = 2.0
        self.vw = kp_heading * heading_error
        
        max_angular_vel = 2.0
        self.vw = max(-max_angular_vel, min(max_angular_vel, self.vw))
        
        if abs(self.vx) < 0.1 and abs(self.vy) < 0.1 and abs(heading_error) < 0.1:
            self.vw = 0.0
        
        wheel_speeds = self.compute_wheel_speeds(self.vx, self.vy, self.vw)
        
        wheel_msg = Float64MultiArray()
        wheel_msg.data = wheel_speeds
        self.wheel_speed_pub.publish(wheel_msg)
        
        cmd_vel = Twist()
        cmd_vel.linear.x = self.vx
        cmd_vel.linear.y = self.vy
        cmd_vel.angular.z = self.vw
        self.cmd_vel_pub.publish(cmd_vel)

def main(args=None):
    rclpy.init(args=args)
    node = OmniRobotKinematics()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()