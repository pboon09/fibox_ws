#!/usr/bin/python3

import rclpy
from rclpy.node import Node
import math
import numpy as np
import time
from sensor_msgs.msg import Joy
from std_msgs.msg import Float64, Float64MultiArray
from geometry_msgs.msg import Twist


class OmniRobotKinematics(Node):
    def __init__(self):
        super().__init__('omni_robot_kinematics')
        
        self.wheel_radius = 0.06  
        self.robot_radius = 0.16  
        
        self.current_heading = 0.0
        self.zero_heading = 0.0
        self.zero_set = False
        
        self.vx = 0.0
        self.vy = 0.0
        self.vw = 0.0
        
        self.joy_msg = None
        self.prev_l1 = False
        self.prev_l2 = False
        self.imu_received = False
        self.slow_mode = False
        
        self.kp_heading = 0.008
        self.ki_heading = 0.0
        self.kd_heading = 0.001
        self.integral_error = 0.0
        self.prev_error = 0.0
        self.max_integral = 10.0
        self.pid_active = False
        
        self.heading_deadband = 1.0
        
        self.debug_counter = 0
        
        self.wheel_angles = np.array([
            7*math.pi/6,
            math.pi/2,
            11*math.pi/6
        ])
        
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
        
        self.target_heading_sub = self.create_subscription(
            Float64,
            '/target_head_robot',
            self.target_heading_callback,
            10
        )
        
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
        
        self.control_timer = self.create_timer(0.01, self.control_loop)
        
        self.get_logger().info("Omni Robot Kinematics Node Started")

    def joy_callback(self, msg):
        self.joy_msg = msg
        
        if len(msg.axes) >= 4:
            self.vx = float(msg.axes[0] * 1.0)
            self.vy = float(msg.axes[1] * 1.0)
            
            if not self.pid_active:
                self.vw = float(msg.axes[3] * 2.0)
        
        if len(msg.buttons) >= 8:
            l1_pressed = msg.buttons[6]
            l2_pressed = msg.buttons[7]
            
            self.slow_mode = l1_pressed
            
            if l2_pressed and not self.prev_l2:
                self.zero_heading = self.current_heading
                self.zero_set = True
                self.get_logger().info(f"L2 pressed: Zero heading set to {self.zero_heading:.1f}°")
            
            self.prev_l1 = l1_pressed
            self.prev_l2 = l2_pressed

    def imu_callback(self, msg):
        self.current_heading = float(msg.data)
        if not self.imu_received:
            self.imu_received = True
            self.get_logger().info(f"IMU data received. Current heading: {self.current_heading:.1f}°")

    def target_heading_callback(self, msg):
        target_heading = float(msg.data)
        
        if abs(target_heading) < 0.1:
            if self.pid_active:
                self.get_logger().info("Target heading reset to 0 - Stopping PID control")
                self.pid_active = False
                self.vw = 0.0
            else:
                self.get_logger().info("Target heading is 0 - No action taken")
            return
        
        self.zero_heading = target_heading
        self.zero_set = True
        self.pid_active = True
        self.reset_pid()
        self.get_logger().info(f"Mani target heading received: {target_heading:.1f}° - Auto heading to target")

    def normalize_angle_degrees(self, angle):
        while angle > 180.0:
            angle -= 360.0
        while angle < -180.0:
            angle += 360.0
        return float(angle)

    def reset_pid(self):
        self.integral_error = 0.0
        self.prev_error = 0.0

    def compute_pid_output(self, error, dt=0.02):
        p_term = self.kp_heading * error
        
        self.integral_error += error * dt
        self.integral_error = max(-self.max_integral, min(self.max_integral, self.integral_error))
        i_term = self.ki_heading * self.integral_error
        
        d_term = self.kd_heading * (error - self.prev_error) / dt
        self.prev_error = error
        
        output = p_term + i_term + d_term
        
        return output

    def compute_wheel_speeds(self, vx, vy, vw):
        wheel_speeds = []
        
        for i, angle in enumerate(self.wheel_angles):
            v_wheel = (vx * math.cos(angle) + 
                      vy * math.sin(angle) + 
                      vw * self.robot_radius)

            wheel_speed = v_wheel / self.wheel_radius * 120 / 10
            
            if self.slow_mode:
                max_speed = 100
            else:
                max_speed = 255
                
            wheel_speed = max(-max_speed, min(max_speed, wheel_speed))
            wheel_speeds.append(float(wheel_speed))
        
        return wheel_speeds

    def control_loop(self):
        if self.joy_msg is None or not self.imu_received:
            wheel_msg = Float64MultiArray()
            wheel_msg.data = [0.0, 0.0, 0.0]
            self.wheel_speed_pub.publish(wheel_msg)
            return
        
        if self.pid_active:
            heading_error = self.normalize_angle_degrees(self.zero_heading - self.current_heading)
            
            if abs(heading_error) <= self.heading_deadband:
                self.pid_active = False
                self.vw = 0.0
                self.get_logger().info("Reached zero heading - PID deactivated")
            else:
                pid_output = self.compute_pid_output(heading_error)
                
                if abs(pid_output) > 0.05:
                    if pid_output > 0:
                        pid_output = max(pid_output, 0.2)
                    else:
                        pid_output = min(pid_output, -0.2)
                
                self.vw = pid_output
                
                max_angular_vel = 2.0
                self.vw = float(max(-max_angular_vel, min(max_angular_vel, self.vw)))
        
        if abs(self.vx) < 0.05 and abs(self.vy) < 0.05 and abs(self.vw) < 0.05:
            wheel_speeds = [0.0, 0.0, 0.0]
        else:
            wheel_speeds = self.compute_wheel_speeds(self.vx, self.vy, self.vw)
        
        wheel_msg = Float64MultiArray()
        wheel_msg.data = [float(speed) for speed in wheel_speeds]
        self.wheel_speed_pub.publish(wheel_msg)
        
        cmd_vel = Twist()
        cmd_vel.linear.x = float(self.vx)
        cmd_vel.linear.y = float(self.vy)
        cmd_vel.angular.z = float(self.vw)
        self.cmd_vel_pub.publish(cmd_vel)

def main(args=None):
    rclpy.init(args=args)
    node = OmniRobotKinematics()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()