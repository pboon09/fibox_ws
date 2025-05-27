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
        
        # Robot parameters
        self.wheel_radius = 0.06  
        self.robot_radius = 0.16  
        
        # Robot state
        self.current_heading = 0.0  # Current robot heading in radians
        self.target_heading = 0.0   # Target heading after L1/L2 press
        self.heading_locked = False  # Whether heading control is active
        
        # Control inputs
        self.vx = 0.0  # Linear velocity in x (forward/backward)
        self.vy = 0.0  # Linear velocity in y (left/right)
        self.vw = 0.0  # Angular velocity (rotation)
        
        # Joystick state
        self.joy_msg = None
        self.prev_l1 = False
        self.prev_l2 = False
        self.imu_received = False  # Track if we've received IMU data
        
        # PID parameters for heading control (working in degrees)
        self.kp_heading = 0.008   # Proportional gain (adjusted for degrees)
        self.ki_heading = 0.0   # Integral gain (adjusted for degrees)
        self.kd_heading = 0.001   # Derivative gain (adjusted for degrees)
        self.integral_error = 0.0
        self.prev_error = 0.0
        self.max_integral = 10.0  # Integral windup protection (adjusted for degrees)
        
        # Control parameters
        self.heading_deadband = 1.0  # 2 degrees deadband
        self.movement_threshold = 0.1  # Minimum movement to consider "moving"
        self.heading_unlock_timeout = 3.0  # Seconds of no movement before unlocking
        self.last_movement_time = time.time()
        
        # Debug counter for printing
        self.debug_counter = 0
        
        # Wheel configuration: [motor1(left), motor2(front), motor3(right)]
        # motor1(left) = 210°, motor2(front) = 90°, motor3(right) = 330°
        self.wheel_angles = np.array([
            7*math.pi/6,    # Motor 1 (left wheel) at 210°
            math.pi/2,      # Motor 2 (front wheel) at 90°
            11*math.pi/6    # Motor 3 (right wheel) at 330°
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
        
        # New subscriber for target heading
        self.target_heading_sub = self.create_subscription(
            Float64,
            '/target_head_robot',
            self.target_heading_callback,
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
        self.get_logger().info("Motor mapping: [Motor1(Left), Motor2(Front), Motor3(Right)]")
        self.get_logger().info(f"PID Gains (degrees) - P: {self.kp_heading}, I: {self.ki_heading}, D: {self.kd_heading}")
        self.get_logger().info("Waiting for IMU data...")

    def joy_callback(self, msg):
        self.joy_msg = msg
        
        if len(msg.axes) >= 4:
            self.vx = float(msg.axes[1] * 1.0)  # Left stick Y-axis (forward/backward)
            self.vy = float(msg.axes[0] * 1.0)  # Left stick X-axis (left/right)
        
        if len(msg.buttons) >= 8:
            l1_pressed = msg.buttons[6]  # L1 button
            l2_pressed = msg.buttons[7]  # L2 button
            
            if l1_pressed and not self.prev_l1:
                self.target_heading += 90.0  # Rotate -90 degrees
                self.target_heading = self.normalize_angle_degrees(self.target_heading)
                self.heading_locked = True
                self.reset_pid()  # Reset PID when new target is set
                self.get_logger().info(f"L1 pressed: Target heading = {self.target_heading:.1f}°")
            
            if l2_pressed and not self.prev_l2:
                self.target_heading -= 90.0  # Rotate +90 degrees
                self.target_heading = self.normalize_angle_degrees(self.target_heading)
                self.heading_locked = True
                self.reset_pid()  # Reset PID when new target is set
                self.get_logger().info(f"L2 pressed: Target heading = {self.target_heading:.1f}°")
            
            self.prev_l1 = l1_pressed
            self.prev_l2 = l2_pressed

    def imu_callback(self, msg):
        # IMU data is already in degrees, keep as degrees
        self.current_heading = float(msg.data)
        if not self.imu_received:
            self.imu_received = True
            self.target_heading = self.current_heading  # Initialize target to current
            self.get_logger().info(f"IMU data received. Initial heading: {self.current_heading:.1f}°")

    def target_heading_callback(self, msg):
        """Callback for receiving target heading from topic (in degrees)"""
        # Target heading is already in degrees, keep as degrees
        self.target_heading = float(msg.data)
        self.target_heading = self.normalize_angle_degrees(self.target_heading)
        self.heading_locked = True
        self.reset_pid()  # Reset PID when new target is set
        self.get_logger().info(f"Target heading updated from topic: {self.target_heading:.1f}°")

    def normalize_angle_degrees(self, angle):
        """Normalize angle to [-180, 180] range in degrees"""
        while angle > 180.0:
            angle -= 360.0
        while angle < -180.0:
            angle += 360.0
        return float(angle)

    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi] range"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return float(angle)

    def reset_pid(self):
        """Reset PID controller state"""
        self.integral_error = 0.0
        self.prev_error = 0.0

    def compute_pid_output(self, error, dt=0.02):
        """Compute PID output for heading control"""
        # Proportional term
        p_term = self.kp_heading * error
        
        # Integral term with windup protection
        self.integral_error += error * dt
        self.integral_error = max(-self.max_integral, min(self.max_integral, self.integral_error))
        i_term = self.ki_heading * self.integral_error
        
        # Derivative term
        d_term = self.kd_heading * (error - self.prev_error) / dt
        self.prev_error = error
        
        # Total PID output
        output = p_term + i_term + d_term
        
        return output

    def compute_wheel_speeds(self, vx, vy, vw):
        wheel_speeds = []
        
        for i, angle in enumerate(self.wheel_angles):
            v_wheel = (vx * math.cos(angle) + 
                      vy * math.sin(angle) + 
                      vw * self.robot_radius)

            wheel_speed = v_wheel / self.wheel_radius * 120 / 10
            
            wheel_speed = max(-120, min(120, wheel_speed))
            # Explicitly convert to Python float
            wheel_speeds.append(float(wheel_speed))
        
        return wheel_speeds

    def control_loop(self):
        if self.joy_msg is None or not self.imu_received:
            # Publish zero speeds if no joystick or IMU data
            wheel_msg = Float64MultiArray()
            wheel_msg.data = [0.0, 0.0, 0.0]  # [Motor1(Left), Motor2(Front), Motor3(Right)]
            self.wheel_speed_pub.publish(wheel_msg)
            return
        
        # Calculate heading error in degrees with proper normalization
        heading_error = self.normalize_angle_degrees(self.target_heading - self.current_heading)
        
        # Determine if we should apply heading correction
        is_moving = abs(self.vx) > self.movement_threshold or abs(self.vy) > self.movement_threshold
        significant_heading_error = abs(heading_error) > self.heading_deadband
        
        # Should correct heading if:
        # 1. Robot is moving (maintain heading during movement)
        # 2. OR heading is explicitly locked AND there's a significant error
        # 3. OR there's a significant error (always try to correct large errors)
        should_correct_heading = (is_moving or 
                                (self.heading_locked and significant_heading_error) or 
                                significant_heading_error)
        
        # Update movement time for timeout
        current_time = time.time()
        if is_moving:
            self.last_movement_time = current_time
        
        # Store PID output for debugging
        pid_output = 0.0
        
        # Apply heading correction when needed
        if should_correct_heading:
            # When moving, always maintain heading (field-centric control)
            if is_moving:
                # Lock heading to current target when starting to move
                if not self.heading_locked:
                    self.target_heading = self.current_heading
                    self.heading_locked = True
                    self.reset_pid()
                    self.get_logger().info(f"Movement detected - Locking heading at {self.target_heading:.1f}°")
            
            # Apply PID correction for any significant error
            pid_output = self.compute_pid_output(heading_error)
            
            # Add minimum output to overcome static friction
            if abs(pid_output) > 0.05:  # Only apply if PID output is significant (adjusted for degrees)
                if pid_output > 0:
                    pid_output = max(pid_output, 0.2)  # Minimum positive output (adjusted for degrees)
                else:
                    pid_output = min(pid_output, -0.2)  # Minimum negative output (adjusted for degrees)
            
            self.vw = pid_output
            
            max_angular_vel = 2.0
            self.vw = float(max(-max_angular_vel, min(max_angular_vel, self.vw)))
        else:
            self.vw = 0.0
            pid_output = 0.0
            # Only unlock heading when there's no significant error AND not moving
            time_since_movement = current_time - self.last_movement_time
            if not is_moving and not significant_heading_error and time_since_movement > 1.0:
                if self.heading_locked:
                    self.get_logger().info("Heading unlocked - target reached")
                self.heading_locked = False
                self.reset_pid()  # Reset PID when unlocking
        
        # If robot is completely stationary (no movement, no rotation needed)
        if abs(self.vx) < 0.05 and abs(self.vy) < 0.05 and abs(self.vw) < 0.05:
            wheel_speeds = [0.0, 0.0, 0.0]
        else:
            wheel_speeds = self.compute_wheel_speeds(self.vx, self.vy, self.vw)
        
        # Publish wheel speeds: [Motor1(Left), Motor2(Front), Motor3(Right)]
        wheel_msg = Float64MultiArray()
        wheel_msg.data = [float(speed) for speed in wheel_speeds]
        self.wheel_speed_pub.publish(wheel_msg)
        
        # Publish cmd_vel
        cmd_vel = Twist()
        cmd_vel.linear.x = float(self.vx)
        cmd_vel.linear.y = float(self.vy)
        cmd_vel.angular.z = float(self.vw)
        self.cmd_vel_pub.publish(cmd_vel)
        
        # Debug output - print every 25 loops (0.5 seconds at 50Hz)
        # self.debug_counter += 1
        # if self.debug_counter >= 25:
        #     self.debug_counter = 0
        #     print(f"DEBUG: Current_H={self.current_heading:6.1f}° | "
        #           f"Target_H={self.target_heading:6.1f}° | "
        #           f"Error={heading_error:6.1f}° | "
        #           f"PID_Out={pid_output:6.3f} | "
        #           f"VW={self.vw:6.3f} | "
        #           f"Locked={self.heading_locked} | "
        #           f"Moving={is_moving} | "
        #           f"SigErr={significant_heading_error} | "
        #           f"ShouldCorr={should_correct_heading} | "
        #           f"Wheels=[{wheel_speeds[0]:6.1f}, {wheel_speeds[1]:6.1f}, {wheel_speeds[2]:6.1f}]")

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