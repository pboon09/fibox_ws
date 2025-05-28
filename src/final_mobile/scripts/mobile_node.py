#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from std_msgs.msg import Float64MultiArray, Float64
import math

class OmniKinematicsNode(Node):
    def __init__(self):
        super().__init__('omni_kinematics_node')
        
        self.wheel_radius = 0.06
        self.robot_radius = 0.16
        self.max_speed = 20.0
        self.max_angular_speed = 20.0
        
        # PID control variables
        self.zero_deg = None
        self.imu_offset = None  # Store the first IMU reading as offset
        self.current_imu_yaw = 0.0  # This will be the offset-corrected reading
        self.raw_imu_yaw = 0.0  # Store the raw IMU reading
        self.target_heading = 0.0
        self.pid_kp = 0.1
        self.pid_ki = 0.0
        self.pid_kd = 0.0
        self.pid_integral = 0.0
        self.pid_last_error = 0.0
        
        # Button state tracking
        self.button6_last_state = 0
        self.pid_active = False
        self.auto_heading_active = False  # New flag for automatic heading control
        
        # Create subscriber for joy messages
        self.joy_subscription = self.create_subscription(
            Joy,
            'joy',
            self.joy_callback,
            10
        )
        
        # Create IMU subscriber
        self.imu_sub = self.create_subscription(
            Float64,
            '/imu_yaw',
            self.imu_callback,
            10
        )
        
        # Create target heading subscriber
        self.target_heading_sub = self.create_subscription(
            Float64,
            '/target_head_robot',
            self.target_heading_callback,  
            10
        )
        
        # Create publisher for wheel speeds
        self.wheel_speed_publisher = self.create_publisher(
            Float64MultiArray,
            'wheel_speed',
            10
        )
        
        # Initialize last command time for safety
        self.last_joy_time = self.get_clock().now()
        
        # Create timer for watchdog (stop robot if no commands received)
        self.create_timer(0.1, self.watchdog_callback)
        
        self.get_logger().info('Omni Kinematics Node started')
        self.get_logger().info(f'Wheel radius: {self.wheel_radius}m')
        self.get_logger().info(f'Robot radius: {self.robot_radius}m')
        self.get_logger().info(f'Max speed: {self.max_speed}m/s')
        self.get_logger().info(f'Max angular speed: {self.max_angular_speed}rad/s')

    def imu_callback(self, msg):
        self.raw_imu_yaw = msg.data
        
        # Set the first IMU reading as the offset (this becomes our 0 degree reference)
        if self.imu_offset is None:
            self.imu_offset = msg.data
            self.zero_deg = 0.0  # Zero degree is now 0 after offset
            self.current_imu_yaw = 0.0  # First reading becomes 0
            self.get_logger().info(f'IMU offset set: {self.imu_offset:.2f}° (raw) -> 0.0° (offset)')
        else:
            # Apply offset to get relative heading (first reading = 0°)
            self.current_imu_yaw = self.raw_imu_yaw - self.imu_offset
            
            # Normalize to [-180, 180] degrees
            while self.current_imu_yaw > 180.0:
                self.current_imu_yaw -= 360.0
            while self.current_imu_yaw < -180.0:
                self.current_imu_yaw += 360.0

    def target_heading_callback(self, msg):
        self.target_heading = msg.data
        
        # Auto-activate heading control when target heading is not zero
        if abs(self.target_heading) > 0.1:  # Small threshold to avoid noise
            self.auto_heading_active = True
            self.pid_integral = 0.0
            self.pid_last_error = 0.0
            self.get_logger().info(f'AUTO HEADING ACTIVATED - Target: {self.target_heading:.2f}, Current: {self.current_imu_yaw:.2f}')
        else:
            # Target heading is 0, deactivate automatic heading control
            self.auto_heading_active = False
            self.get_logger().info('AUTO HEADING DEACTIVATED - Target heading is 0')

    def joy_callback(self, msg):
        self.last_joy_time = self.get_clock().now()
        
        # Debug: show button presses
        if len(msg.buttons) > 6:
            if msg.buttons[6] == 1:
                self.get_logger().info('Button 6 pressed!')
        
        # Check for button 6 press to activate PID back to zero_deg
        if len(msg.buttons) > 6:
            button6_current_state = msg.buttons[6]
            if button6_current_state == 1 and self.button6_last_state == 0:
                if self.imu_offset is not None:  # Changed from zero_deg to imu_offset
                    # Deactivate auto heading and activate manual PID to zero
                    self.auto_heading_active = False
                    self.pid_active = True
                    self.pid_integral = 0.0
                    self.pid_last_error = 0.0
                    self.get_logger().info(f'MANUAL PID ACTIVATED - Going back to zero: {self.zero_deg:.2f}, Current: {self.current_imu_yaw:.2f}')
                else:
                    self.get_logger().info('Cannot activate PID - IMU offset not set yet')
            self.button6_last_state = button6_current_state
        
        # Analog speed control using axes[5] (ranges from 1 to -1)
        axes5_value = msg.axes[5]
        current_max_speed = 8.0 + (1.0 - axes5_value) * (17.0 - 8.0) / 2.0
        
        # Extract velocities from joystick
        vy = msg.axes[1] * current_max_speed    # Forward/backward
        vx = 0.0
        wz = msg.axes[2] * self.max_angular_speed # Rotation
        
        # Apply manual PID control if active (button 6 pressed - go to zero)
        if self.pid_active and self.imu_offset is not None:  # Changed from zero_deg to imu_offset
            pid_output = self.calculate_pid_control_to_zero()
            wz = pid_output
            
            # Check if we've reached the target angle (within tolerance)
            error = abs(self.zero_deg - self.current_imu_yaw)  # zero_deg is now 0.0
            if error > 180.0:
                error = 360.0 - error
            
            if error < 0.1:  # Within tolerance of target
                self.pid_active = False
                self.get_logger().info(f'MANUAL PID DEACTIVATED - Zero target reached! Error: {error:.2f} degrees')
        
        # Apply automatic heading control if active and no manual rotation input
        elif self.auto_heading_active and abs(msg.axes[2]) < 0.1:  # No manual rotation input
            pid_output = self.calculate_pid_control_to_target()
            wz = pid_output
            
            # Check if we've reached the target heading (within tolerance)
            error = abs(self.target_heading - self.current_imu_yaw)
            if error > 180.0:
                error = 360.0 - error
            
            if error < 3.0:  # Within 3 degrees of target
                self.auto_heading_active = False
                self.get_logger().info(f'AUTO HEADING DEACTIVATED - Target reached! Error: {error:.2f} degrees')
        
        # If there's manual rotation input, disable auto heading
        elif abs(msg.axes[2]) > 0.1:
            if self.auto_heading_active:
                self.auto_heading_active = False
                self.get_logger().info('AUTO HEADING OVERRIDDEN - Manual rotation detected')
        
        # Calculate wheel speeds using omni-directional kinematics
        wheel_speeds = self.calculate_wheel_speeds(vx, vy, wz)
        
        # Publish wheel speeds
        self.publish_wheel_speeds(wheel_speeds)

    def calculate_pid_control_to_zero(self):
        """PID control to return to zero degree reference (which is 0.0 after offset)"""
        # Work with degrees since IMU outputs degrees
        error = self.zero_deg - self.current_imu_yaw  # zero_deg is 0.0, so error = -current_imu_yaw
        
        # Normalize error to [-180, 180] degrees
        while error > 180.0:
            error -= 360.0
        while error < -180.0:
            error += 360.0
        
        self.pid_integral += error
        derivative = error - self.pid_last_error
        
        output = (self.pid_kp * error + 
                 self.pid_ki * self.pid_integral + 
                 self.pid_kd * derivative)
        
        output = max(-self.max_angular_speed, min(self.max_angular_speed, output))
        
        self.pid_last_error = error
        
        # Debug PID
        self.get_logger().info(f'ZERO PID - Current: {self.current_imu_yaw:.2f}, Error: {error:.2f}, Output: {output:.2f}')
        
        return output

    def calculate_pid_control_to_target(self):
        """PID control to reach target heading"""
        # Work with degrees since IMU outputs degrees
        error = self.target_heading - self.current_imu_yaw
        
        # Normalize error to [-180, 180] degrees
        while error > 180.0:
            error -= 360.0
        while error < -180.0:
            error += 360.0
        
        self.pid_integral += error
        derivative = error - self.pid_last_error
        
        output = (self.pid_kp * error + 
                 self.pid_ki * self.pid_integral + 
                 self.pid_kd * derivative)
        
        output = max(-self.max_angular_speed, min(self.max_angular_speed, output))
        
        self.pid_last_error = error
        
        # Debug PID
        self.get_logger().info(f'TARGET PID - Target: {self.target_heading:.2f}, Current: {self.current_imu_yaw:.2f}, Error: {error:.2f}, Output: {output:.2f}')
        
        return output

    def normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def calculate_wheel_speeds(self, vx, vy, wz):
        # Wheel configuration for 3-wheel omni robot
        wheel_angles = [
            7*math.pi/6,    # Motor 0: Left wheel at 210 degrees
            math.pi/2,      # Motor 1: Front wheel at 90 degrees  
            11*math.pi/6    # Motor 2: Right wheel at 330 degrees
        ]
        
        wheel_speeds = []
        
        for angle in wheel_angles:
            linear_component = (-vx * math.sin(angle) + vy * math.cos(angle))
            angular_component = wz * self.robot_radius
            
            wheel_linear_speed = linear_component + angular_component
            wheel_angular_speed = wheel_linear_speed / self.wheel_radius
            
            wheel_speeds.append(wheel_angular_speed)
        
        return wheel_speeds

    def publish_wheel_speeds(self, wheel_speeds):
        msg = Float64MultiArray()
        msg.data = [float(speed) for speed in wheel_speeds]
        
        self.wheel_speed_publisher.publish(msg)

    def watchdog_callback(self):
        current_time = self.get_clock().now()
        time_diff = (current_time - self.last_joy_time).nanoseconds / 1e9
        
        if time_diff > 0.5:
            zero_speeds = [0.0, 0.0, 0.0]
            self.publish_wheel_speeds(zero_speeds)

def main(args=None):
    rclpy.init(args=args)
    
    node = OmniKinematicsNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        zero_speeds = [0.0, 0.0, 0.0]
        node.publish_wheel_speeds(zero_speeds)
        
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()