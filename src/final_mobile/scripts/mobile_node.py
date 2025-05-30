#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from std_msgs.msg import Float64MultiArray, Float64, Int64
import math

class OmniKinematicsNode(Node):
    def __init__(self):
        super().__init__('omni_kinematics_node')

        self.ping = 1
        
        self.wheel_radius = 0.06
        self.robot_radius = 0.16
        self.max_speed = 20.0
        self.max_angular_speed = 30.0
        
        self.imu_offset = None
        self.current_imu_yaw = 0.0
        self.raw_imu_yaw = 0.0
        self.target_heading = 0.0
        
        self.heading_pid_kp = 0.15
        self.heading_pid_ki = 0.0   
        self.heading_pid_kd = 0.01
        self.heading_pid_integral = 0.0
        self.heading_pid_last_error = 0.0
        
        self.auto_heading_active = False
        self.target_reached = True  # Track if we've reached the current target
        self.target_reach_status = 0  # 0: idle, 1: reached, -1: not reached
        
        self.joy_subscription = self.create_subscription(
            Joy,
            'joy',
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
        
        self.wheel_speed_publisher = self.create_publisher(
            Float64MultiArray,
            'wheel_speed',
            10
        )
        
        self.target_reach_publisher = self.create_publisher(
            Int64,
            'target_reach',
            10
        )
        
        self.last_joy_time = self.get_clock().now()
        
        self.create_timer(0.1, self.watchdog_callback)
        
        self.get_logger().info('Omni Kinematics Node started')
        self.get_logger().info(f'Wheel radius: {self.wheel_radius}m')
        self.get_logger().info(f'Robot radius: {self.robot_radius}m')
        self.get_logger().info(f'Max speed: {self.max_speed}m/s')
        self.get_logger().info(f'Max angular speed: {self.max_angular_speed}rad/s')

    def imu_callback(self, msg):
        self.raw_imu_yaw = msg.data
        
        if self.imu_offset is None:
            self.imu_offset = msg.data
            self.current_imu_yaw = 0.0
            self.get_logger().info(f'IMU offset set: {self.imu_offset:.2f}° (raw) -> 0.0° (offset)')
        else:
            self.current_imu_yaw = self.raw_imu_yaw - self.imu_offset
            
            while self.current_imu_yaw > 180.0:
                self.current_imu_yaw -= 360.0
            while self.current_imu_yaw < -180.0:
                self.current_imu_yaw += 360.0

    def target_heading_callback(self, msg):
        new_target = msg.data
        
        # Reset condition: target is very close to zero
        if abs(new_target) <= 0.1:
            self.auto_heading_active = False
            self.target_reached = True
            self.target_reach_status = 0  # Idle
            self.target_heading = 0.0  # Explicitly reset target heading
            self.get_logger().info('TARGET CLEARED - Target heading is 0')
            self.publish_target_reach_status()
        
        # New target condition: accept any non-zero target when we're idle OR target is significantly different
        elif abs(new_target) > 0.1 and (self.target_reach_status == 0 or abs(new_target - self.target_heading) > 0.5):
            self.target_heading = new_target
            self.auto_heading_active = True
            self.target_reached = False
            self.target_reach_status = -1  # Not reached yet
            self.heading_pid_integral = 0.0
            self.heading_pid_last_error = 0.0
            self.get_logger().info(f'NEW TARGET RECEIVED - Target: {self.target_heading:.2f}, Current: {self.current_imu_yaw:.2f}')
            self.publish_target_reach_status()
        else:
            self.get_logger().info(f'TARGET IGNORED - Too similar to current target or system busy. New: {new_target:.2f}, Current: {self.target_heading:.2f}, Status: {self.target_reach_status}')

    def joy_callback(self, msg):
        self.last_joy_time = self.get_clock().now()
        
        # Speed control using axis 5
        if self.ping == 1:
            boost = msg.axes[4]
        else:
            boost = msg.axes[5]
            
        current_max_speed = 5.0 + (1.0 - boost) * (30.0 - 5.0) / 2.0
        current_max_angular_speed = 15.0 + (1.0 - boost) * (50.0 - 15.0) / 2.0
        
        # Get movement commands
        if self.ping == 1:
            vy = msg.axes[3] * current_max_speed
            vx = msg.axes[2] * -current_max_speed
            wz = msg.axes[0] * current_max_angular_speed
        else:
            vy = msg.axes[1] * current_max_speed
            vx = msg.axes[0] * -current_max_speed
            wz = msg.axes[2] * current_max_angular_speed
        
        # Handle auto heading mode - only when target heading is not 0
        if self.auto_heading_active and abs(self.target_heading) > 0.1 and abs(msg.axes[2]) < 0.1:
            pid_output = self.calculate_pid_control_to_target(current_max_angular_speed)
            wz = pid_output
            
            # Calculate error the same way as in PID control
            error = self.target_heading - self.current_imu_yaw
            
            # Normalize to [-180, 180] range
            while error > 180.0:
                error -= 360.0
            while error < -180.0:
                error += 360.0
            
            # Check if target reached using absolute error
            if abs(error) < 1.0:
                self.auto_heading_active = False
                self.target_reached = True
                self.target_reach_status = 1  # Reached
                self.get_logger().info(f'TARGET REACHED! Error: {error:.2f} degrees. PID stopped. Free to turn manually until new setpoint.')
                self.publish_target_reach_status()
        
        # Manual rotation overrides
        elif abs(msg.axes[2]) > 0.1:
            if self.auto_heading_active:
                self.auto_heading_active = False
                self.get_logger().info('AUTO HEADING OVERRIDDEN - Manual rotation detected')
        
        # Calculate and publish wheel speeds
        wheel_speeds = self.calculate_wheel_speeds(vx, vy, wz)
        self.publish_wheel_speeds(wheel_speeds)

    def calculate_pid_control_to_target(self, max_angular_speed):
        error = self.target_heading - self.current_imu_yaw
        
        while error > 180.0:
            error -= 360.0
        while error < -180.0:
            error += 360.0
        
        self.heading_pid_integral += error
        derivative = error - self.heading_pid_last_error
        
        output = (self.heading_pid_kp * error + 
                 self.heading_pid_ki * self.heading_pid_integral + 
                 self.heading_pid_kd * derivative)
        
        output = max(-max_angular_speed, min(max_angular_speed, output))
        
        self.heading_pid_last_error = error
        
        self.get_logger().info(f'TARGET PID - Target: {self.target_heading:.2f}, Current: {self.current_imu_yaw:.2f}, Error: {error:.2f}, Output: {output:.2f}')
        
        return output*2

    def normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def calculate_wheel_speeds(self, vx, vy, wz):
        wheel_angles = [
            7*math.pi/6,
            math.pi/2,
            11*math.pi/6
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

    def publish_target_reach_status(self):
        msg = Int64()
        msg.data = self.target_reach_status
        self.target_reach_publisher.publish(msg)

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