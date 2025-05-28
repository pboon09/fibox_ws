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
        self.max_angular_speed = 30.0
        
        self.zero_deg = None
        self.imu_offset = None
        self.current_imu_yaw = 0.0
        self.raw_imu_yaw = 0.0
        self.target_heading = 0.0
        
        self.zero_pid_kp = 0.1
        self.zero_pid_ki = 0.0
        self.zero_pid_kd = 0.01
        self.zero_pid_integral = 0.0
        self.zero_pid_last_error = 0.0
        
        self.heading_pid_kp = 0.15
        self.heading_pid_ki = 0.0
        self.heading_pid_kd = 0.1
        self.heading_pid_integral = 0.0
        self.heading_pid_last_error = 0.0
        
        self.button6_last_state = 0
        self.button7_last_state = 0
        self.pid_active = False
        self.auto_heading_active = False
        self.straight_forward_active = False
        self.straight_forward_heading = 0.0
        self.auto_lock_active = False
        self.locked_heading = 0.0
        self.last_manual_rotation_time = None
        
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
            self.zero_deg = 0.0
            self.current_imu_yaw = 0.0
            self.get_logger().info(f'IMU offset set: {self.imu_offset:.2f}° (raw) -> 0.0° (offset)')
        else:
            self.current_imu_yaw = self.raw_imu_yaw - self.imu_offset
            
            while self.current_imu_yaw > 180.0:
                self.current_imu_yaw -= 360.0
            while self.current_imu_yaw < -180.0:
                self.current_imu_yaw += 360.0

    def target_heading_callback(self, msg):
        self.target_heading = msg.data
        
        if abs(self.target_heading) > 0.1:
            self.auto_heading_active = True
            self.straight_forward_active = False
            self.auto_lock_active = False
            self.heading_pid_integral = 0.0
            self.heading_pid_last_error = 0.0
            self.get_logger().info(f'AUTO HEADING ACTIVATED - Target: {self.target_heading:.2f}, Current: {self.current_imu_yaw:.2f}')
        else:
            self.auto_heading_active = False
            self.get_logger().info('AUTO HEADING DEACTIVATED - Target heading is 0')

    def joy_callback(self, msg):
        self.last_joy_time = self.get_clock().now()
        
        if len(msg.buttons) > 6:
            if msg.buttons[6] == 1:
                self.get_logger().info('Button 6 pressed!')
        
        if len(msg.buttons) > 7:
            if msg.buttons[7] == 1:
                self.get_logger().info('Button 7 pressed!')
        
        if len(msg.buttons) > 6:
            button6_current_state = msg.buttons[6]
            if button6_current_state == 1 and self.button6_last_state == 0:
                if self.imu_offset is not None:
                    self.auto_heading_active = False
                    self.straight_forward_active = False
                    self.auto_lock_active = False
                    self.pid_active = True
                    self.zero_pid_integral = 0.0
                    self.zero_pid_last_error = 0.0
                    self.get_logger().info(f'MANUAL PID ACTIVATED - Going back to zero: {self.zero_deg:.2f}, Current: {self.current_imu_yaw:.2f}')
                else:
                    self.get_logger().info('Cannot activate PID - IMU offset not set yet')
            self.button6_last_state = button6_current_state
        
        if len(msg.buttons) > 7:
            button7_current_state = msg.buttons[7]
            if button7_current_state == 1 and self.button7_last_state == 0:
                if self.imu_offset is not None:
                    self.straight_forward_active = True
                    self.straight_forward_heading = self.current_imu_yaw
                    self.auto_heading_active = False
                    self.auto_lock_active = False
                    self.pid_active = False
                    self.heading_pid_integral = 0.0
                    self.heading_pid_last_error = 0.0
                    self.get_logger().info(f'STRAIGHT FORWARD ACTIVATED - Locked heading: {self.straight_forward_heading:.2f}')
                else:
                    self.get_logger().info('Cannot activate straight forward - IMU offset not set yet')
            self.button7_last_state = button7_current_state
        
        axes5_value = msg.axes[5]
        current_max_speed = 8.0 + (1.0 - axes5_value) * (17.0 - 8.0) / 2.0
        
        vy = msg.axes[3] * current_max_speed
        vx = msg.axes[2] * -current_max_speed
        wz = msg.axes[0] * self.max_angular_speed
        
        current_time = self.get_clock().now()
        
        if abs(msg.axes[0]) > 0.1:
            self.last_manual_rotation_time = current_time
            if self.auto_lock_active:
                self.auto_lock_active = False
                self.get_logger().info('AUTO LOCK DEACTIVATED - Manual rotation detected')
        
        elif self.last_manual_rotation_time is not None:
            time_since_rotation = (current_time - self.last_manual_rotation_time).nanoseconds / 1e9
            if time_since_rotation > 0.5 and not self.pid_active and not self.auto_heading_active and not self.straight_forward_active:
                if not self.auto_lock_active:
                    self.auto_lock_active = True
                    self.locked_heading = self.current_imu_yaw
                    self.heading_pid_integral = 0.0
                    self.heading_pid_last_error = 0.0
                    self.get_logger().info(f'AUTO LOCK ACTIVATED - Locked heading: {self.locked_heading:.2f}')
        
        if self.pid_active and self.imu_offset is not None:
            pid_output = self.calculate_pid_control_to_zero()
            wz = pid_output
            
            error = abs(self.zero_deg - self.current_imu_yaw)
            if error > 180.0:
                error = 360.0 - error
            
            if error < 0.3:
                self.pid_active = False
                self.get_logger().info(f'MANUAL PID DEACTIVATED - Zero target reached! Error: {error:.2f} degrees')
        
        elif self.straight_forward_active and self.imu_offset is not None:
            if abs(msg.axes[3]) > 0.1:
                pid_output = self.calculate_pid_control_to_straight_heading()
                wz = pid_output
                
                error = abs(self.straight_forward_heading - self.current_imu_yaw)
                if error > 180.0:
                    error = 360.0 - error
                
                if error < 1.0:
                    pass
            else:
                self.straight_forward_active = False
                self.get_logger().info('STRAIGHT FORWARD DEACTIVATED - No forward movement')
        
        elif self.auto_heading_active and abs(msg.axes[2]) < 0.1:
            pid_output = self.calculate_pid_control_to_target()
            wz = pid_output
            
            error = abs(self.target_heading - self.current_imu_yaw)
            if error > 180.0:
                error = 360.0 - error
            
            if error < 3.0:
                self.auto_heading_active = False
                self.get_logger().info(f'AUTO HEADING DEACTIVATED - Target reached! Error: {error:.2f} degrees')
        
        elif self.auto_lock_active and self.imu_offset is not None:
            pid_output = self.calculate_pid_control_to_locked_heading()
            wz = pid_output
        
        elif abs(msg.axes[2]) > 0.1:
            if self.auto_heading_active:
                self.auto_heading_active = False
                self.get_logger().info('AUTO HEADING OVERRIDDEN - Manual rotation detected')
            if self.straight_forward_active:
                self.straight_forward_active = False
                self.get_logger().info('STRAIGHT FORWARD OVERRIDDEN - Manual rotation detected')
            if self.auto_lock_active:
                self.auto_lock_active = False
                self.get_logger().info('AUTO LOCK OVERRIDDEN - Manual rotation detected')
        
        wheel_speeds = self.calculate_wheel_speeds(vx, vy, wz)
        
        self.publish_wheel_speeds(wheel_speeds)

    def calculate_pid_control_to_zero(self):
        error = self.zero_deg - self.current_imu_yaw
        
        while error > 180.0:
            error -= 360.0
        while error < -180.0:
            error += 360.0
        
        self.zero_pid_integral += error
        derivative = error - self.zero_pid_last_error
        
        output = (self.zero_pid_kp * error + 
                 self.zero_pid_ki * self.zero_pid_integral + 
                 self.zero_pid_kd * derivative)
        
        output = max(-self.max_angular_speed, min(self.max_angular_speed, output))
        
        self.zero_pid_last_error = error
        
        self.get_logger().info(f'ZERO PID - Current: {self.current_imu_yaw:.2f}, Error: {error:.2f}, Output: {output:.2f}')
        
        return output

    def calculate_pid_control_to_locked_heading(self):
        error = self.locked_heading - self.current_imu_yaw
        
        while error > 180.0:
            error -= 360.0
        while error < -180.0:
            error += 360.0
        
        self.heading_pid_integral += error
        derivative = error - self.heading_pid_last_error
        
        output = (self.heading_pid_kp * error + 
                 self.heading_pid_ki * self.heading_pid_integral + 
                 self.heading_pid_kd * derivative)
        
        output = max(-self.max_angular_speed, min(self.max_angular_speed, output))
        
        self.heading_pid_last_error = error
        
        return output

    def calculate_pid_control_to_target(self):
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
        
        output = max(-self.max_angular_speed, min(self.max_angular_speed, output))
        
        self.heading_pid_last_error = error
        
        self.get_logger().info(f'TARGET PID - Target: {self.target_heading:.2f}, Current: {self.current_imu_yaw:.2f}, Error: {error:.2f}, Output: {output:.2f}')
        
        return output

    def calculate_pid_control_to_straight_heading(self):
        error = self.straight_forward_heading - self.current_imu_yaw
        
        while error > 180.0:
            error -= 360.0
        while error < -180.0:
            error += 360.0
        
        self.heading_pid_integral += error
        derivative = error - self.heading_pid_last_error
        
        output = (self.heading_pid_kp * error + 
                 self.heading_pid_ki * self.heading_pid_integral + 
                 self.heading_pid_kd * derivative)
        
        output = max(-self.max_angular_speed, min(self.max_angular_speed, output))
        
        self.heading_pid_last_error = error
        
        self.get_logger().info(f'STRAIGHT PID - Target: {self.straight_forward_heading:.2f}, Current: {self.current_imu_yaw:.2f}, Error: {error:.2f}, Output: {output:.2f}')
        
        return output

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