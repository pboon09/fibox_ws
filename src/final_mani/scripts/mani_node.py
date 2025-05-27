#!/usr/bin/python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from std_msgs.msg import Float64MultiArray
import threading
import time


class ManiNode(Node):
    def __init__(self):
        super().__init__('mani_node')

        self.max_speed = 230.0
        self.toggle_speed = 100.0
        
        # Button states for edge detection
        self.prev_square = False
        self.prev_triangle = False
        self.prev_circle = False
        
        # State variables
        self.shoot_active = False
        self.spin_active = False
        self.triangle_toggle_state = False  # False = off, True = on
        
        # Motor speeds [motor1, motor2, motor3]
        self.current_motor_speeds = [0.0, 0.0, 0.0]
        
        # Subscriber for joystick
        self.joy_sub = self.create_subscription(
            Joy,
            '/joy',
            self.joy_callback,
            10
        )
        
        # Publisher for motor speeds
        self.motor_pub = self.create_publisher(
            Float64MultiArray,
            '/mani_joy_out',
            10
        )
        
        # Timer for regular publishing
        self.publish_timer = self.create_timer(0.1, self.publish_motor_speeds)
        
        self.get_logger().info("Mani Node Started")
        self.get_logger().info("X: Shoot sequence | Y: Toggle motors 1&2 | B: Spin sequence")

    def joy_callback(self, msg):
        """Process joystick button inputs"""
        if len(msg.buttons) < 4:
            return
            
        # Xbox controller button mapping
        x_pressed = msg.buttons[3]         # X button 
        y_pressed = msg.buttons[4]         # Y button
        b_pressed = msg.buttons[1]         # B button
        
        # Edge detection for button presses
        if x_pressed and not self.prev_square:
            if not self.shoot_active and not self.spin_active:
                self.get_logger().info("X pressed - Starting SHOOT sequence")
                self.start_shoot_sequence()
        
        if y_pressed and not self.prev_circle:
            if not self.shoot_active and not self.spin_active:
                self.triangle_toggle()
        
        if b_pressed and not self.prev_triangle:
            if not self.shoot_active and not self.spin_active:
                self.get_logger().info("B pressed - Starting SPIN sequence")
                self.start_spin_sequence()
        
        # Update previous button states
        self.prev_square = x_pressed
        self.prev_triangle = b_pressed
        self.prev_circle = y_pressed

    def triangle_toggle(self):
        """Toggle triangle button functionality"""
        self.triangle_toggle_state = not self.triangle_toggle_state
        
        if self.triangle_toggle_state:
            self.get_logger().info("Y pressed - Motors 2&3 ON (-100)")
            self.set_motor_speeds([0.0, -self.toggle_speed.0, -self.toggle_speed.0])
        else:
            self.get_logger().info("Y pressed - All motors OFF")
            self.set_motor_speeds([0.0, 0.0, 0.0])

    def start_shoot_sequence(self):
        """Start the shoot sequence in a separate thread"""
        if self.shoot_active:
            return
        
        self.shoot_active = True
        shoot_thread = threading.Thread(target=self.shoot_sequence)
        shoot_thread.daemon = True
        shoot_thread.start()

    def shoot_sequence(self):
        """Execute the shoot sequence"""
        try:
            self.get_logger().info("SHOOT Step 1: Motors 2&3 = 200")
            self.set_motor_speeds([0.0, self.max_speed, self.max_speed])
            time.sleep(4.0)
            
            self.get_logger().info("SHOOT Step 2: All motors = 200")
            self.set_motor_speeds([-self.max_speed, self.max_speed, self.max_speed])
            time.sleep(2.0)
            
            self.get_logger().info("SHOOT Step 3: Motor2&3 = 0, Motor1 = -200")
            self.set_motor_speeds([self.max_speed, 0.0, 0.0])
            time.sleep(4.0)
            
            self.get_logger().info("SHOOT Step 4: All motors OFF")
            self.set_motor_speeds([0.0, 0.0, 0.0])
            
            self.get_logger().info("SHOOT sequence completed")
            
        except Exception as e:
            self.get_logger().error(f"Error in shoot sequence: {e}")
        finally:
            self.shoot_active = False

    def start_spin_sequence(self):
        """Start the spin sequence in a separate thread"""
        if self.spin_active:
            return
            
        self.spin_active = True
        spin_thread = threading.Thread(target=self.spin_sequence)
        spin_thread.daemon = True
        spin_thread.start()

    def spin_sequence(self):
        """Execute the spin sequence"""
        try:
            self.get_logger().info("SPIN Step 1: Motor3 = 200")
            self.set_motor_speeds([0.0, 0.0, self.max_speed])
            time.sleep(4.0)
            
            self.get_logger().info("SPIN Step 2: Motor1&3 = 200")
            self.set_motor_speeds([-self.max_speed, 0.0, self.max_speed])
            time.sleep(2.0)
            
            self.get_logger().info("SPIN Step 3: Motor1 = -200, Motor2&3 = 0")
            self.set_motor_speeds([self.max_speed, 0.0, 0.0])
            time.sleep(4.0)
            
            self.get_logger().info("SPIN Step 4: All motors OFF")
            self.set_motor_speeds([0.0, 0.0, 0.0])
            
            self.get_logger().info("SPIN sequence completed")
            
        except Exception as e:
            self.get_logger().error(f"Error in spin sequence: {e}")
        finally:
            self.spin_active = False

    def set_motor_speeds(self, speeds):
        """Set motor speeds [motor1, motor2, motor3]"""
        self.current_motor_speeds = speeds.copy()

    def publish_motor_speeds(self):
        """Publish current motor speeds"""
        msg = Float64MultiArray()
        msg.data = self.current_motor_speeds
        self.motor_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ManiNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()