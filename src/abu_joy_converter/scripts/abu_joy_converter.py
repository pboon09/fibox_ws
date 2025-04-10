#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist

class AbuJoyConverter(Node):
    def __init__(self):
        super().__init__('abu_joy_converter')
        
        self.declare_parameter('input_topic', '/joy')
        self.declare_parameter('output_topic', '/cmd_vel')
        
        joy_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        cmd_vel_topic = self.get_parameter('output_topic').get_parameter_value().string_value
        self.linear_axis = 1
        self.lateral_axis = 0
        self.left_trigger = 2
        self.right_trigger = 5
        self.max_linear = 0.5
        self.max_lateral = 0.5
        self.max_angular = 1.0
        
        self.cmd_vel_pub = self.create_publisher(Twist, cmd_vel_topic, 10)
        
        self.joy_sub = self.create_subscription(Joy, joy_topic, self.joy_callback, 10)
        
        self.get_logger().info(
            f"abu_joy_converter node started\n"
            f"Subscribing to {joy_topic}\n"
            f"Publishing to {cmd_vel_topic}")

    def joy_callback(self, msg):
        twist = Twist()

        # Get movement values (-1.0 to 1.0 range)
        linear = -msg.axes[self.linear_axis] if self.linear_axis < len(msg.axes) else 0.0
        lateral = msg.axes[self.lateral_axis] if self.lateral_axis < len(msg.axes) else 0.0
        
        # Get trigger values (typically 1.0 to -1.0, where unpressed is -1.0)
        lt = (msg.axes[self.left_trigger] + 1.0) / 2 if self.left_trigger < len(msg.axes) else 0.0  # Convert to 0-1 range
        rt = (msg.axes[self.right_trigger] + 1.0) / 2 if self.right_trigger < len(msg.axes) else 0.0 # Convert to 0-1 range
        
        angular = rt - lt
        
        twist.linear.x = linear * self.max_linear
        twist.linear.y = lateral * self.max_lateral
        twist.angular.z = angular * self.max_angular
        
        self.cmd_vel_pub.publish(twist)
        self.get_logger().debug(
            f"Published Twist: vx={twist.linear.x:.2f}, vy={twist.linear.y:.2f}, wz={twist.angular.z:.2f}"
        )

def main(args=None):
    rclpy.init(args=args)
    node = AbuJoyConverter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()