#!/usr/bin/python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from std_msgs.msg import Float64MultiArray, Float64, Int32
import threading
import time
from final_mani.Callback import VisionPipeline
import cv2
import numpy as np


class ManiNode(Node):
    def __init__(self):
        super().__init__('mani_node')

        self.enable_save_video = False #Change here
        self.video_save_interval = 300
        self.frames = []
        self.frame_count = 0
        self.output_folder = None

        self.shoot_speed = 150.0
        self.spin_speed = 255.0
        self.push_speed = 255.0
        self.toggle_speed = 120.0
        
        self.prev_square = False
        self.prev_triangle = False
        self.prev_circle = False
        self.prev_a = False
        
        self.shoot_active = False
        self.spin_active = False
        self.realsense_active = False 
        self.triangle_toggle_state = False
        
        self.current_motor_speeds = [0.0, 0.0, 0.0]
        
        self.pipeline = None
        self.vision_thread_active = False
        self.latest_tracking_data = None
        self.latest_vis_img = None
        self.vision_lock = threading.Lock()
        
        if self.enable_save_video:
            self.setup_video_saving()
        
        self.start_vision_thread()
        
        self.joy_sub = self.create_subscription(
            Joy,
            '/joy',
            self.joy_callback,
            10
        )
        
        self.motor_pub = self.create_publisher(
            Float64MultiArray,
            '/mani_joy_out',
            10
        )
        
        self.target_heading_pub = self.create_publisher(
            Float64,
            '/target_head_robot',
            10
        )
        
        self.pilot_lamp_pub = self.create_publisher(
            Int32,
            '/pilot_lamp',
            10
        )
        
        self.publish_timer = self.create_timer(0.01, self.publish_motor_speeds)
        self.pilot_timer = self.create_timer(0.1, self.update_pilot_lamp)
        
        self.get_logger().info("Mani Node Started")
        self.get_logger().info("X: Shoot sequence | Y: Toggle motors 1&2 | B: Spin sequence | A: RealSense sequence (only when detected)")
        if self.enable_save_video:
            self.get_logger().info(f"Video saving enabled - Output folder: {self.output_folder}")
        else:
            self.get_logger().info("Video saving disabled")

    def setup_video_saving(self):
        import os
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        self.output_folder = f"output_frames_{timestamp}"
        try:
            os.makedirs(self.output_folder, exist_ok=True)
            self.get_logger().info(f"Created output directory: {os.path.abspath(self.output_folder)}")
        except Exception as e:
            self.get_logger().error(f"Failed to create output directory: {e}")
            self.enable_save_video = False

    def start_vision_thread(self):
        self.vision_thread_active = True
        vision_thread = threading.Thread(target=self.vision_processing_loop)
        vision_thread.daemon = True
        vision_thread.start()

    def vision_processing_loop(self):
        try:
            self.pipeline = VisionPipeline(camera_type='realsense', enable_visualization=True, enable_save_video=self.enable_save_video)
            self.get_logger().info("Vision pipeline initialized successfully in separate thread")
        except Exception as e:
            self.get_logger().error(f"Failed to initialize vision pipeline: {e}")
            self.pipeline = None
            return

        while self.vision_thread_active and rclpy.ok():
            try:
                if self.pipeline is None:
                    time.sleep(0.1)
                    continue
                    
                tracking_data, vis_img = self.pipeline.process_single_frame()
                
                if tracking_data is not None and vis_img is not None:
                    with self.vision_lock:
                        self.latest_tracking_data = tracking_data
                        self.latest_vis_img = vis_img
                
                time.sleep(0.05)
                
            except Exception as e:
                self.get_logger().error(f"Error in vision processing loop: {e}")
                time.sleep(0.5)

    def update_pilot_lamp(self):
        try:
            with self.vision_lock:
                tracking_data = self.latest_tracking_data
                vis_img = self.latest_vis_img
            
            if tracking_data is None:
                return
            
            pilot_msg = Int32()
            pilot_msg.data = 1 if tracking_data.get('detected', False) else 0
            self.pilot_lamp_pub.publish(pilot_msg)
            
            if vis_img is not None:
                cv2.imshow("Debug View", vis_img)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.get_logger().info("'q' pressed in OpenCV window - shutting down")
                    if self.enable_save_video:
                        self.save_final_video()
                    rclpy.shutdown()
            
            if self.enable_save_video and vis_img is not None:
                self.save_frame_and_video(vis_img)
            
        except Exception as e:
            self.get_logger().error(f"Error in update_pilot_lamp: {e}")

    def save_frame_and_video(self, vis_img):
        if not self.enable_save_video or vis_img is None:
            return
            
        try:
            import os
            self.frame_count += 1
            
            if self.frame_count % 10 == 0:
                frame_filename = os.path.join(self.output_folder, f"frame_{cv2.getTickCount()}.jpg")
                success = cv2.imwrite(frame_filename, vis_img)
                
                if success:
                    self.get_logger().info(f"Frame saved: {frame_filename} (Count: {self.frame_count})")
                else:
                    self.get_logger().error(f"Failed to save frame: {frame_filename}")
            
            self.frames.append(vis_img.copy())
            
            if self.frame_count % self.video_save_interval == 0:
                video_filename = f"output_{self.frame_count}"
                self.get_logger().info(f"Saving video at frame {self.frame_count}")
                if self.pipeline and hasattr(self.pipeline, 'save_video'):
                    self.pipeline.save_video(self.frames, video_filename, fps=15)
                    self.get_logger().info(f"Video saved: {video_filename}.mp4")
                
        except Exception as e:
            self.get_logger().error(f"Error saving frame/video: {e}")

    def save_final_video(self):
        if self.enable_save_video and self.frames and len(self.frames) > 0:
            try:
                self.get_logger().info(f"Saving final video with {len(self.frames)} frames...")
                if self.pipeline and hasattr(self.pipeline, 'save_video'):
                    self.pipeline.save_video(self.frames, "output_final", fps=15)
                    self.get_logger().info("Final video saved as output_final.mp4")
                else:
                    self.get_logger().error("Pipeline save_video method not available")
            except Exception as e:
                self.get_logger().error(f"Error saving final video: {e}")

    def joy_callback(self, msg):
        if len(msg.buttons) < 5:
            return
            
        a_pressed = msg.buttons[0]
        b_pressed = msg.buttons[1]
        x_pressed = msg.buttons[3]
        y_pressed = msg.buttons[4]
        
        if a_pressed and not self.prev_a:
            if not self.shoot_active and not self.spin_active and not self.realsense_active:
                self.get_logger().info("A pressed - Checking for target detection")
                self.check_and_execute_realsense()
            else:
                self.get_logger().info("A pressed - Another sequence is already active")
        
        if x_pressed and not self.prev_square:
            if not self.shoot_active and not self.spin_active and not self.realsense_active:
                self.get_logger().info("X pressed - Starting SHOOT sequence")
                self.start_shoot_sequence()
        
        if y_pressed and not self.prev_circle:
            if not self.shoot_active and not self.spin_active and not self.realsense_active:
                self.triangle_toggle()
        
        if b_pressed and not self.prev_triangle:
            if not self.shoot_active and not self.spin_active and not self.realsense_active:
                self.get_logger().info("B pressed - Starting SPIN sequence")
                self.start_spin_sequence()
        
        self.prev_a = a_pressed
        self.prev_square = x_pressed
        self.prev_triangle = b_pressed
        self.prev_circle = y_pressed

    def check_and_execute_realsense(self):
        try:
            with self.vision_lock:
                tracking_data = self.latest_tracking_data
            
            if tracking_data and tracking_data.get('detected', False):
                self.get_logger().info("Target detected - Starting RealSense sequence")
                self.realsense_sequence(tracking_data)
            else:
                self.get_logger().info("A pressed - No target detected, sequence not started")
                
        except Exception as e:
            self.get_logger().error(f"Error in detection check: {e}")

    def triangle_toggle(self):
        self.triangle_toggle_state = not self.triangle_toggle_state
        
        if self.triangle_toggle_state:
            self.get_logger().info("Y pressed - Motors 2&3 ON (-120)")
            self.set_motor_speeds([0.0, -self.toggle_speed, -self.toggle_speed])
        else:
            self.get_logger().info("Y pressed - All motors OFF")
            self.set_motor_speeds([0.0, 0.0, 0.0])

    def realsense_sequence(self, tracking_data):
        try:
            self.realsense_active = True
            
            x = tracking_data.get('rel_x', 0)
            y = tracking_data.get('rel_y', 0)
            z = tracking_data.get('z', 0)
            angle = tracking_data.get('angle', 0)
            confidence = tracking_data.get('confidence', 0)
            
            self.get_logger().info(f"RealSense: Executing sequence with data: x={x}, y={y}, z={z}, angle={angle}, confidence={confidence}")
            
            target_heading_msg = Float64()
            target_heading_msg.data = float(angle)
            self.target_heading_pub.publish(target_heading_msg)
            self.get_logger().info(f"RealSense Step 1: Target heading sent = {angle}")
            
            min_speed = 50.0
            shoot_speed = 200.0
            if z > 0:
                computed_speed = max(min_speed, min(shoot_speed, shoot_speed * (1.0 / z) * 100))
            else:
                computed_speed = min_speed
            
            self.get_logger().info(f"RealSense Step 2: Computed speed from distance z={z} -> speed={computed_speed}")
            
            self.get_logger().info(f"RealSense Step 3: Motors 2&3 spin at {computed_speed} for 4 seconds")
            self.set_motor_speeds([0.0, computed_speed, computed_speed])
            time.sleep(4.0)
            
            self.get_logger().info("RealSense Step 4: Motor 1 ON")
            self.set_motor_speeds([-self.push_speed, computed_speed, computed_speed])
            
            self.get_logger().info("RealSense Step 5: Wait 4 seconds")
            time.sleep(4.0)
            
            self.get_logger().info("RealSense Step 6: Motor1 = 255, Motor2&3 = 0")
            self.set_motor_speeds([self.push_speed, 0.0, 0.0])
            time.sleep(4.0)
            
            self.get_logger().info("RealSense Step 7: All motors OFF")
            self.set_motor_speeds([0.0, 0.0, 0.0])
            
            target_heading_msg.data = 0.0
            self.target_heading_pub.publish(target_heading_msg)
            self.get_logger().info("RealSense Step 8: Heading reset to 0")
            
            self.get_logger().info("RealSense sequence completed successfully")
            
        except Exception as e:
            self.get_logger().error(f"Error in RealSense sequence: {e}")
        finally:
            self.realsense_active = False

    def start_shoot_sequence(self):
        if self.shoot_active:
            return
        
        self.shoot_active = True
        shoot_thread = threading.Thread(target=self.shoot_sequence)
        shoot_thread.daemon = True
        shoot_thread.start()

    def shoot_sequence(self):
        try:
            self.get_logger().info("SHOOT Step 1: Motors 2&3 = 255")
            self.set_motor_speeds([0.0, self.shoot_speed, self.shoot_speed])
            time.sleep(4.0)
            
            self.get_logger().info("SHOOT Step 2: All motors = 255")
            self.set_motor_speeds([-self.push_speed, self.shoot_speed, self.shoot_speed])
            time.sleep(2.0)
            
            self.get_logger().info("SHOOT Step 3: Motor2&3 = 0, Motor1 = 255")
            self.set_motor_speeds([self.push_speed, 0.0, 0.0])
            time.sleep(4.0)
            
            self.get_logger().info("SHOOT Step 4: All motors OFF")
            self.set_motor_speeds([0.0, 0.0, 0.0])
            
            self.get_logger().info("SHOOT sequence completed")
            
        except Exception as e:
            self.get_logger().error(f"Error in shoot sequence: {e}")
        finally:
            self.shoot_active = False

    def start_spin_sequence(self):
        if self.spin_active:
            return
            
        self.spin_active = True
        spin_thread = threading.Thread(target=self.spin_sequence)
        spin_thread.daemon = True
        spin_thread.start()

    def spin_sequence(self):
        try:
            self.get_logger().info("SPIN Step 1: Motor3 = 255")
            self.set_motor_speeds([0.0, 0.0, self.spin_speed])
            time.sleep(4.0)
            
            self.get_logger().info("SPIN Step 2: Motor1&3 = 255")
            self.set_motor_speeds([-self.push_speed, 0.0, self.spin_speed])
            time.sleep(1.0)
            
            self.get_logger().info("SPIN Step 3: Motor1 = 255, Motor2&3 = Toggle")
            self.set_motor_speeds([self.push_speed, -self.toggle_speed, -self.toggle_speed])
            time.sleep(4.0)

            self.get_logger().info("SPIN Step 4 All motors OFF")
            self.set_motor_speeds([0.0, 0.0, 0.0])
            
            self.get_logger().info("SPIN sequence completed")
            
        except Exception as e:
            self.get_logger().error(f"Error in spin sequence: {e}")
        finally:
            self.spin_active = False

    def set_motor_speeds(self, speeds):
        self.current_motor_speeds = speeds.copy()

    def publish_motor_speeds(self):
        msg = Float64MultiArray()
        msg.data = self.current_motor_speeds
        self.motor_pub.publish(msg)

    def cleanup(self):
        self.vision_thread_active = False
        cv2.destroyAllWindows()
        if self.enable_save_video:
            self.save_final_video()
        if self.pipeline:
            try:
                self.pipeline.stop()
            except:
                pass

    def __del__(self):
        self.cleanup()


def main(args=None):
    rclpy.init(args=args)
    node = ManiNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nExiting...")
        node.get_logger().info("Received interrupt signal, cleaning up...")
    finally:
        node.cleanup()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()