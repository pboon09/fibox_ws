#!/usr/bin/env python3
# callback.py - Vision Pipeline for coordinate tracking

import math
import numpy as np
import cv2
from camera_handler import CameraHandler
from object_detector import ObjectDetector
from depth_estimator import DepthEstimator
import visualization as vis  # Import the visualization module

class VisionPipeline:
    """
    Class to handle the vision pipeline, using all components to process inputs and return coordinates
    """
    def __init__(self, camera_type='realsense', target_class='person', enable_visualization=False, enable_save_video=False):
        """
        Initialize the vision pipeline
        
        Args:
            camera_type (str): Type of camera to use ('realsense', 'webcam', or 'video')
            target_class (str): Target class to detect and track
            enable_visualization (bool): Whether to generate visualization images
        """
        # Initialize all components
        self.camera = CameraHandler(source_type=camera_type)
        if not self.camera.initialized:
            raise RuntimeError("Failed to initialize camera")
            
        self.detector = ObjectDetector()
        if not self.detector.initialized:
            self.camera.release()
            raise RuntimeError("Failed to initialize object detector")
            
        self.depth_estimator = DepthEstimator()
        
        # Get camera dimensions
        self.width, self.height = self.camera.get_dimensions()
        self.origin_x = self.width // 2
        self.origin_y = self.height // 2
        
        # Store configuration
        self.camera_type = camera_type
        self.target_class = target_class
        self.enable_visualization = enable_visualization
        self.enable_save_video = enable_save_video
        self.is_running = True
        
        # For FPS calculation
        self.frame_times = []
        
        print(f"Vision Pipeline initialized with:")
        print(f"- Camera type: {camera_type}")
        print(f"- Frame dimensions: {self.width}x{self.height}")
        print(f"- Target class: {target_class}")
        print(f"- Visualization: {'enabled' if enable_visualization else 'disabled'}")
        print(f"- Save video: {'enabled' if enable_save_video else 'disabled'}")

    def save_video(self, frames, filename, fps=30, codec='XVID', fourcc=None):
        """
        Save a list of frames as a video file.
        
        Args:
            frames (list): List of frames (numpy arrays) to save.
            filename (str): Output video file name.
            fps (int): Frames per second for the video.
            codec (str): Codec to use for video encoding (default: 'mp4v').
            fourcc (int): Optional precomputed fourcc code.
        """
        if self.enable_save_video:
            # if not frames:
            #     print("No frames to save.")
            #     return
            
            if fourcc is None:
                fourcc = cv2.VideoWriter_fourcc(*codec)
            
            # Ensure the filename has the correct extension
            if not filename.endswith('.mp4'):
                filename += '.mp4'
            
            # Get the dimensions of the first frame
            height, width = frames[0].shape[:2]
            
            # Create a VideoWriter object
            out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
            
            for frame in frames:
                # Ensure frame dimensions match the first frame
                if frame.shape[:2] != (height, width):
                    print("Error: Frame dimensions do not match. Skipping frame.")
                    continue
                out.write(frame)
            
            out.release()
            print(f"Video saved as {filename}")
        else:
            print("Video saving is disabled.")

    def process_single_frame(self):
        """
        Process a single frame and return the results.
        
        Returns:
            dict: Results including coordinates, angle, and detection status
            numpy.ndarray: Visualization image (or None if visualization is disabled)
        """
        if not self.is_running:
            print("Pipeline is stopped. Call start() to restart.")
            return None, None
            
        # Get frame from camera
        ret, color_frame, depth_frame = self.camera.get_frame()
        color_frame = cv2.flip(color_frame, 0)
        if not ret or color_frame is None:
            print("Failed to get frame from camera")
            return None, None
            
        # Process the frame to get coordinates
        depth_scale = self.camera.get_depth_scale() if self.camera_type == 'realsense' else 1.0
        
        # Start time for FPS calculation
        import time
        start_time = time.time()
        
        # Detect objects and find target
        detections = self.detector.detect(color_frame)
        target_detection = detections[0] if detections else None
        
        # Get depth information
        z_value = 0.0
        raw_depth = 0.0
        if target_detection:
            z_value, raw_depth = self.depth_estimator.get_depth(
                target_detection,
                depth_frame,
                depth_scale
            )
        
        # Process tracking data
        tracking_data = self.calculate_tracking_data(target_detection, (z_value, raw_depth))
        
        # End time for FPS calculation
        end_time = time.time()
        frame_time = end_time - start_time
        self.frame_times.append(frame_time)
        if len(self.frame_times) > 30:
            self.frame_times.pop(0)
        fps = 1.0 / (sum(self.frame_times) / len(self.frame_times)) if self.frame_times else 0
        
        # Generate visualization if enabled
        vis_img = None
        if self.enable_visualization and color_frame is not None:
            vis_img = color_frame.copy()
            
            # Use the visualization module
            vis.draw_coordinate_system(vis_img, self.origin_x, self.origin_y)
            
            if tracking_data['detected']:
                # Prepare visualization data
                detection_vis_data = {
                    'box': tracking_data['box'],
                    'conf': tracking_data['conf'],
                    'class_id': tracking_data['class_id'],
                    'center_x': tracking_data['center_x'],
                    'center_y': tracking_data['center_y']
                }
                
                # Draw detection results
                class_names = self.detector.get_class_names()
                vis.draw_detection_results(vis_img, detection_vis_data, class_names)
                vis.draw_origin_to_target_line(vis_img, self.origin_x, self.origin_y,
                                          tracking_data['center_x'], tracking_data['center_y'])
            
            # Draw info panel
            vis.draw_info_panel(vis_img, tracking_data)
            
            # Draw UI elements (mode, FPS)
            vis.draw_ui_elements(vis_img, self.camera_type.capitalize(), fps)
            
            # Draw instructions
            vis.draw_instructions(vis_img)
            
        return tracking_data, vis_img
    
    def calculate_tracking_data(self, target_detection, depth_result):
        """
        Calculate tracking data from detection and depth information
        
        Args:
            target_detection: Detection information from object detector
            depth_result: Tuple of (z_value, raw_depth) from depth estimator
            
        Returns:
            dict: Complete tracking data
        """
        # Initialize with default values
        tracking_data = {
            'detected': False,
            'box': None,
            'conf': 0.0,
            'class_id': -1,
            'center_x': None,
            'center_y': None,
            'rel_x': 0.0,
            'rel_y': 0.0,
            'z': 0.0,
            'raw_depth': 0.0,
            'dist': 0.0,
            'angle': 0.0
        }
        
        # If no detection, return default values
        if target_detection is None:
            return tracking_data
        
        # Extract basic detection info
        tracking_data['detected'] = True
        tracking_data['box'] = target_detection['box']
        tracking_data['conf'] = target_detection['conf']
        tracking_data['class_id'] = target_detection['class_id']
        tracking_data['center_x'] = target_detection['center_x']
        tracking_data['center_y'] = target_detection['center_y']
        
        # Extract depth information
        z_value, raw_depth = depth_result
        tracking_data['z'] = z_value
        tracking_data['raw_depth'] = raw_depth
        
        # Calculate relative coordinates
        rel_x = float(tracking_data['center_x'] - self.origin_x)
        rel_y = float(tracking_data['center_y'] - self.origin_y)
        tracking_data['rel_x'] = rel_x
        tracking_data['rel_y'] = rel_y
        
        # Calculate angle
        if z_value > 0:
            angle_rad = math.atan2(rel_x, z_value)
            tracking_data['angle'] = math.degrees(angle_rad)
        
        # Calculate 3D distance
        try:
            # Import calculate_3d_distance from utils
            from utils import calculate_3d_distance
            tracking_data['dist'] = calculate_3d_distance(rel_x, rel_y, z_value)
        except ImportError:
            # Fallback if utils module is not available
            tracking_data['dist'] = math.sqrt(rel_x**2 + rel_y**2 + z_value**2)
        
        return tracking_data
    
    def get_coordinates(self):
        """
        Get current coordinates as a simple list [x, y, z, angle]
        
        Returns:
            list: [x, y, z, angle] or [0, 0, 0, 0] if no target detected
        """
        tracking_data, _ = self.process_single_frame()
        if tracking_data is None or not tracking_data['detected']:
            return [0, 0, 0, 0]
            
        return [
            tracking_data['rel_x'],
            tracking_data['rel_y'],
            tracking_data['z'],
            tracking_data['angle']
        ]
    
    def stop(self):
        """
        Stop the pipeline and release resources
        """
        self.is_running = False
        if hasattr(self, 'camera') and self.camera:
            self.camera.release()
        print("Vision pipeline stopped and resources released")
        
    def __del__(self):
        """
        Destructor to ensure resources are released
        """
        self.stop()

# Example usage
if __name__ == "__main__":
    try:
        # Create the pipeline with visualization enabled
        pipeline = VisionPipeline(camera_type='realsense', enable_visualization=True)
        
        print("Processing frames. Press 'q' to quit.")
        while True:
            # Process a single frame
            tracking_data, vis_img = pipeline.process_single_frame()
            # Display results
            if tracking_data and tracking_data['detected']:
                print(f"Target: X={tracking_data['rel_x']:.1f}, Y={tracking_data['rel_y']:.1f}, "
                      f"Z={tracking_data['z']:.2f}m, Angle={tracking_data['angle']:.1f}°")
            else:
                print("No target detected")
                
            # Show visualization if available
            if vis_img is not None:
                cv2.imshow("Vision Pipeline", vis_img)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
            
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        # Clean up
        if 'pipeline' in locals():
            pipeline.stop()
        cv2.destroyAllWindows()