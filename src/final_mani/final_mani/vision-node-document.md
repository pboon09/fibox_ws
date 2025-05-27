# High-Level Vision Node Architecture
## ROS-Based Computer Vision System

## 1. Overview

This document outlines the architecture and implementation strategy for developing a high-level ROS node for the vision system. The high-level node will encapsulate the existing vision code and provide a standardized interface for other system components to interact with the vision functionality.

### 1.1 Input-Output Summary

**Inputs:**
- RealSense camera data (RGB and depth streams)
- YOLO object detection model
- Regression model for depth estimation

**Outputs:**
- พิกัด (x, y, z)
  - x: left-right position relative to camera center
  - y: up-down position relative to camera center
  - z: depth from camera
- Angle: Horizontal angle (XZ plane) for target tracking

The existing vision system provides real-time 3D object detection and tracking capabilities, supporting multiple camera types (RealSense, webcam, video files) and utilizing YOLOv8 for object detection with optional depth estimation through multiple methods.

## 2. System Architecture

### 2.1 Components Diagram

```
+---------------------------+
|    High-Level Vision Node |
+---------------------------+
|                           |
|  +---------------------+  |
|  |  Vision Core System |  |
|  |---------------------|  |
|  | - CameraHandler     |  |
|  | - ObjectDetector    |  |    +----------------+
|  | - DepthEstimator    |<---->| Configuration   |
|  | - Tracker           |  |    | Parameters     |
|  | - Visualization     |  |    +----------------+
|  +---------------------+  |
|                           |
|  +---------------------+  |
|  |   ROS Interface     |  |
|  |---------------------|  |
|  | - Publishers        |  |    +----------------+
|  | - Subscribers       |<---->| Other ROS Nodes |
|  | - Services          |  |    +----------------+
|  | - Action Servers    |  |
|  +---------------------+  |
|                           |
+---------------------------+
```

### 2.2 Integration Points

The high-level node will interface with:
- ROS topics for publishing detection results and receiving control commands
- ROS parameters for configuration
- ROS services for on-demand functionality
- ROS action servers for long-running operations

## 3. ROS Node Implementation

### 3.1 Core Input-Output Processing Pipeline

The most essential function of the vision node is to transform inputs (RealSense data, detection models) into the required outputs (x, y, z coordinates and angle). Here's how this pipeline works:

```python
def process_inputs_to_outputs(self, color_frame, depth_frame, depth_scale):
    """
    Process camera inputs to produce coordinate outputs
    
    Args:
        color_frame: RGB image from RealSense
        depth_frame: Depth image from RealSense
        depth_scale: Scaling factor for depth values
        
    Returns:
        dict: Contains coordinates (x,y,z), angle, and detection status
    """
    # Detection phase
    detections = self.detector.detect(color_frame)
    target_detection = detections[0] if detections else None
    
    # Initialize result dictionary
    result = {
        'detected': False,
        'x': 0.0,  # Left-right (positive is right)
        'y': 0.0,  # Up-down (positive is down)
        'z': 0.0,  # Depth (distance from camera)
        'angle': 0.0,  # Horizontal angle in XZ plane
        'confidence': 0.0
    }
    
    if not target_detection:
        return result
        
    # Extract target information
    box = target_detection['box']  # [x1, y1, x2, y2]
    confidence = target_detection['conf']
    center_x = (box[0] + box[2]) // 2
    center_y = (box[1] + box[3]) // 2
    
    # Calculate relative coordinates from image center
    origin_x = self.width // 2
    origin_y = self.height // 2
    rel_x = float(center_x - origin_x)  # + is right
    rel_y = float(center_y - origin_y)  # + is down
    
    # Get depth using appropriate method
    z_value, raw_depth = self.depth_estimator.get_depth(
        target_detection,
        depth_frame,
        depth_scale
    )
    
    # Calculate horizontal angle in XZ plane
    # (0° is straight ahead, positive is right, negative is left)
    angle_deg = 0.0
    if z_value > 0:
        angle_rad = math.atan2(rel_x, z_value)
        angle_deg = math.degrees(angle_rad)
    
    # Populate result
    result['detected'] = True
    result['x'] = rel_x
    result['y'] = rel_y
    result['z'] = z_value
    result['angle'] = angle_deg
    result['confidence'] = confidence
    
    return result
```

### 3.2 Node Structure

```python
#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import math
from cv_bridge import CvBridge

# ROS message imports
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, Point
from std_msgs.msg import Bool, String, Float32MultiArray
from vision_msgs.msg import Detection2DArray, ObjectHypothesisWithPose

# Import existing vision system modules
from camera_handler import CameraHandler
from object_detector import ObjectDetector
from depth_estimator import DepthEstimator
from tracker import Tracker
import visualization as vis
import config

class VisionNode:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('high_level_vision_node')
        
        # Load parameters
        self.load_parameters()
        
        # Initialize CV Bridge
        self.bridge = CvBridge()
        
        # Initialize vision system components
        self.initialize_vision_system()
        
        # Set up publishers and subscribers
        self.setup_interfaces()
        
        # Start main processing loop
        rospy.Timer(rospy.Duration(1.0/30.0), self.process_frame)
        
        rospy.loginfo("Vision node initialized successfully")
    
    def load_parameters(self):
        # Load parameters from ROS parameter server
        self.camera_type = rospy.get_param('~camera_type', 'realsense')  # 'realsense', 'webcam', or 'video'
        self.target_class = rospy.get_param('~target_class', config.TARGET_CLASS_NAME)
        self.conf_threshold = rospy.get_param('~conf_threshold', config.CONF_THRESHOLD)
        self.publish_visualization = rospy.get_param('~publish_visualization', True)
        # Add more parameters as needed
    
    def initialize_vision_system(self):
        # Initialize vision components
        self.camera = CameraHandler(source_type=self.camera_type)
        if not self.camera.initialized:
            rospy.logerr("Camera initialization failed")
            rospy.signal_shutdown("Camera initialization failed")
            return
            
        self.detector = ObjectDetector()
        if not self.detector.initialized:
            rospy.logerr("Object detector initialization failed")
            rospy.signal_shutdown("Object detector initialization failed")
            return
            
        self.depth_estimator = DepthEstimator()
        self.tracker = Tracker()
        
        # Get frame dimensions for origin
        self.width, self.height = self.camera.get_dimensions()
        self.origin_x = self.width // 2
        self.origin_y = self.height // 2
        
        rospy.loginfo(f"Vision system initialized with camera type: {self.camera_type}")
        rospy.loginfo(f"Target class: {self.target_class}")
        rospy.loginfo(f"Frame dimensions: {self.width}x{self.height}")
    
    def setup_interfaces(self):
        # Publishers
        self.coordinates_pub = rospy.Publisher('vision/coordinates', Float32MultiArray, queue_size=10)
        self.pose_pub = rospy.Publisher('vision/target_pose', PoseStamped, queue_size=10)
        self.detection_pub = rospy.Publisher('vision/detections', Detection2DArray, queue_size=10)
        self.status_pub = rospy.Publisher('vision/status', String, queue_size=10)
        
        # Visualization publishers
        if self.publish_visualization:
            self.vis_pub = rospy.Publisher('vision/visualization', Image, queue_size=10)
            self.marker_pub = rospy.Publisher('vision/target_marker', Marker, queue_size=10)
        
        # Subscribers
        rospy.Subscriber('vision/enable', Bool, self.enable_callback)
        rospy.Subscriber('vision/set_target_class', String, self.set_target_callback)
        
        # Services
        self.enable_service = rospy.Service('vision/toggle', Bool, self.toggle_service_handler)
        self.get_coordinates_service = rospy.Service('vision/get_coordinates', GetCoordinates, self.get_coordinates_handler)
        
        # TF2 broadcaster for coordinate frames
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        
        # Add more interfaces as needed
    
    def enable_callback(self, msg):
        self.enabled = msg.data
        rospy.loginfo(f"Vision processing {'enabled' if self.enabled else 'disabled'}")
    
    def set_target_callback(self, msg):
        self.target_class = msg.data
        rospy.loginfo(f"Set target class to: {self.target_class}")
    
    def toggle_service_handler(self, req):
        self.enabled = req.data
        return self.enabled
    
    def process_frame(self, event=None):
        if not self.enabled:
            return
            
        # 1. Get frame from camera
        ret, color_frame, depth_frame = self.camera.get_frame()
        if not ret or color_frame is None:
            rospy.logwarn("Failed to get frame")
            return
            
        # 2. Detect objects
        detections = self.detector.detect(color_frame)
        
        # 3. Select target (first detection that matches our criteria)
        target_detection_info = detections[0] if detections else None
        
        # 4. Estimate depth
        depth_scale = self.camera.get_depth_scale() if self.camera_type == 'realsense' else 1.0
        current_z, raw_depth = self.depth_estimator.get_depth(
            target_detection_info,
            depth_frame if self.camera_type == 'realsense' else None,
            depth_scale
        )
        depth_result = (current_z, raw_depth)
        
        # 5. Track and calculate relative coordinates
        tracking_data = self.tracker.process_detection(
            target_detection_info,
            depth_result,
            self.origin_x,
            self.origin_y
        )
        
        # 6. Publish results
        self.publish_results(tracking_data, color_frame)
    
    def publish_results(self, tracking_data, color_frame):
        # Publish target pose and coordinates
        if tracking_data['detected']:
            # 1. Publish custom coordinates message with Thai convention
            coord_msg = Float32MultiArray()
            coord_msg.data = [
                tracking_data['rel_x'],    # x (left-right, + is right)
                tracking_data['rel_y'],    # y (up-down, + is down)
                tracking_data['z'],        # z (depth in meters)
                tracking_data['angle']     # angle in degrees (horizontal plane)
            ]
            self.coordinates_pub.publish(coord_msg)
            
            # 2. Create and publish standard ROS pose message
            pose_msg = PoseStamped()
            pose_msg.header.stamp = rospy.Time.now()
            pose_msg.header.frame_id = "camera_frame"
            
            # Convert to ROS coordinate convention
            # In ROS: X forward, Y left, Z up
            # Convert from camera coordinates: Z forward, X right, Y down
            pose_msg.pose.position.x = tracking_data['z']           # forward (camera Z -> ROS X)
            pose_msg.pose.position.y = -tracking_data['rel_x'] / 100.0  # left (camera X -> ROS -Y) with scaling to meters
            pose_msg.pose.position.z = -tracking_data['rel_y'] / 100.0  # up (camera Y -> ROS -Z) with scaling to meters
            
            # Set orientation based on angle
            angle_rad = math.radians(tracking_data['angle'])
            # Create quaternion from yaw angle (rotation around vertical axis)
            quat = tf_transformations.quaternion_from_euler(0, 0, angle_rad)
            pose_msg.pose.orientation.x = quat[0]
            pose_msg.pose.orientation.y = quat[1]
            pose_msg.pose.orientation.z = quat[2]
            pose_msg.pose.orientation.w = quat[3]
            
            self.pose_pub.publish(pose_msg)
            
            # 3. Publish status
            self.status_pub.publish(String(data="TRACKING"))
        else:
            # Publish null/empty coordinates
            coord_msg = Float32MultiArray()
            coord_msg.data = [0.0, 0.0, 0.0, 0.0]  # x, y, z, angle all zero
            self.coordinates_pub.publish(coord_msg)
            
            # Publish status when no target is detected
            self.status_pub.publish(String(data="NO_TARGET"))
        
        # Publish visualization if enabled
        if self.publish_visualization:
            # Create visualization image
            display_image = color_frame.copy()
            
            # Draw coordinate system
            vis.draw_coordinate_system(display_image, self.origin_x, self.origin_y)
            
            # Draw detection results if available
            if tracking_data['detected']:
                detection_vis_data = {
                    'box': tracking_data['box'],
                    'conf': tracking_data['conf'],
                    'class_id': tracking_data['class_id'],
                    'center_x': tracking_data['center_x'],
                    'center_y': tracking_data['center_y']
                }
                class_names = self.detector.get_class_names()
                vis.draw_detection_results(display_image, detection_vis_data, class_names)
                vis.draw_origin_to_target_line(display_image, self.origin_x, self.origin_y,
                                               tracking_data['center_x'], tracking_data['center_y'])
            
            # Draw info panel
            vis.draw_info_panel(display_image, tracking_data)
            
            # Calculate FPS (simplified for example)
            fps = 30.0
            vis.draw_ui_elements(display_image, self.camera_type.capitalize(), fps)
            
            # Convert to ROS image and publish
            try:
                img_msg = self.bridge.cv2_to_imgmsg(display_image, "bgr8")
                self.vis_pub.publish(img_msg)
            except Exception as e:
                rospy.logerr(f"Error converting/publishing visualization: {e}")
    
    def shutdown(self):
        # Clean up resources
        if hasattr(self, 'camera') and self.camera:
            self.camera.release()
        rospy.loginfo("Vision node shutting down")

if __name__ == '__main__':
    try:
        node = VisionNode()
        rospy.on_shutdown(node.shutdown)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Error in vision node: {e}")
```

### 3.3 Coordinate Systems & ROS Message Types

#### Coordinate System Conventions

The node handles two coordinate systems:

1. **Camera/Thai Vision Convention:**
   - Origin at image center
   - X: right is positive (+), left is negative (-)
   - Y: down is positive (+), up is negative (-)
   - Z: depth/distance from camera in meters
   - Angle: 0° straight ahead, right is positive, left is negative

2. **ROS Standard Convention:**
   - Origin at camera optical center
   - X: forward (camera's Z)
   - Y: left (camera's -X)
   - Z: up (camera's -Y)
   - Orientation: quaternion representing angle in 3D space

#### ROS Message Types

1. Input:
   - `sensor_msgs/Image` - For raw camera frames if subscribing to camera topic
   - `sensor_msgs/CameraInfo` - Camera calibration and metadata
   - `std_msgs/Bool` - For enabling/disabling processing
   - `std_msgs/String` - For setting target class

2. Output:
   - **Primary Output:** `std_msgs/Float32MultiArray` - For coordinates [x, y, z, angle] in Thai convention
   - `geometry_msgs/PoseStamped` - Target position and orientation in ROS convention
   - `vision_msgs/Detection2DArray` - Detailed detection information
   - `visualization_msgs/Marker` - 3D visualization of target for RViz
   - `sensor_msgs/Image` - Visualization output
   - `std_msgs/String` - Status messages

3. Custom Service:
   - `GetCoordinates` - Service for on-demand coordinate retrieval
     ```
     # Request (empty)
     ---
     # Response
     float32 x
     float32 y
     float32 z
     float32 angle
     bool detected
     ```

### 3.4 ROS Parameters

The node should be configurable through the following parameters:

```yaml
high_level_vision_node:
  # Camera configuration
  camera_type: "realsense"  # Options: "realsense", "webcam", "video"
  camera_id: 0
  video_path: ""
  frame_width: 640
  frame_height: 480
  
  # Detection configuration
  model_path: "best.pt"
  target_class: "person"  # Target class to track
  conf_threshold: 0.15
  
  # Depth estimation
  use_regression_model: true  # Whether to use the regression model for depth
  depth_model_path: "final_calibrated_depth_model_outdoor_v2"
  
  # Coordinate system options
  pixel_to_meter_factor: 0.01  # Conversion factor from pixels to meters for x,y
  coordinate_frame_convention: "thai"  # "thai" or "ros" - determines primary output format
  
  # Processing options
  publish_rate: 30
  publish_visualization: true
  
  # Coordinate frames
  camera_frame: "camera_link"
  target_frame: "target"
  
  # Advanced options
  use_tracking_filter: false  # Enable Kalman filter for smoother tracking
  filter_process_noise: 0.01
  filter_measurement_noise: 0.1
```

### 3.5 ROS Topic Structure

```
/vision/
  |
  +-- coordinates          # [x,y,z,angle] in Thai convention (Float32MultiArray)
  |
  +-- target_pose          # Target pose in ROS convention (PoseStamped)
  |
  +-- status               # "TRACKING" or "NO_TARGET" (String)
  |
  +-- detections           # Full detection information (Detection2DArray)
  |
  +-- visualization/
  |     |
  |     +-- image          # Visualized camera feed with overlays (Image)
  |     |
  |     +-- target_marker  # 3D marker for RViz (Marker)
  |
  +-- parameters/          # Parameters accessible via dynamic_reconfigure
        |
        +-- set_parameters # Parameter update service
```

## 4. Implementation Plan

### 4.1 Phase 1: Core ROS Node

1. Set up basic ROS node structure
2. Implement parameter loading
3. Initialize vision system components
4. Create simple publishers for detection results

### 4.2 Phase 2: Integration with Existing Code

1. Integrate camera input (support both direct camera and ROS topics)
2. Connect object detection pipeline
3. Implement depth estimation
4. Add tracking functionality

### 4.3 Phase 3: Full ROS Interface

1. Implement all publishers and subscribers
2. Add service interfaces
3. Create action servers for long-running operations
4. Implement visualization publishing

### 4.4 Phase 4: Testing and Optimization

1. Develop unit tests for ROS interfaces
2. Performance testing and optimization
3. Documentation and examples

## 5. Key Code Modifications

### 5.1 Camera Handler Modifications

Adapt the `CameraHandler` class to optionally subscribe to ROS camera topics:

```python
def _setup_ros_camera(self):
    """Setup ROS camera subscriber if using ROS camera source"""
    self.image_sub = rospy.Subscriber(
        self.config.ROS_CAMERA_TOPIC, 
        Image, 
        self.image_callback,
        queue_size=1
    )
    self.camera_info_sub = rospy.Subscriber(
        self.config.ROS_CAMERA_INFO_TOPIC,
        CameraInfo,
        self.camera_info_callback,
        queue_size=1
    )
    # Set up buffer for latest frame
    self.latest_frame = None
    self.latest_frame_time = None
    return True  # Initially return True, actual status depends on receiving frames
```

### 5.2 Configuration File Extensions

Add ROS-specific parameters to the configuration file:

```python
# --- ROS Configuration ---
ROS_ENABLED = True
ROS_CAMERA_TOPIC = '/camera/color/image_raw'
ROS_CAMERA_INFO_TOPIC = '/camera/color/camera_info'
ROS_DEPTH_TOPIC = '/camera/depth/image_rect_raw'
ROS_TARGET_FRAME = 'map'
ROS_CAMERA_FRAME = 'camera_link'
```

### 5.3 Visualization Enhancements

Add ROS message type conversion for visualization:

```python
def create_visualization_message(image, tracking_data, header):
    """Create a visualization message with tracking data overlaid on image"""
    # Draw all visualization elements
    display_image = image.copy()
    vis.draw_coordinate_system(display_image, origin_x, origin_y)
    # ... (other visualization functions)
    
    # Convert to ROS message
    try:
        bridge = CvBridge()
        img_msg = bridge.cv2_to_imgmsg(display_image, "bgr8")
        img_msg.header = header
        return img_msg
    except Exception as e:
        rospy.logerr(f"Error converting visualization to ROS message: {e}")
        return None
```

## 6. Launch File Example

```xml
<launch>
  <!-- Load configuration parameters -->
  <rosparam command="load" file="$(find vision_node)/config/vision_params.yaml" />
  
  <!-- Launch the vision node -->
  <node name="high_level_vision_node" pkg="vision_node" type="vision_node.py" output="screen">
    <param name="camera_type" value="realsense" />
    <param name="publish_visualization" value="true" />
  </node>
  
  <!-- Optional: Launch RViz for visualization -->
  <node name="rviz" pkg="rviz" type="rviz" args="-d $(find vision_node)/rviz/vision.rviz" if="$(arg launch_rviz)" />
</launch>
```

## 7. Testing & Verification

### 7.1 Coordinate Output Validation

It's essential to verify that the coordinate outputs match the expected Thai convention:

```python
def test_coordinate_convention():
    """Test that coordinates follow the Thai convention"""
    # Set up test case
    # Create sample detection in right side of frame
    mock_detection = {
        'box': [400, 200, 500, 300],  # Right side of frame (if frame is 640x480)
        'conf': 0.9,
        'class_id': 0
    }
    
    # Create sample depth data (1.5 meters away)
    mock_z = 1.5
    
    # Process coordinates
    result = self.vision_node.calculate_coordinates(
        mock_detection, mock_z, 320, 240)  # origin at center
    
    # Verify Thai convention: 
    # - X positive for right side
    # - Y positive for bottom half
    # - Angle positive for right side
    assert result['x'] > 0  # Object on right side should have positive X
    assert result['y'] > 0  # Object in bottom half should have positive Y
    assert result['z'] == mock_z  # Z should match input depth
    assert result['angle'] > 0  # Angle should be positive for right side
    
    # Verify another position (left side, upper half)
    mock_detection2 = {'box': [100, 100, 200, 200]}  # Left upper part
    result2 = self.vision_node.calculate_coordinates(
        mock_detection2, mock_z, 320, 240)
    
    assert result2['x'] < 0  # Object on left side should have negative X
    assert result2['y'] < 0  # Object in upper half should have negative Y
    assert result2['angle'] < 0  # Angle should be negative for left side
```

### 7.2 ROS Message Verification

Test that the ROS messages contain the correct data:

```python
def test_ros_message_generation():
    """Test that ROS messages are correctly generated"""
    # Create sample tracking data
    tracking_data = {
        'detected': True,
        'rel_x': 100.0,  # 100 pixels to the right
        'rel_y': -50.0,  # 50 pixels up
        'z': 2.0,        # 2 meters away
        'angle': 30.0,   # 30 degrees to the right
        'conf': 0.85
    }
    
    # Generate messages
    coords_msg = self.vision_node.create_coordinates_message(tracking_data)
    pose_msg = self.vision_node.create_pose_message(tracking_data)
    
    # Verify coordinates message (Thai convention)
    assert len(coords_msg.data) == 4  # x, y, z, angle
    assert coords_msg.data[0] == 100.0  # x
    assert coords_msg.data[1] == -50.0  # y
    assert coords_msg.data[2] == 2.0    # z
    assert coords_msg.data[3] == 30.0   # angle
    
    # Verify pose message (ROS convention)
    assert pose_msg.pose.position.x == 2.0          # z becomes x
    assert abs(pose_msg.pose.position.y - (-1.0)) < 0.001  # -x/100 becomes y
    assert abs(pose_msg.pose.position.z - 0.5) < 0.001     # -y/100 becomes z
    
    # Verify orientation represents the angle
    # Convert quaternion back to Euler angles
    quat = [pose_msg.pose.orientation.x,
            pose_msg.pose.orientation.y,
            pose_msg.pose.orientation.z,
            pose_msg.pose.orientation.w]
    euler = tf_transformations.euler_from_quaternion(quat)
    yaw_deg = math.degrees(euler[2])
    assert abs(yaw_deg - 30.0) < 0.001  # Should be approximately 30 degrees
```

### 7.3 Full Integration Testing

Test the complete pipeline with real RealSense data:

```python
def test_realsense_pipeline():
    """Integration test with actual RealSense data"""
    # Initialize ROS node in test mode
    rospy.init_node('test_vision_node', anonymous=True)
    
    # Create the vision node
    node = VisionNode()
    
    # Wait for RealSense to initialize
    time.sleep(2)
    
    # Create a subscriber to receive coordinate messages
    results = []
    def callback(msg):
        results.append(msg.data)
    
    sub = rospy.Subscriber('/vision/coordinates', Float32MultiArray, callback)
    
    # Wait for messages (5 seconds)
    timeout = time.time() + 5.0
    while time.time() < timeout and not results:
        time.sleep(0.1)
    
    # Verify that we received at least one valid message
    assert len(results) > 0
    
    # Verify message format
    assert len(results[0]) == 4  # x, y, z, angle
    
    # Clean up
    sub.unregister()
    node.shutdown()
```

## 8. Deployment Considerations

### 8.1 Dependencies

Ensure all dependencies are properly specified in the package.xml file:

```xml
<depend>roscpp</depend>
<depend>rospy</depend>
<depend>std_msgs</depend>
<depend>sensor_msgs</depend>
<depend>geometry_msgs</depend>
<depend>vision_msgs</depend>
<depend>cv_bridge</depend>
<depend>image_transport</depend>
<depend>tf2</depend>
<depend>tf2_ros</depend>
```

### 8.2 System Requirements

- CUDA-capable GPU (recommended for optimal performance)
- Intel RealSense SDK (if using RealSense cameras)
- Python 3.9 (เท่านั้น)
- ROS Noetic (or compatible version)
- Sufficient RAM (8GB minimum, 16GB recommended)

### 8.3 Performance Optimization

- Consider using image_transport for compressed image publishing
- Implement dynamic reconfigure for runtime parameter adjustment
- Add multi-threading for separate processing paths
- Use nodelets for zero-copy communication where appropriate

## 9. Coordinate Visualization 

To help visualize the coordinate system and tracking outputs, the node includes visualization capabilities:

```python
def publish_coordinate_visualization(self, tracking_data):
    """Publish visualization markers showing the coordinate system and detected target"""
    if not self.publish_visualization:
        return
        
    # 1. Create coordinate frame marker
    frame_marker = Marker()
    frame_marker.header.frame_id = self.camera_frame_id
    frame_marker.header.stamp = rospy.Time.now()
    frame_marker.ns = "vision_coordinates"
    frame_marker.id = 0
    frame_marker.type = Marker.ARROW
    frame_marker.action = Marker.ADD
    
    # X-axis (right) in RED
    frame_marker.points = []
    origin = Point(0, 0, 0)
    x_axis = Point(0.5, 0, 0)  # 0.5m in X direction
    frame_marker.points.append(origin)
    frame_marker.points.append(x_axis)
    frame_marker.scale.x = 0.01  # shaft diameter
    frame_marker.scale.y = 0.02  # head diameter
    frame_marker.color.r = 1.0
    frame_marker.color.g = 0.0
    frame_marker.color.b = 0.0
    frame_marker.color.a = 1.0
    self.marker_pub.publish(frame_marker)
    
    # Y-axis (down) in GREEN
    frame_marker.id = 1
    frame_marker.points = []
    y_axis = Point(0, 0.5, 0)  # 0.5m in Y direction
    frame_marker.points.append(origin)
    frame_marker.points.append(y_axis)
    frame_marker.color.r = 0.0
    frame_marker.color.g = 1.0
    self.marker_pub.publish(frame_marker)
    
    # Z-axis (forward) in BLUE
    frame_marker.id = 2
    frame_marker.points = []
    z_axis = Point(0, 0, 0.5)  # 0.5m in Z direction
    frame_marker.points.append(origin)
    frame_marker.points.append(z_axis)
    frame_marker.color.g = 0.0
    frame_marker.color.b = 1.0
    self.marker_pub.publish(frame_marker)
    
    # 2. If target detected, show target and angle
    if tracking_data['detected']:
        # Convert pixel coordinates to meters
        scale = self.pixel_to_meter_factor
        x = tracking_data['rel_x'] * scale
        y = tracking_data['rel_y'] * scale
        z = tracking_data['z']
        angle = tracking_data['angle']
        
        # Target position marker (YELLOW sphere)
        target_marker = Marker()
        target_marker.header.frame_id = self.camera_frame_id
        target_marker.header.stamp = rospy.Time.now()
        target_marker.ns = "vision_target"
        target_marker.id = 0
        target_marker.type = Marker.SPHERE
        target_marker.action = Marker.ADD
        target_marker.pose.position.x = z      # In ROS, X is forward (Z in camera)
        target_marker.pose.position.y = -x     # In ROS, Y is left (-X in camera)
        target_marker.pose.position.z = -y     # In ROS, Z is up (-Y in camera)
        target_marker.scale.x = 0.1
        target_marker.scale.y = 0.1
        target_marker.scale.z = 0.1
        target_marker.color.r = 1.0
        target_marker.color.g = 1.0
        target_marker.color.b = 0.0
        target_marker.color.a = 0.8
        self.marker_pub.publish(target_marker)
        
        # Angle visualization (line showing direction)
        angle_marker = Marker()
        angle_marker.header.frame_id = self.camera_frame_id
        angle_marker.header.stamp = rospy.Time.now()
        angle_marker.ns = "vision_angle"
        angle_marker.id = 0
        angle_marker.type = Marker.ARROW
        angle_marker.action = Marker.ADD
        
        # Start at target position
        angle_marker.points = []
        start = Point(z, -x, -y)
        
        # Calculate end point based on angle
        angle_rad = math.radians(angle)
        end_x = z + 0.5 * math.cos(angle_rad)  # 0.5m in direction of angle
        end_y = -x + 0.5 * math.sin(angle_rad)
        end = Point(end_x, end_y, -y)
        
        angle_marker.points.append(start)
        angle_marker.points.append(end)
        angle_marker.scale.x = 0.01  # shaft diameter
        angle_marker.scale.y = 0.02  # head diameter
        angle_marker.color.r = 0.0
        angle_marker.color.g = 1.0
        angle_marker.color.b = 1.0
        angle_marker.color.a = 1.0
        self.marker_pub.publish(angle_marker)
```
