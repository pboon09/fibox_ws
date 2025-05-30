# -*- coding: utf-8 -*-
import numpy as np
import cv2
import torch
from ultralytics import YOLO
import time
import datetime
# import csv # --- ลบ CSV ---
import os
import math
# from pycaret.classification import *
from pycaret.regression import *
import pandas as pd

# Try to import RealSense library, but continue if not available
try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
    print("RealSense SDK found")
except ImportError:
    REALSENSE_AVAILABLE = False
    print("RealSense SDK not found")

# --- Configuration ---
MODEL_PATH = r"yolo11l.pt"  # <<< ตรวจสอบ Path โมเดล YOLO
REGRESSION_MODEL_PATH = './model/final_calibrated_depth_model_outdoor_v2' # <<< ตรวจสอบ Path โมเดล Regression

# YOLO Model Configuration
CONF_THRESHOLD = 0.15
TARGET_CLASS_NAME = 'person' # <<< ตั้งค่าเป้าหมาย

# Camera Configuration
CAMERA_ID = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30

# Reference object size in meters (used for Z estimation fallback)
REFERENCE_OBJECT_WIDTH = 0.5
REFERENCE_OBJECT_HEIGHT = 0.5

# Focal length estimation (can be calibrated, used for fallback Z estimation)
FOCAL_LENGTH = 500

def main():
    # --- Ask user for camera choice ---
    print("\n===== Camera Selection =====")
    print("A: Use RealSense camera" if REALSENSE_AVAILABLE else "A: RealSense not available")
    print("B: Use notebook webcam")
    print("C: Use video file")
    print("=========================")

    USE_REALSENSE = False
    USING_VIDEO = False
    video_path = None
    cap = None
    pipeline = None
    align = None
    depth_scale = 1.0

    while True:
        choice = input("Enter your choice (A/B/C): ").strip().upper()
        if choice == 'A' and REALSENSE_AVAILABLE:
            USE_REALSENSE = True; USING_VIDEO = False; print("Selected: RealSense camera"); break
        elif choice == 'B':
            USE_REALSENSE = False; USING_VIDEO = False; print("Selected: Notebook webcam"); break
        elif choice == 'C':
            USE_REALSENSE = False; USING_VIDEO = True; print("Selected: Video file")
            video_path = "./src/video/Screen Recording 2025-01-24 085343.mp4" # Update path if needed
            if not os.path.exists(video_path):
                print(f"Error: Video file not found at {video_path}. Please check path."); continue
            break
        else:
            if choice == 'A' and not REALSENSE_AVAILABLE: print("RealSense is not available. Select B or C.")
            else: print("Invalid choice. Please enter A, B, or C.")

    # --- Initialization ---
    frame_count = 0
    robot_rotation_angle_xz = 0.0

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    print(f"Loading YOLO model from {MODEL_PATH}...")
    try:
        model = YOLO(MODEL_PATH)
        target_class_id = -1
        if hasattr(model, 'names'):
            class_names = model.names
            print(f"Model class names: {class_names}")
            found = False
            for i, name in class_names.items():
                if name.lower() == TARGET_CLASS_NAME.lower():
                    target_class_id = i; print(f"Target class '{TARGET_CLASS_NAME}' found with ID: {target_class_id}"); found = True; break
            if not found: print(f"Warning: Target class '{TARGET_CLASS_NAME}' not found in model names."); print("Model will detect all classes.")
        else: print("Warning: Could not access model class names.")
    except Exception as e: print(f"Error loading YOLO model: {e}"); exit()

    print(f"Loading Regression model from {REGRESSION_MODEL_PATH}...")
    try:
        loaded_model = load_model(REGRESSION_MODEL_PATH)
        print("Regression model loaded successfully.")
    except Exception as e:
        print(f"Error loading Regression model: {e}"); loaded_model = None
        print("Warning: Proceeding without the regression model for Z prediction.")

    # Initialize camera
    if USE_REALSENSE:
        print("Initializing RealSense camera...")
        try:
            pipeline = rs.pipeline(); config = rs.config()
            pipeline_wrapper = rs.pipeline_wrapper(pipeline); pipeline_profile = config.resolve(pipeline_wrapper)
            device_cam = pipeline_profile.get_device()
            found_rgb = any(s.get_info(rs.camera_info.name) == 'RGB Camera' for s in device_cam.sensors)
            found_depth = any(s.get_info(rs.camera_info.name) == 'Stereo Module' for s in device_cam.sensors)
            if not found_rgb or not found_depth: print("Error: Required RGB or Depth camera not found."); exit()
            config.enable_stream(rs.stream.depth, FRAME_WIDTH, FRAME_HEIGHT, rs.format.z16, FPS)
            config.enable_stream(rs.stream.color, FRAME_WIDTH, FRAME_HEIGHT, rs.format.bgr8, FPS)
            print("Starting RealSense pipeline...")
            profile = pipeline.start(config); depth_sensor = profile.get_device().first_depth_sensor()
            depth_scale = depth_sensor.get_depth_scale(); print(f"Depth Scale is: {depth_scale}")
            align_to = rs.stream.color; align = rs.align(align_to); realsense_initialized = True
        except Exception as e:
            print(f"Error initializing RealSense: {e}"); print("Falling back to regular webcam...")
            realsense_initialized = False; USE_REALSENSE = False
            cap = cv2.VideoCapture(CAMERA_ID)
            if not cap.isOpened(): print(f"Error: Could not open fallback webcam with ID {CAMERA_ID}"); exit()
    elif USING_VIDEO:
        print(f"Opening video file: {video_path}..."); cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): print(f"Error: Could not open video file {video_path}"); exit()
    else: # Use Webcam
        print(f"Opening webcam (ID: {CAMERA_ID})..."); cap = cv2.VideoCapture(CAMERA_ID)
        if not cap.isOpened():
            print(f"Error: Could not open webcam with ID {CAMERA_ID}"); cap = cv2.VideoCapture(0)
            if not cap.isOpened(): print("Error: Could not open any camera"); exit()

    if cap:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT); cap.set(cv2.CAP_PROP_FPS, FPS)
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS); print(f"Actual camera properties: {actual_width}x{actual_height}, {actual_fps} FPS")


    def estimate_z_from_bbox(bbox_width, bbox_height):
        """Estimate Z distance based on object size in image. Needs calibration."""
        if bbox_width <= 0 or bbox_height <= 0: return 0.0
        z_from_width = (REFERENCE_OBJECT_WIDTH * FOCAL_LENGTH) / bbox_width
        z_from_height = (REFERENCE_OBJECT_HEIGHT * FOCAL_LENGTH) / bbox_height
        return (z_from_width + z_from_height) / 2.0

    def calculate_3d_distance(x, y, z):
        """Calculate 3D Euclidean distance from origin (0,0,0)"""
        try:
            return math.sqrt(float(x)**2 + float(y)**2 + float(z)**2)
        except ValueError:
            return 0.0

    def draw_coordinate_system(image, orig_x, orig_y, size=60, thickness=3):
        """Draw coordinate system at origin point - made clearer"""
        cv2.arrowedLine(image, (orig_x, orig_y), (orig_x + size, orig_y), (0, 0, 255), thickness) # X red right
        cv2.arrowedLine(image, (orig_x, orig_y), (orig_x, orig_y - size), (0, 255, 0), thickness) # Y green up
        cv2.circle(image, (orig_x, orig_y), thickness + 2, (255, 0, 0), -1) # Z blue dot
        cv2.circle(image, (orig_x, orig_y), thickness + 7, (255, 0, 0), 1) # Z blue circle
        font_scale = 0.6
        cv2.putText(image, "X", (orig_x + size + 5, orig_y + 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 2)
        cv2.putText(image, "Y", (orig_x + 10, orig_y - size - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)
        cv2.putText(image, "Z", (orig_x - 25, orig_y + 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), 2)
        cv2.putText(image, "Origin", (orig_x - 30, orig_y + 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale*0.9, (255,255,255), 1)

    def get_realsense_depth(depth_frame, x, y, width, height, scale):
        """Get average depth from RealSense depth frame for a region, converting to meters"""
        if depth_frame is None or scale <= 0: return 0.0
        x1_roi=max(0,int(x)); y1_roi=max(0,int(y))
        x2_roi=min(depth_frame.get_width(),int(x+width)); y2_roi=min(depth_frame.get_height(),int(y+height))
        if x1_roi>=x2_roi or y1_roi>=y2_roi: return 0.0
        try:
            depth_image=np.asanyarray(depth_frame.get_data()); depth_roi=depth_image[y1_roi:y2_roi,x1_roi:x2_roi]
            valid_depths=depth_roi[depth_roi>0]
            if valid_depths.size>0: return np.mean(valid_depths)*scale
            else: return 0.0
        except Exception as e: print(f"Error processing depth frame: {e}"); return 0.0

    # --- Main Loop ---
    print("\nStarting camera tracking with XYZ coordinates...")
    print("Press 'q' to quit")
    if USE_REALSENSE: print("Using RealSense for depth data")
    elif USING_VIDEO: print("Using video file for input")
    else: print("Using webcam (depth prediction relies on model/estimation)")

    frame_times = []

    try:
        while True:
            start_time = time.time()
            frame = None; depth_frame = None; color_image = None

            # Get frame
            if USE_REALSENSE and pipeline:
                try:
                    frames = pipeline.wait_for_frames(timeout_ms=5000);
                    if not frames: continue
                    aligned_frames = align.process(frames); depth_frame = aligned_frames.get_depth_frame(); color_frame = aligned_frames.get_color_frame()
                    if not depth_frame or not color_frame: continue
                    frame = np.asanyarray(color_frame.get_data())
                except RuntimeError as e: print(f"RealSense runtime error: {e}. Trying to continue..."); time.sleep(0.1); continue
            elif cap:
                ret, frame = cap.read()
                if not ret:
                    if USING_VIDEO: print("End of video file reached."); break
                    else: print("Error reading frame from webcam."); break
            else: print("Error: No camera source available."); break
            if frame is None: continue

            frame_count += 1
            display_image = frame.copy()
            height, width = frame.shape[:2]
            origin_x = width // 2; origin_y = height // 2

            # --- ค่าเริ่มต้นสำหรับเฟรมนี้ ---
            display_rel_x = 0.0
            display_rel_y = 0.0
            current_z = 0.0
            robot_rotation_angle_xz = 0.0
            distance_3d_origin = 0.0
            detected_target_this_frame = False

            # Object detection
            try: results = model(frame, conf=CONF_THRESHOLD, device=device, verbose=False)
            except Exception as e: print(f"Error during YOLO inference: {e}"); results = []

            # Process detections
            for result in results:
                if result.boxes is None: continue
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0]); cls_id = int(box.cls[0])
                    if target_class_id != -1 and cls_id != target_class_id: continue
                    detected_target_this_frame = True

                    center_box_x = (x1 + x2) // 2; center_box_y = (y1 + y2) // 2
                    raw_rel_x = center_box_x - origin_x; raw_rel_y = center_box_y - origin_y
                    display_rel_x = float(raw_rel_x)
                    display_rel_y = float(raw_rel_y)

                    # Depth Calculation (Z Coordinate)
                    bbox_width = x2 - x1; bbox_height = y2 - y1
                    z_est = estimate_z_from_bbox(bbox_width, bbox_height)
                    realsense_depth_raw = 0.0; current_z_calc = 0.0
                    if USE_REALSENSE and depth_frame:
                        realsense_depth_raw = get_realsense_depth(depth_frame, x1, y1, bbox_width, bbox_height, depth_scale)
                    if loaded_model:
                        input_depth_feature = realsense_depth_raw if realsense_depth_raw > 0 else z_est
                        predict_data = pd.DataFrame({ 'X_min': [x1], 'Y_min': [y1], 'X_max': [x2], 'Y_max': [y2],
                                                      'Confidence': [conf], 'Average_Depth_m': [input_depth_feature],
                                                      'area': [bbox_width * bbox_height] })
                        try:
                            prediction_result = predict_model(loaded_model, data=predict_data)
                            predicted_z_value = prediction_result['prediction_label'].iloc[0] # <<< ตรวจสอบชื่อ Label
                            current_z_calc = float(predicted_z_value)
                            if current_z_calc <= 0: current_z_calc = input_depth_feature if input_depth_feature > 0 else z_est
                        except Exception as e:
                            print(f"Error during regression prediction: {e}")
                            current_z_calc = realsense_depth_raw if realsense_depth_raw > 0 else z_est
                    else:
                        current_z_calc = realsense_depth_raw if realsense_depth_raw > 0 else z_est
                    current_z = current_z_calc

                    # Calculate Robot Rotation Angle (X-Z Plane)
                    robot_rotation_angle_xz_calc = 0.0
                    if current_z > 0:
                        angle_rad_xz = math.atan2(display_rel_x, current_z)
                        robot_rotation_angle_xz_calc = math.degrees(angle_rad_xz)
                    robot_rotation_angle_xz = robot_rotation_angle_xz_calc

                    # Calculate 3D distance (always from origin now)
                    distance_3d_origin = calculate_3d_distance(display_rel_x, display_rel_y, current_z)

                    # --- Drawing Visuals ---
                    draw_coordinate_system(display_image, origin_x, origin_y)
                    cv2.line(display_image, (origin_x, origin_y), (center_box_x, center_box_y), (0, 255, 255), 2) # Yellow line thicker
                    color = (0, 255, 0); cv2.rectangle(display_image, (x1, y1), (x2, y2), color, 2) # Thicker box
                    label = f"{model.names[cls_id]} {conf:.2f}"
                    cv2.putText(display_image, label, (x1, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                    # --- ลบ: เส้นประแนวตั้งผ่าน Target ---
                    # vertical_line_color = (180, 180, 180)
                    # cv2.line(display_image, (center_box_x, 0), (center_box_x, height), vertical_line_color, 1)

                    # --- ทำให้จุดศูนย์กลางวัตถุชัดขึ้น + เพิ่ม Label "Target" ---
                    target_marker_color = (0, 255, 255) # สีเหลือง Cyan
                    target_marker_radius = 8
                    target_marker_thickness = 2
                    inner_dot_radius = 2
                    cv2.circle(display_image, (center_box_x, center_box_y), target_marker_radius, target_marker_color, target_marker_thickness)
                    cv2.circle(display_image, (center_box_x, center_box_y), inner_dot_radius, (0,0,0), -1)
                    cv2.putText(display_image, "Target", (center_box_x + target_marker_radius + 5, center_box_y + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, target_marker_color, 1)

                    # --- เพิ่ม: การวาด Arc แสดงมุมเทียบกับแกน Y ---
                    if detected_target_this_frame:
                        # คำนวณมุมของวัตถุในระนาบภาพ (0 ขวา, 90 ลง, 180 ซ้าย, 270 ขึ้น)
                        object_screen_angle_rad = math.atan2(-display_rel_y, display_rel_x) # ใช้ -y เพราะ y ของ cv2 ชี้ลง
                        object_screen_angle_deg = (math.degrees(object_screen_angle_rad) + 360) % 360

                        arc_radius = 35 # รัศมีของ Arc
                        arc_color = (255, 165, 0) # สีส้ม
                        arc_thickness = 2
                        y_axis_angle = 270 # มุมของแกน Y ชี้ขึ้น

                        # วาดเส้นอ้างอิงแกน Y ชี้ขึ้น
                        cv2.line(display_image, (origin_x, origin_y), (origin_x, origin_y - arc_radius), (0, 255, 0), 1)

                        # วาด Arc จากแกน Y (270) ไปยังมุมของวัตถุ
                        cv2.ellipse(display_image, (origin_x, origin_y), (arc_radius, arc_radius),
                                    0, y_axis_angle, object_screen_angle_deg, arc_color, arc_thickness)

                    break # Process only the first detected target

            # --- End of Detection Loop ---

            # Calculate and display FPS
            end_time = time.time(); frame_time = end_time - start_time
            frame_times.append(frame_time)
            if len(frame_times) > 30: frame_times.pop(0)
            avg_frame_time = sum(frame_times) / len(frame_times) if frame_times else 0
            fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0

            # Determine mode text
            if USE_REALSENSE: mode_text = "RealSense"
            elif USING_VIDEO: mode_text = "Video File"
            else: mode_text = "Webcam"

            # Display mode & FPS (มุมซ้ายบน)
            font_ui = cv2.FONT_HERSHEY_SIMPLEX
            font_scale_ui = 0.55
            thickness_ui = 2
            cv2.putText(display_image, f"Mode: {mode_text}", (10, 25), font_ui, font_scale_ui, (255,255,255), thickness_ui)
            cv2.putText(display_image, f"FPS: {fps:.1f}", (10, 50), font_ui, font_scale_ui, (0, 255, 0), thickness_ui)

            # --- แสดงข้อมูลหลักมุมซ้ายล่าง (Clean UI) ---
            info_text_y_start = height - 15
            line_height = 20
            info_bg_color = (0, 0, 0)
            text_color = (255, 255, 255)
            angle_color = (0, 255, 255)
            font_info = cv2.FONT_HERSHEY_SIMPLEX
            font_scale_info = 0.5
            thickness_info = 1

            info_lines_data = []
            if detected_target_this_frame:
                 info_lines_data.append( ("Pos:", f"X{display_rel_x:+.0f} Y{display_rel_y:+.0f} Z{current_z:.2f}m", text_color) )
                 info_lines_data.append( ("Dist:", f"{distance_3d_origin:.2f} m", text_color) )
                 info_lines_data.append( ("Angle:", f"{robot_rotation_angle_xz:.1f} deg", angle_color) )
            else:
                 info_lines_data.append( ("Status:", "Target not detected", (0, 165, 255)) )

            if info_lines_data:
                max_text_width = 0
                for label, value, color in info_lines_data:
                    text = label + " " + value
                    text_size = cv2.getTextSize(text, font_info, font_scale_info, thickness_info)[0]
                    if text_size[0] > max_text_width: max_text_width = text_size[0]
                panel_width = max_text_width + 20
                panel_height = len(info_lines_data) * line_height + 10
                panel_x = 5; panel_y = info_text_y_start - panel_height
                cv2.rectangle(display_image, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), info_bg_color, -1)
                y_pos = panel_y + int(line_height * 0.7)
                for label, value, color in info_lines_data:
                    cv2.putText(display_image, label, (panel_x + 5, y_pos), font_info, font_scale_info, text_color, thickness_info)
                    value_x = panel_x + 5 + cv2.getTextSize(label + " ", font_info, font_scale_info, thickness_info)[0][0]
                    cv2.putText(display_image, value, (value_x - 5, y_pos), font_info, font_scale_info, color, thickness_info)
                    y_pos += line_height

            # Add instructions at the bottom
            instruction_y = height - 5
            cv2.putText(display_image, "q: Quit",
                       (width - 50, instruction_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

            # Display the frame
            window_title = f"XYZ Tracking - Target: {TARGET_CLASS_NAME}"
            cv2.imshow(window_title, display_image)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("\nExiting program..."); break

    except KeyboardInterrupt: print("\nProgram interrupted by user (Ctrl+C).")
    finally:
        # Cleanup
        print("\n--- Starting Cleanup ---")
        if USE_REALSENSE and pipeline:
            try: print("Stopping RealSense pipeline..."); pipeline.stop()
            except Exception as e: print(f"Error stopping RealSense pipeline: {e}")
        elif cap: print("Releasing webcam or video file..."); cap.release()
        print("Closing OpenCV windows..."); cv2.destroyAllWindows()
        for i in range(5): cv2.waitKey(1)
        print("--- Cleanup finished. ---")

if __name__ == "__main__":
    main()