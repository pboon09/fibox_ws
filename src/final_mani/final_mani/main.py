# ไฟล์: main.py
# คำอธิบาย: ไฟล์หลักสำหรับรันโปรแกรม ควบคุมการทำงานทั้งหมด

import cv2
import time
import math # ยังคงต้องใช้ ถ้ามีการคำนวณใน main (ตอนนี้ไม่น่ามี)

# Import โมดูลที่สร้างขึ้น
import config
import os
import utils # อาจจะไม่ต้องใช้โดยตรง แต่ module อื่นเรียกใช้
from camera_handler import CameraHandler, REALSENSE_AVAILABLE # Import REALSENSE_AVAILABLE มาด้วย
from object_detector import ObjectDetector
from depth_estimator import DepthEstimator
from tracker import Tracker
import visualization as vis # ตั้งชื่อย่อให้เรียกใช้ง่าย
import numpy as np
def main_app():
    """ ฟังก์ชันหลักในการรัน Application """

    # --- เลือกแหล่งภาพ ---
    print("\n===== Camera Selection =====")
    print("A: Use RealSense camera" if REALSENSE_AVAILABLE else "A: RealSense not available")
    print("B: Use notebook webcam")
    print("C: Use video file")
    print("=========================")
    source_type = 'webcam' # ค่าเริ่มต้น
    while True:
        choice = input("Enter your choice (A/B/C): ").strip().upper()
        if choice == 'A' and REALSENSE_AVAILABLE:
            source_type = 'realsense'; print("Selected: RealSense camera"); break
        elif choice == 'B':
            source_type = 'webcam'; print("Selected: Notebook webcam"); break
        elif choice == 'C':
            source_type = 'video'; print("Selected: Video file"); break
        else:
            if choice == 'A' and not REALSENSE_AVAILABLE: print("RealSense is not available. Select B or C.")
            else: print("Invalid choice. Please enter A, B, or C.")

    # --- นำเข้าโมดูลต่างๆ ---
    print("-" * 30)
    camera = CameraHandler(source_type=source_type)
    if not camera.initialized:
        print("[Fatal] Camera could not be initialized. Exiting.")
        return # ออกจากโปรแกรมถ้าเปิดกล้องไม่ได้

    detector = ObjectDetector()
    if not detector.initialized:
        print("[Fatal] Object Detector could not be initialized. Exiting.")
        camera.release()
        return

    depth_estimator = DepthEstimator()
    # ไม่ต้องเช็คก็ได้ เพราะมี Fallback แต่ถ้าต้องการความแม่นยำจาก Model อาจจะต้องเช็ค
    # if not depth_estimator.model_loaded:
    #     print("[Warning] Depth regression model not loaded.")

    tracker = Tracker()
    class_names = detector.get_class_names() # เอาชื่อคลาสมาใช้ในการแสดงผล

    # --- เตรียมตัวแปรสำหรับ Loop หลัก ---
    frame_count = 0
    frame_times = []
    width, height = camera.get_dimensions()
    origin_x = width // 2
    origin_y = height // 2

    print("-" * 30)
    print("\nStarting camera tracking with XYZ coordinates...")
    print("Press 'q' to quit")
    print("-" * 30)

    try:
        while True:
            start_time = time.time()

            # 1. อ่านเฟรมจากกล้อง
            ret, color_frame, depth_frame = camera.get_frame()
            # color_frame =  cv2.flip(color_frame, 0) # Flip ภาพในแนวแกน Y (แนวนอน) 
            if not ret:
                print("Failed to get frame or end of video. Exiting.")
                break # ออกจากลูปถ้าอ่านเฟรมไม่ได้
            if color_frame is None:
                continue # ข้ามเฟรมนี้ถ้าได้ภาพเป็น None

            frame_count += 1
            display_image = color_frame.copy() # สร้างสำเนาสำหรับวาดภาพ

            # 2. ตรวจจับวัตถุ
            detections = detector.detect(color_frame)

            # 3. เลือกเป้าหมาย (เลือกตัวแรกที่เจอ)
            target_detection_info = detections[0] if detections else None

            # 4. ประมาณค่า Depth
            # ส่ง depth_frame และ depth_scale ถ้าเป็น RealSense
            depth_scale = camera.get_depth_scale() if source_type == 'realsense' else 1.0
            current_z, raw_depth = depth_estimator.get_depth(
                target_detection_info,
                depth_frame if source_type == 'realsense' else None,
                depth_scale
            )
            depth_result = (current_z, raw_depth) # รวมผลลัพธ์ depth

            # 5. คำนวณค่า Tracking
            tracking_data = tracker.process_detection(
                target_detection_info,
                depth_result,
                origin_x,
                origin_y
            )

            # 6. วาดผลลัพธ์ลงบนภาพ
            # 6.1 วาดแกน Origin
            vis.draw_coordinate_system(display_image, origin_x, origin_y)

            # 6.2 วาดผลการตรวจจับ (ถ้ามี)
            if tracking_data['detected']:
                # เตรียมข้อมูลสำหรับฟังก์ชันวาด detection
                detection_vis_data = {
                    'box': tracking_data['box'],
                    'conf': tracking_data['conf'],
                    'class_id': tracking_data['class_id'],
                    'center_x': tracking_data['center_x'],
                    'center_y': tracking_data['center_y']
                }
                vis.draw_detection_results(display_image, detection_vis_data, class_names)
                vis.draw_origin_to_target_line(display_image, origin_x, origin_y,
                                               tracking_data['center_x'], tracking_data['center_y'])

            # 6.3 วาด Info Panel
            vis.draw_info_panel(display_image, tracking_data)

            # 6.4 คำนวณและวาด FPS / Mode
            end_time = time.time(); frame_time = end_time - start_time
            frame_times.append(frame_time)
            if len(frame_times) > 30: frame_times.pop(0)
            avg_frame_time = sum(frame_times) / len(frame_times) if frame_times else 0
            fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
            mode_text = source_type.capitalize() # ใช้ source_type ที่เลือกมาแสดง
            vis.draw_ui_elements(display_image, mode_text, fps)

            # 6.5 วาดคำแนะนำ
            vis.draw_instructions(display_image)

            # 7. แสดงผลภาพ
            window_title = f"XYZ Tracking - Target: {config.TARGET_CLASS_NAME}"
            cv2.imshow(window_title, display_image)

            # 8. จัดการ Input ผู้ใช้
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nExiting program...")
                break

    except KeyboardInterrupt:
        print("\nProgram interrupted by user (Ctrl+C).")
    finally:
        # --- Cleanup ---
        print("\n--- Starting Cleanup ---")
        camera.release() # ปิดกล้องผ่าน CameraHandler
        print("Closing OpenCV windows...")
        cv2.destroyAllWindows()
        for i in range(5): cv2.waitKey(1) # Ensure windows close
        print("--- Cleanup finished. ---")

if __name__ == "__main__":
    main_app() # เรียกฟังก์ชันหลัก
