# ไฟล์: depth_estimator.py
# คำอธิบาย: คลาสสำหรับจัดการการประมาณค่าความลึก (Depth)

import numpy as np
import pandas as pd
import config  # Import การตั้งค่า
import utils   # Import ฟังก์ชันช่วยเหลือ เช่น estimate_z_from_bbox

# ตรวจสอบว่ามี RealSense SDK หรือไม่ (สำหรับ type hint หรือ logic อื่นๆ ถ้าจำเป็น)
try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False

# Import PyCaret regression model functions
try:
    from pycaret.regression import load_model, predict_model
    PYCARET_AVAILABLE = True
except ImportError:
    PYCARET_AVAILABLE = False
    print("[Warning] PyCaret regression library not found. Depth prediction will rely on RealSense or estimation.")

class DepthEstimator:
    """
    คลาสสำหรับประมาณค่าความลึก (Z) โดยใช้ Regression Model, RealSense,
    หรือการประมาณค่าจาก Bounding Box เป็นลำดับความสำคัญ
    """
    def __init__(self):
        """ khởi tạo คลาส DepthEstimator และโหลด Regression Model """
        self.config = config
        self.regression_model = None
        self.model_loaded = False

        if PYCARET_AVAILABLE:
            self._load_regression_model()
        else:
            print("[Info] Skipping regression model loading as PyCaret is not available.")

    def _load_regression_model(self):
        """ โหลด Regression Model จาก PyCaret """
        model_path = self.config.DEPTH_MODEL_PATH
        print(f"Loading Regression model from: {model_path}")
        try:
            self.regression_model = load_model(model_path)
            self.model_loaded = True
            print("Regression model loaded successfully.")
        except Exception as e:
            print(f"[Error] Failed to load Regression model from {model_path}: {e}")
            print("[Warning] Proceeding without the regression model for Z prediction.")
            self.model_loaded = False

    def _get_realsense_depth_roi(self, depth_frame, x1, y1, x2, y2, scale):
        """
        (Internal) ดึงค่า Depth เฉลี่ยจาก ROI ใน RealSense depth frame

        Args:
            depth_frame (pyrealsense2.depth_frame): เฟรมข้อมูลความลึก
            x1, y1, x2, y2 (int): พิกัด Bounding Box
            scale (float): Depth scale ของกล้อง RealSense

        Returns:
            float: ค่า Depth เฉลี่ยในหน่วยเมตร, หรือ 0.0 หากไม่มีข้อมูล
        """
        if depth_frame is None or scale <= 0:
            return 0.0

        # กำหนดขอบเขต ROI ให้อยู่ภายในขนาดของ depth frame
        frame_h, frame_w = depth_frame.get_height(), depth_frame.get_width()
        roi_x1 = max(0, int(x1))
        roi_y1 = max(0, int(y1))
        roi_x2 = min(frame_w, int(x2))
        roi_y2 = min(frame_h, int(y2))

        if roi_x1 >= roi_x2 or roi_y1 >= roi_y2:
            return 0.0 # ROI ไม่ถูกต้อง

        try:
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_roi = depth_image[roi_y1:roi_y2, roi_x1:roi_x2]
            valid_depths = depth_roi[depth_roi > 0] # กรองค่า 0 ออก

            if valid_depths.size > 0:
                average_depth_mm = np.mean(valid_depths)
                return average_depth_mm * scale # แปลงเป็นเมตร
            else:
                return 0.0
        except Exception as e:
            print(f"[Error] Processing RealSense depth ROI: {e}")
            return 0.0

    def get_depth(self, detection_info, depth_frame=None, depth_scale=1.0):
        """
        คำนวณค่าความลึก (Z) สำหรับวัตถุที่ตรวจพบ

        Args:
            detection_info (dict หรือ None): ข้อมูลวัตถุที่ตรวจพบจาก ObjectDetector
                                            (ต้องมี 'box', 'conf') หรือ None ถ้าไม่เจอ
            depth_frame (pyrealsense2.depth_frame หรือ None): เฟรมข้อมูลความลึกจาก RealSense
            depth_scale (float): Depth scale ของกล้อง RealSense

        Returns:
            tuple: (current_z, realsense_depth_raw)
                   current_z (float): ค่าความลึกสุดท้ายที่คำนวณได้ (เมตร)
                   realsense_depth_raw (float): ค่าความลึกดิบที่ได้จาก RealSense (เมตร), หรือ 0.0
        """
        if detection_info is None:
            return 0.0, 0.0 # ไม่เจอวัตถุ

        # --- ค่าเริ่มต้น ---
        current_z = 0.0
        realsense_depth_raw = 0.0
        z_est = 0.0

        # --- ดึงข้อมูลที่จำเป็น ---
        try:
            x1, y1, x2, y2 = detection_info['box']
            conf = detection_info['conf']
            bbox_width = x2 - x1
            bbox_height = y2 - y1
        except (KeyError, TypeError) as e:
            print(f"[Error] Invalid detection_info format: {e}")
            return 0.0, 0.0

        # 1. คำนวณค่าประมาณเบื้องต้น (Fallback)
        z_est = utils.estimate_z_from_bbox(bbox_width, bbox_height)

        # 2. ดึงค่าจาก RealSense (ถ้ามี)
        if REALSENSE_AVAILABLE and depth_frame:
            realsense_depth_raw = self._get_realsense_depth_roi(depth_frame, x1, y1, x2, y2, depth_scale)

        # 3. ใช้ Regression Model (ถ้ามีและโหลดสำเร็จ)
        if self.model_loaded and self.regression_model:
            # เลือก Input feature สำหรับโมเดล (ใช้ RealSense ถ้าได้ค่า, ไม่งั้นใช้ค่าประมาณ)
            input_depth_feature = realsense_depth_raw if realsense_depth_raw > 0 else z_est

            # สร้าง DataFrame สำหรับ predict
            predict_data = pd.DataFrame({
                'X_min': [x1], 'Y_min': [y1], 'X_max': [x2], 'Y_max': [y2],
                'Confidence': [conf],
                'Average_Depth_m': [input_depth_feature], # ชื่อ Feature ต้องตรงกับตอนเทรน
                'area': [bbox_width * bbox_height]
            })

            try:
                # ทำนายค่า Z
                prediction_result = predict_model(self.regression_model, data=predict_data)
                # ดึงค่าที่ทำนายได้ (ตรวจสอบชื่อคอลัมน์ 'prediction_label' หรือ 'Label' ให้ถูกต้อง)
                predicted_z_value = prediction_result['prediction_label'].iloc[0]
                current_z = float(predicted_z_value)

                # ตรวจสอบค่า Z ที่ได้ ถ้าไม่สมเหตุสมผล ให้ใช้ค่าอื่นแทน
                if current_z <= 0:
                    print(f"[Warning] Regression model predicted non-positive Z ({current_z:.3f}). Falling back.")
                    current_z = input_depth_feature if input_depth_feature > 0 else z_est # Fallback ไปที่ input หรือค่าประมาณ
            except Exception as e:
                print(f"[Error] During regression prediction: {e}")
                # ถ้า predict ไม่สำเร็จ ใช้ค่า RealSense หรือค่าประมาณแทน
                current_z = realsense_depth_raw if realsense_depth_raw > 0 else z_est
        else:
            # ถ้าไม่มี Regression Model ให้ใช้ค่า RealSense หรือค่าประมาณ
            current_z = realsense_depth_raw if realsense_depth_raw > 0 else z_est

        return current_z, realsense_depth_raw
