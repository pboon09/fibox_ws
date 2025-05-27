# ไฟล์: object_detector.py
# คำอธิบาย: คลาสสำหรับโหลดโมเดล YOLO และตรวจจับวัตถุเป้าหมาย

import torch
from ultralytics import YOLO
import config # Import การตั้งค่า เช่น Model Path, Threshold, Target Class
import cv2 # อาจจะต้องใช้สำหรับคำนวณ center หรืออื่นๆ ในอนาคต
import numpy as np
import os
class ObjectDetector:
    """
    คลาสสำหรับจัดการโมเดล YOLO และการตรวจจับวัตถุ
    """
    def __init__(self):
        """ khởi tạo คลาส ObjectDetector """
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.target_class_id = -1
        self.class_names = {}
        self.initialized = False

        print(f"Object Detector will use device: {self.device}")
        self._load_model()

    def _load_model(self):
        """ โหลดโมเดล YOLO และหา ID ของคลาสเป้าหมาย """
        model_path = self.config.YOLO_MODEL_PATH
        target_name = self.config.TARGET_CLASS_NAME.lower() # ทำให้เป็นตัวพิมพ์เล็กเพื่อง่ายต่อการเปรียบเทียบ

        print(f"Loading YOLO model from: {model_path}")
        try:
            self.model = YOLO(model_path)
            print("YOLO model loaded successfully.")

            # ตรวจสอบและเก็บชื่อคลาสทั้งหมด
            if hasattr(self.model, 'names'):
                self.class_names = self.model.names
                print(f"Model class names: {self.class_names}")

                # ค้นหา ID ของคลาสเป้าหมาย
                found = False
                for i, name in self.class_names.items():
                    if name.lower() == target_name:
                        self.target_class_id = i
                        print(f"Target class '{config.TARGET_CLASS_NAME}' found with ID: {self.target_class_id}")
                        found = True
                        break
                if not found:
                    print(f"[Warning] Target class '{config.TARGET_CLASS_NAME}' not found in model names. Detector will detect ALL classes.")
                    # ตั้ง target_class_id เป็น -1 เพื่อให้ตรวจจับทุกคลาส
                    self.target_class_id = -1
            else:
                print("[Warning] Could not access model class names. Detector will detect ALL classes.")
                self.target_class_id = -1 # ตั้งเป็น -1 ถ้าไม่มีข้อมูลชื่อคลาส

            self.initialized = True

        except Exception as e:
            print(f"[Error] Failed to load YOLO model from {model_path}: {e}")
            self.initialized = False
            # อาจจะ raise Exception ต่อเพื่อให้โปรแกรมหลักหยุดทำงาน
            # raise RuntimeError(f"Failed to load YOLO model: {e}")

    def detect(self, frame):
        """
        ทำการตรวจจับวัตถุในเฟรมภาพที่กำหนด

        Args:
            frame (numpy.ndarray): เฟรมภาพสี BGR

        Returns:
            list: รายการของ dictionary ที่บรรจุข้อมูลวัตถุที่ตรวจพบและเป็นเป้าหมาย
                  แต่ละ dictionary มี key: 'box' (list [x1,y1,x2,y2]), 'conf' (float),
                  'class_id' (int), 'center_x' (int), 'center_y' (int).
                  คืนค่า list ว่าง หากไม่พบวัตถุเป้าหมาย หรือโมเดลยังไม่พร้อมใช้งาน
        """
        if not self.initialized or self.model is None:
            # print("[Warning] Object detector not initialized or model not loaded.")
            return [] # คืน list ว่างถ้าโมเดลยังไม่พร้อม

        detected_targets = []
        try:
            # ทำการ Inference
            results = self.model(frame, conf=self.config.CONF_THRESHOLD, device=self.device, verbose=False)

            # ประมวลผลผลลัพธ์
            for result in results:
                if result.boxes is None:
                    continue # ข้ามไปถ้าไม่มี boxes ใน result นี้

                boxes = result.boxes
                for box in boxes:
                    # ดึงข้อมูลพื้นฐาน
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])

                    # กรองเฉพาะคลาสเป้าหมาย (ถ้า target_class_id ไม่ใช่ -1)
                    if self.target_class_id != -1 and cls_id != self.target_class_id:
                        continue # ข้ามคลาสที่ไม่ใช่เป้าหมาย

                    # คำนวณจุดศูนย์กลาง
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2 # อาจจะปรับเป็น (y1+y2)//2 ถ้าต้องการจุดกลางจริงๆ

                    # เก็บข้อมูลวัตถุเป้าหมายที่เจอ
                    detection_info = {
                        'box': [x1, y1, x2, y2],
                        'conf': conf,
                        'class_id': cls_id,
                        'center_x': center_x,
                        'center_y': center_y
                    }
                    detected_targets.append(detection_info)

                    # *** หมายเหตุ: โค้ดปัจจุบันจะประมวลผลเฉพาะวัตถุแรกที่เจอใน main.py ***
                    # *** หากต้องการรองรับหลายวัตถุ ต้องปรับแก้ logic ใน main.py ***
                    # *** แต่ฟังก์ชัน detect นี้สามารถคืนค่าหลายวัตถุได้ ***
                    # break # ถ้าต้องการแค่วัตถุแรกที่เจอ ให้ uncomment บรรทัดนี้

        except Exception as e:
            print(f"[Error] Exception during object detection: {e}")
            return [] # คืน list ว่างหากเกิดข้อผิดพลาด

        return detected_targets

    def get_class_names(self):
        """ คืนค่า dictionary ของชื่อคลาสทั้งหมดที่โมเดลรู้จัก """
        return self.class_names