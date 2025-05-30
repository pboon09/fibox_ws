# ไฟล์: tracker.py
# คำอธิบาย: คลาสสำหรับคำนวณค่า Tracking หลัก (พิกัดสัมพัทธ์, มุม, ระยะทาง)

import math
import config # ไม่ได้ใช้โดยตรง แต่เผื่ออนาคต
import utils  # ใช้ calculate_3d_distance

class Tracker:
    """
    คลาสสำหรับประมวลผลข้อมูลการตรวจจับและข้อมูลความลึก เพื่อคำนวณค่า Tracking
    เช่น พิกัดสัมพัทธ์, มุมที่ต้องหมุน, ระยะทาง
    (ปัจจุบันยังไม่มี State หรือ Filter แต่สร้างเป็นคลาสเผื่อการขยายในอนาคต)
    """
    def __init__(self):
        """ khởi tạo คลาส Tracker """
        self.config = config
        # สามารถเพิ่มการ khởi tạo ค่า หรือ Filter ต่างๆ ที่นี่ได้ในอนาคต
        # เช่น Kalman Filter สำหรับ Smoothing ตำแหน่ง

    def process_detection(self, detection_info, depth_result, origin_x, origin_y):
        """
        ประมวลผลข้อมูลจาก detection และ depth เพื่อคำนวณค่า tracking

        Args:
            detection_info (dict หรือ None): ข้อมูลวัตถุที่ตรวจพบจาก ObjectDetector
            depth_result (tuple): ผลลัพธ์จาก DepthEstimator (current_z, realsense_depth_raw)
            origin_x (int): พิกัด X ของจุด Origin (ศูนย์กลางภาพ)
            origin_y (int): พิกัด Y ของจุด Origin (ศูนย์กลางภาพ)

        Returns:
            dict: ข้อมูล Tracking ที่ประมวลผลแล้ว ประกอบด้วย key:
                  'detected' (bool), 'rel_x' (float), 'rel_y' (float), 'z' (float),
                  'dist' (float), 'angle' (float), 'raw_depth' (float),
                  'center_x' (int), 'center_y' (int), 'box' (list), 'conf' (float), 'class_id' (int)
                  โดยค่าต่างๆ จะเป็น 0 หรือ None หาก 'detected' เป็น False
        """
        tracking_data = {
            'detected': False, 'rel_x': 0.0, 'rel_y': 0.0, 'z': 0.0,
            'dist': 0.0, 'angle': 0.0, 'raw_depth': 0.0,
            'center_x': None, 'center_y': None, 'box': None, 'conf': 0.0, 'class_id': -1
        }

        if detection_info is None:
            return tracking_data # ไม่เจอวัตถุ

        try:
            # ดึงข้อมูลจาก detection และ depth
            tracking_data['detected'] = True
            tracking_data['center_x'] = detection_info['center_x']
            tracking_data['center_y'] = detection_info['center_y']
            tracking_data['box'] = detection_info['box']
            tracking_data['conf'] = detection_info['conf']
            tracking_data['class_id'] = detection_info['class_id']

            current_z, realsense_depth_raw = depth_result
            tracking_data['z'] = current_z
            tracking_data['raw_depth'] = realsense_depth_raw

            # คำนวณพิกัดสัมพัทธ์กับ Origin (X ขวาเป็นบวก, Y ลงเป็นบวก ตาม pixel coord)
            rel_x = float(tracking_data['center_x'] - origin_x) * current_z / 716
            rel_y = float(tracking_data['center_y'] - origin_y) * current_z / 716
            tracking_data['rel_x'] = rel_x
            tracking_data['rel_y'] = rel_y

            # คำนวณมุม Robot Rotation Angle (X-Z Plane)
            # มุม 0 คือตรงหน้า, บวกคือขวา, ลบคือซ้าย
            robot_angle_xz = 0.0
            if current_z > 0:
                angle_rad_xz = math.atan2(rel_x, current_z) # atan2(x, z)
                robot_angle_xz = math.degrees(angle_rad_xz)
            tracking_data['angle'] = robot_angle_xz

            # คำนวณระยะทาง 3D จาก Origin
            # หมายเหตุ: x, y เป็น pixel, z เป็นเมตร -> ระยะทางมีหน่วยผสม
            # หากต้องการระยะทางเป็นเมตรล้วน ต้องแปลง x, y เป็นเมตรก่อน
            distance_origin = utils.calculate_3d_distance(rel_x, rel_y, current_z)
            tracking_data['dist'] = distance_origin

        except (KeyError, TypeError, ValueError) as e:
            print(f"[Error] Processing tracking data: {e}")
            tracking_data['detected'] = False # ถ้ามีปัญหา ให้ถือว่าตรวจไม่เจอ

        return tracking_data
