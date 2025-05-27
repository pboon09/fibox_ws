# ไฟล์: utils.py
# คำอธิบาย: รวมฟังก์ชันช่วยเหลือทั่วไปสำหรับโปรแกรม

import math
import config # Import ค่า Config ที่ต้องใช้

def estimate_z_from_bbox(bbox_width, bbox_height):
    """
    ประมาณค่าความลึก (แกน Z) จากขนาดของ Bounding Box บนภาพ
    เป็นวิธีสำรองที่ความแม่นยำขึ้นอยู่กับการตั้งค่าใน config.py
    และมุมมองของกล้อง ควร Calibrate ค่าใน config ก่อนใช้งานจริงจัง

    Args:
        bbox_width (int): ความกว้างของ Bounding Box (pixel)
        bbox_height (int): ความสูงของ Bounding Box (pixel)

    Returns:
        float: ค่า Z โดยประมาณ (เมตร), หรือ 0.0 หากคำนวณไม่ได้
    """
    if bbox_width <= 0 or bbox_height <= 0:
        # ป้องกันการหารด้วยศูนย์
        return 0.0
    try:
        # คำนวณ Z จากความกว้างและสูง แล้วหาค่าเฉลี่ย
        z_from_width = (config.REFERENCE_OBJECT_WIDTH * config.ESTIMATED_FOCAL_LENGTH) / bbox_width
        z_from_height = (config.REFERENCE_OBJECT_HEIGHT * config.ESTIMATED_FOCAL_LENGTH) / bbox_height
        z_est = (z_from_width + z_from_height) / 2.0
        return z_est
    except ZeroDivisionError:
        # ดักจับเผื่อกรณีอื่นที่อาจหารด้วยศูนย์ (ถึงแม้จะเช็คไปแล้ว)
        return 0.0
    except Exception as e:
        # จัดการ Error อื่นๆ ที่อาจเกิดขึ้น
        print(f"[Error] in estimate_z_from_bbox: {e}")
        return 0.0

def calculate_3d_distance(x, y, z):
    """
    คำนวณระยะทาง 3 มิติ แบบยุคลิด (Euclidean distance) จากจุด Origin (0,0,0)
    ไปยังจุด (x, y, z) ที่กำหนด

    Args:
        x (float): พิกัดแกน X
        y (float): พิกัดแกน Y
        z (float): พิกัดแกน Z (ความลึก)

    Returns:
        float: ระยะทาง 3 มิติ, หรือ 0.0 หากข้อมูลนำเข้าไม่ถูกต้อง
    """
    try:
        # ใช้ทฤษฎีบทพีทาโกรัสใน 3 มิติ
        distance = math.sqrt(float(x)**2 + float(y)**2 + float(z)**2)
        return distance
    except ValueError:
        # กรณีแปลงค่าเป็น float ไม่ได้
        print(f"[Warning] Invalid input for calculate_3d_distance: x={x}, y={y}, z={z}")
        return 0.0
    except Exception as e:
        # จัดการ Error อื่นๆ
        print(f"[Error] in calculate_3d_distance: {e}")
        return 0.0