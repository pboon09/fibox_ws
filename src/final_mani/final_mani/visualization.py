# ไฟล์: visualization.py
# คำอธิบาย: ฟังก์ชันสำหรับวาดองค์ประกอบต่างๆ ลงบนเฟรมภาพ

import cv2
import config # Import การตั้งค่าสี, ขนาด, ฯลฯ

def draw_coordinate_system(image, orig_x, orig_y):
    """
    วาดแกนพิกัด X, Y, Z ที่จุด Origin บนภาพ
    ใช้ค่าขนาดและความหนาจาก config.py

    Args:
        image (numpy.ndarray): ภาพที่ต้องการวาดทับ
        orig_x (int): พิกัด X ของ Origin
        orig_y (int): พิกัด Y ของ Origin
    """
    size = config.COORDINATE_SYSTEM_SIZE
    thickness = config.COORDINATE_SYSTEM_THICKNESS
    font_scale = 0.6
    font = cv2.FONT_HERSHEY_SIMPLEX

    # แกน X (แดง) ชี้ขวา
    cv2.arrowedLine(image, (orig_x, orig_y), (orig_x + size, orig_y), (0, 0, 255), thickness)
    cv2.putText(image, "X", (orig_x + size + 5, orig_y + 10), font, font_scale, (0, 0, 255), 2)
    # แกน Y (เขียว) ชี้ขึ้น
    cv2.arrowedLine(image, (orig_x, orig_y), (orig_x, orig_y - size), (0, 255, 0), thickness)
    cv2.putText(image, "Y", (orig_x + 10, orig_y - size - 5), font, font_scale, (0, 255, 0), 2)
    # แกน Z (น้ำเงิน) จุด
    cv2.circle(image, (orig_x, orig_y), thickness + 2, (255, 0, 0), -1)
    cv2.circle(image, (orig_x, orig_y), thickness + 7, (255, 0, 0), 1)
    cv2.putText(image, "Z", (orig_x - 25, orig_y + 10), font, font_scale, (255, 0, 0), 2)
    # ป้าย Origin
    cv2.putText(image, "Origin", (orig_x - 30, orig_y + 30), font, font_scale*0.9, (255,255,255), 1)

def draw_detection_results(image, detection_data, class_names):
    """
    วาดผลลัพธ์การตรวจจับลงบนภาพ (กรอบ, ป้ายชื่อ, จุดศูนย์กลาง Target)

    Args:
        image (numpy.ndarray): ภาพที่ต้องการวาดทับ
        detection_data (dict): ข้อมูลการตรวจจับที่ได้จาก tracker.py หรือส่วนประมวลผล
                                ควรมี key เช่น 'box', 'conf', 'class_id', 'center_x', 'center_y'
        class_names (dict): Dictionary ที่ map class_id ไปยังชื่อคลาส
    """
    if not detection_data or not detection_data.get('box'):
        return # ไม่มีข้อมูลให้วาด

    x1, y1, x2, y2 = map(int, detection_data['box'])
    conf = detection_data.get('conf', 0.0)
    class_id = detection_data.get('class_id', -1)
    center_x = detection_data.get('center_x')
    center_y = detection_data.get('center_y')

    # --- วาดกรอบ Bounding Box ---
    color = config.BOUNDING_BOX_COLOR
    thickness = config.BOUNDING_BOX_THICKNESS
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    # --- วาดป้ายชื่อคลาสและความมั่นใจ ---
    if class_id != -1 and class_id in class_names:
        label = f"{class_names[class_id]} {conf:.2f}"
        cv2.putText(image, label, (x1, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # --- วาดจุดศูนย์กลาง Target ---
    if center_x is not None and center_y is not None:
        marker_color = config.TARGET_MARKER_COLOR
        radius = config.TARGET_MARKER_RADIUS
        marker_thickness = config.TARGET_MARKER_THICKNESS
        dot_radius = config.INNER_DOT_RADIUS
        # วาดวงนอก
        cv2.circle(image, (center_x, center_y), radius, marker_color, marker_thickness)
        # วาดจุดดำข้างใน
        cv2.circle(image, (center_x, center_y), dot_radius, (0,0,0), -1)
        # เพิ่ม Text Label "Target"
        cv2.putText(image, "Target", (center_x + radius + 5, center_y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, marker_color, 1)

def draw_origin_to_target_line(image, origin_x, origin_y, target_x, target_y):
    """
    วาดเส้นเชื่อมจาก Origin ไปยังจุดศูนย์กลางของ Target

    Args:
        image (numpy.ndarray): ภาพที่ต้องการวาดทับ
        origin_x (int): พิกัด X ของ Origin
        origin_y (int): พิกัด Y ของ Origin
        target_x (int): พิกัด X ของ Target
        target_y (int): พิกัด Y ของ Target
    """
    if target_x is not None and target_y is not None:
        color = config.ORIGIN_TARGET_LINE_COLOR
        thickness = config.ORIGIN_TARGET_LINE_THICKNESS
        cv2.line(image, (origin_x, origin_y), (target_x, target_y), color, thickness)

def draw_info_panel(image, tracking_data):
    """
    วาดกล่องแสดงข้อมูล Tracking มุมซ้ายล่าง

    Args:
        image (numpy.ndarray): ภาพที่ต้องการวาดทับ
        tracking_data (dict): ข้อมูลที่ได้จาก tracker.py ควรมี key เช่น
                               'detected', 'rel_x', 'rel_y', 'z', 'dist', 'angle'
    """
    height, _ = image.shape[:2] # หาความสูงของภาพ
    info_text_y_start = height - 15
    line_height = 20
    bg_color = config.INFO_PANEL_BG_COLOR
    text_color = config.INFO_PANEL_TEXT_COLOR
    angle_color = config.INFO_PANEL_ANGLE_COLOR
    status_color = config.INFO_PANEL_STATUS_COLOR
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1

    info_lines_data = [] # เก็บข้อมูลเป็น tuple (label, value, color)
    if tracking_data.get('detected', False):
         rel_x = tracking_data.get('rel_x', 0.0)
         rel_y = tracking_data.get('rel_y', 0.0)
         z = tracking_data.get('z', 0.0)
         dist = tracking_data.get('dist', 0.0)
         angle = tracking_data.get('angle', 0.0)
         info_lines_data.append( ("Pos:", f"X{rel_x:+.0f} Y{rel_y:+.0f} Z{z:.2f}m", text_color) )
         info_lines_data.append( ("Dist:", f"{dist:.2f} m", text_color) )
         info_lines_data.append( ("Angle:", f"{angle:.1f} deg", angle_color) )
    else:
         info_lines_data.append( ("Status:", "Target not detected", status_color) )

    # คำนวณขนาด Panel และวาด
    if info_lines_data:
        max_text_width = 0
        for label, value, color in info_lines_data:
            text = label + " " + value
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            if text_size[0] > max_text_width: max_text_width = text_size[0]

        panel_width = max_text_width + 20 # เพิ่ม padding
        panel_height = len(info_lines_data) * line_height + 10
        panel_x = 5; panel_y = info_text_y_start - panel_height

        cv2.rectangle(image, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), bg_color, -1)
        y_pos = panel_y + int(line_height * 0.7) # เริ่มวาดข้อความ
        for label, value, color in info_lines_data:
            cv2.putText(image, label, (panel_x + 5, y_pos), font, font_scale, text_color, thickness) # วาด Label
            value_x = panel_x + 5 + cv2.getTextSize(label + " ", font, font_scale, thickness)[0][0] # หาตำแหน่ง Value
            cv2.putText(image, value, (value_x - 5, y_pos), font, font_scale, color, thickness) # วาด Value ด้วยสีที่กำหนด
            y_pos += line_height

def draw_ui_elements(image, mode_text, fps):
    """
    วาดองค์ประกอบ UI อื่นๆ เช่น Mode, FPS

    Args:
        image (numpy.ndarray): ภาพที่ต้องการวาดทับ
        mode_text (str): ข้อความแสดง Mode การทำงาน
        fps (float): ค่า FPS ที่คำนวณได้
    """
    font_ui = cv2.FONT_HERSHEY_SIMPLEX
    font_scale_ui = 0.55
    thickness_ui = 2
    # Mode (มุมซ้ายบน)
    cv2.putText(image, f"Mode: {mode_text}", (10, 25), font_ui, font_scale_ui, (255,255,255), thickness_ui)
    # FPS (มุมซ้ายบน)
    cv2.putText(image, f"FPS: {fps:.1f}", (10, 50), font_ui, font_scale_ui, (0, 255, 0), thickness_ui) # สีเขียวสด

def draw_instructions(image):
    """
    วาดข้อความแนะนำการใช้งาน (ปุ่ม q)

    Args:
        image (numpy.ndarray): ภาพที่ต้องการวาดทับ
    """
    height, width = image.shape[:2]
    instruction_y = height - 5
    cv2.putText(image, "q: Quit",
               (width - 50, instruction_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1) # มุมขวาล่าง