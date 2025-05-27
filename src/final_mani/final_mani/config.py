# ไฟล์: config.py
# คำอธิบาย: เก็บค่าตั้งค่าต่างๆ สำหรับโปรแกรมติดตามวัตถุ XYZ

# --- การตั้งค่าโมเดล ---
# YOLO_MODEL_PATH = r"yolo11l.pt"  # ตำแหน่งไฟล์โมเดล YOLO -> สำหรับ Development (ให้เปลี่ยนเป็น best.pt ตอนใช้งานจริง)
YOLO_MODEL_PATH = r"best.pt"  # ตำแหน่งไฟล์โมเดล YOLO -> สำหรับ Production (ให้เปลี่ยนเป็น yolo11l.pt ตอนใช้งานจริง)
DEPTH_MODEL_PATH = './model/final_calibrated_depth_model_outdoor_v2' # ตำแหน่งไฟล์โมเดล Regression สำหรับ Depth

# --- การตั้งค่า YOLO ---
CONF_THRESHOLD = 0.15           # ค่าความมั่นใจขั้นต่ำในการยอมรับผลการตรวจจับ
TARGET_CLASS_NAME = 'basketball_hoop'    # ชื่อคลาสของวัตถุเป้าหมายที่ต้องการตรวจจับ -> สำหรับ Development (ให้เปลี่ยนเป็น basketball_hoop ตอนใช้งานจริง)

# --- การตั้งค่ากล้อง ---
CAMERA_ID = 0                   # หมายเลข ID ของกล้อง (0 คือกล้องหลักของเครื่อง)
FRAME_WIDTH = 640               # ความกว้างของเฟรมภาพที่ต้องการ
FRAME_HEIGHT = 480              # ความสูงของเฟรมภาพที่ต้องการ
FPS = 30                        # ค่า FPS ของกล้องที่ต้องการ
VIDEO_FILE_PATH = "color_output.mp4" # <<< หากทดสอบเป็น Video (แก้ Path ตามต้องการ)

# --- การตั้งค่าการประมาณค่า Depth สำรอง ---
# (ใช้ในกรณีที่ RealSense หรือ Regression Model ใช้งานไม่ได้)
# ค่าเหล่านี้ควรทำการ Calibrate กับเเป้นบาสหากต้องการความแม่นยำ
REFERENCE_OBJECT_WIDTH = 0.5    # ความกว้างโดยประมาณของวัตถุเป้าหมาย (เมตร)
REFERENCE_OBJECT_HEIGHT = 0.5   # ความสูงโดยประมาณของวัตถุเป้าหมาย (เมตร)
ESTIMATED_FOCAL_LENGTH = 500    # ค่า Focal Length โดยประมาณ (หน่วย pixel) - ควร Calibrate

# --- การตั้งค่าการแสดงผล (Visualization) ---
TARGET_MARKER_COLOR = (0, 255, 255) # สีของสัญลักษณ์ Target (Cyan)
TARGET_MARKER_RADIUS = 8            # รัศมีวงนอกของสัญลักษณ์ Target
TARGET_MARKER_THICKNESS = 2         # ความหนาเส้นวงนอกของสัญลักษณ์ Target
INNER_DOT_RADIUS = 2                # รัศมีจุดดำข้างในสัญลักษณ์ Target

BOUNDING_BOX_COLOR = (0, 255, 0)    # สีของกรอบรอบวัตถุ (เขียว)
BOUNDING_BOX_THICKNESS = 2          # ความหนาเส้นกรอบ

ORIGIN_TARGET_LINE_COLOR = (0, 255, 255) # สีของเส้นเชื่อม Origin-Target (เหลือง)
ORIGIN_TARGET_LINE_THICKNESS = 2         # ความหนาเส้นเชื่อม

COORDINATE_SYSTEM_SIZE = 60         # ขนาดของแกน X, Y ที่วาด
COORDINATE_SYSTEM_THICKNESS = 3     # ความหนาของเส้นแกน

INFO_PANEL_BG_COLOR = (0, 0, 0)     # สีพื้นหลังกล่องข้อความข้อมูล (ดำ)
INFO_PANEL_TEXT_COLOR = (255, 255, 255) # สีข้อความทั่วไป (ขาว)
INFO_PANEL_ANGLE_COLOR = (0, 255, 255)  # สีข้อความมุม (Cyan)
INFO_PANEL_STATUS_COLOR = (0, 165, 255) # สีข้อความสถานะ "ไม่เจอ" (ส้ม)


