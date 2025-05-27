# ไฟล์: camera_handler.py
# คำอธิบาย: คลาสสำหรับจัดการการเชื่อมต่อและอ่านเฟรมจากแหล่งภาพต่างๆ

import cv2
import config # Import การตั้งค่า เช่น Camera ID, Frame Size, FPS, Video Path
import time
import numpy as np

# ตรวจสอบว่ามี RealSense SDK หรือไม่
try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False

class CameraHandler:
    """
    คลาสสำหรับจัดการการทำงานของกล้องหรือวิดีโอ
    รองรับ RealSense, Webcam, และ Video File
    """
    def __init__(self, source_type='webcam'):
        """
        คลาส CameraHandler

        Args:
            source_type (str): ประเภทของแหล่งภาพ ('realsense', 'webcam', 'video')
                               ค่าเริ่มต้นคือ 'webcam'
        """
        self.source_type = source_type
        self.config = config # เก็บ object config ไว้ใช้งาน
        self.pipeline = None # สำหรับ RealSense pipeline
        self.align = None    # สำหรับ RealSense alignment
        self.cap = None      # สำหรับ cv2.VideoCapture (Webcam/Video)
        self.depth_scale = 1.0 # ค่า Scale ของ RealSense Depth
        self.width = self.config.FRAME_WIDTH   # ความกว้างเฟรม
        self.height = self.config.FRAME_HEIGHT # ความสูงเฟรม
        self.initialized = False # สถานะการ khởi tạo สำเร็จหรือไม่

        print(f"Initializing camera source: {self.source_type}")

        if self.source_type == 'realsense':
            if REALSENSE_AVAILABLE:
                self.initialized = self._setup_realsense()
            else:
                print("[Error] RealSense SDK not found, cannot initialize RealSense.")
                self.initialized = False
        elif self.source_type == 'webcam':
            self.initialized = self._setup_webcam()
        elif self.source_type == 'video':
            self.initialized = self._setup_video()
        else:
            print(f"[Error] Unknown source type: {self.source_type}")
            self.initialized = False

        if not self.initialized:
            print(f"[Error] Failed to initialize camera source: {self.source_type}")
            # อาจจะ raise Exception หรือให้โปรแกรมหลักจัดการต่อ
            # raise RuntimeError(f"Failed to initialize camera source: {self.source_type}")

    def _setup_realsense(self):
        """ การเชื่อมต่อกล้อง RealSense """
        try:
            self.pipeline = rs.pipeline()
            rs_config = rs.config()

            # ตรวจสอบ Stream ที่รองรับ (อาจไม่จำเป็นถ้าใช้ค่ามาตรฐาน)
            # pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
            # pipeline_profile = rs_config.resolve(pipeline_wrapper)
            # device = pipeline_profile.get_device()
            # ... (โค้ดตรวจสอบ stream ถ้าต้องการ) ...

            print(f"Configuring RealSense: {self.width}x{self.height} @ {self.config.FPS} FPS")
            rs_config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.config.FPS)
            rs_config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.config.FPS)

            print("Starting RealSense pipeline...")
            profile = self.pipeline.start(rs_config)
            depth_sensor = profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()
            print(f"RealSense Depth Scale: {self.depth_scale}")

            self.align = rs.align(rs.stream.color)
            print("RealSense initialized successfully.")
            return True
        except Exception as e:
            print(f"[Error] Failed to initialize RealSense: {e}")
            if self.pipeline: # พยายาม stop pipeline ถ้ามี error เกิดขึ้น
                try: self.pipeline.stop()
                except: pass
            return False

    def _setup_webcam(self):
        """ การเชื่อมต่อ Webcam """
        try:
            print(f"Opening webcam (ID: {self.config.CAMERA_ID})...")
            self.cap = cv2.VideoCapture(self.config.CAMERA_ID)
            if not self.cap.isOpened():
                print(f"[Warning] Could not open webcam with ID {self.config.CAMERA_ID}, trying default (0)...")
                self.cap = cv2.VideoCapture(0) # ลองเปิด default
                if not self.cap.isOpened():
                    print("[Error] Could not open any webcam.")
                    return False

            print(f"Configuring Webcam: {self.width}x{self.height} @ {self.config.FPS} FPS")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.config.FPS)

            # อ่านค่าจริงที่กล้องตั้งได้ (อาจไม่ตรงกับที่ขอไป)
            actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            print(f"Webcam actual properties: {actual_w}x{actual_h}, {actual_fps:.2f} FPS")
            # อัปเดต width/height ถ้าค่าจริงต่างจากที่ตั้ง (สำคัญต่อการคำนวณ Origin)
            self.width = actual_w
            self.height = actual_h

            print("Webcam initialized successfully.")
            return True
        except Exception as e:
            print(f"[Error] Failed to initialize Webcam: {e}")
            if self.cap:
                try: self.cap.release()
                except: pass
            return False

    def _setup_video(self):
        """ การเปิดไฟล์วิดีโอ """
        try:
            video_path = self.config.VIDEO_FILE_PATH
            if not os.path.exists(video_path):
                print(f"[Error] Video file not found: {video_path}")
                return False

            print(f"Opening video file: {video_path}...")
            self.cap = cv2.VideoCapture(video_path)
            if not self.cap.isOpened():
                print(f"[Error] Could not open video file: {video_path}")
                return False

            # อ่าน width/height จากไฟล์วิดีโอ
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            print(f"Video properties: {self.width}x{self.height}, {actual_fps:.2f} FPS")
            print("Video file opened successfully.")
            return True
        except Exception as e:
            print(f"[Error] Failed to open video file: {e}")
            if self.cap:
                try: self.cap.release()
                except: pass
            return False

    def get_frame(self):
        """
        อ่านเฟรมภาพถัดไปจากแหล่งภาพที่กำหนด

        Returns:
            tuple: (status, color_frame, depth_frame)
                   status (bool): True หากอ่านเฟรมสำเร็จ, False หากมีปัญหาหรือสิ้นสุดวิดีโอ
                   color_frame (numpy.ndarray หรือ None): เฟรมภาพสี BGR
                   depth_frame (pyrealsense2.depth_frame หรือ None): เฟรมข้อมูลความลึก (เฉพาะ RealSense)
        """
        if not self.initialized:
            return False, None, None

        if self.source_type == 'realsense':
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=1000) # เพิ่ม timeout เล็กน้อย
                if not frames:
                    print("[Warning] No frames received from RealSense.")
                    return False, None, None

                aligned_frames = self.align.process(frames)
                depth_frame = aligned_frames.get_depth_frame()
                color_frame_rs = aligned_frames.get_color_frame() # ใช้ชื่อ rs ชั่วคราว

                if not depth_frame or not color_frame_rs:
                    print("[Warning] Could not get aligned depth or color frame.")
                    return False, None, None

                color_image = np.asanyarray(color_frame_rs.get_data())
                return True, color_image, depth_frame
            except RuntimeError as e:
                print(f"[Error] RealSense runtime error during get_frame: {e}")
                # อาจจะลอง restart pipeline หรือจัดการอย่างอื่น
                return False, None, None
            except Exception as e:
                print(f"[Error] Unexpected error during RealSense get_frame: {e}")
                return False, None, None

        elif self.source_type == 'webcam' or self.source_type == 'video':
            if self.cap is None: return False, None, None
            try:
                ret, frame = self.cap.read()
                if not ret:
                    # ถ้าเป็นวิดีโอ อาจจะหมายถึงจบไฟล์
                    # ถ้าเป็น webcam อาจหมายถึงกล้องหลุด
                    # คืน False เพื่อให้ลูปหลักจัดการ (เช่น break)
                    return False, None, None
                return True, frame, None # คืน depth_frame เป็น None
            except Exception as e:
                print(f"[Error] Error reading frame from VideoCapture: {e}")
                return False, None, None
        else:
            return False, None, None # ไม่รู้จัก source type

    def release(self):
        """ ปลดปล่อยทรัพยากรกล้อง/วิดีโอ """
        print(f"Releasing camera source: {self.source_type}")
        if self.pipeline:
            try: self.pipeline.stop()
            except Exception as e: print(f"[Error] Stopping RealSense pipeline: {e}")
        if self.cap:
            try: self.cap.release()
            except Exception as e: print(f"[Error] Releasing VideoCapture: {e}")
        self.initialized = False

    def get_dimensions(self):
        """ คืนค่าความกว้างและความสูงของเฟรม """
        return self.width, self.height

    def get_depth_scale(self):
        """ คืนค่า Depth Scale (สำหรับ RealSense) """
        if self.source_type == 'realsense':
            return self.depth_scale
        else:
            return 1.0 # หรือ None หรือค่าที่เหมาะสมอื่นๆ
