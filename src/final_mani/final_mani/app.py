from Callback import VisionPipeline
import cv2
import numpy as np
import time
import os
## ถ้าไม่มีอะไรผิดพลาดกล้องจะเปิดตลอดจาก init ตรงนี้
# pipeline = VisionPipeline(camera_type='webcam', enable_visualization=True)

# try:
#     while True:
#         tracking_data, _ = pipeline.process_single_frame()

        
#         if tracking_data['detected']:
#             x = tracking_data['rel_x']
#             y = tracking_data['rel_y']
#             z = tracking_data['z']
#             angle = tracking_data['angle']
#             confidence = tracking_data['confidence']

#             print(f"Tracking Data: x={x}, y={y}, z={z}, angle={angle}, confidence={confidence}")
 
# except KeyboardInterrupt:
#     print("Exiting...")
# finally:
#     pipeline.stop()


# for debugging
# realsense
pipeline = VisionPipeline(camera_type='realsense', enable_visualization=True, enable_save_video=True)

# Create a folder to save frames if it doesn't exist
output_folder = f"output_frames_{time.strftime('%Y%m%d_%H%M%S')}"  # Use current timestamp in folder name
os.makedirs(output_folder, exist_ok=True)
frames = []
while True:
    tracking_data, vis_img = pipeline.process_single_frame()
    # Process tracking_data
    # ...
    
    # Show visualization
    if vis_img is not None:
        cv2.imshow("Debug View", vis_img)
        
        # Save the current frame to the folder
        frame_filename = os.path.join(output_folder, f"frame_{cv2.getTickCount()}.jpg")
        cv2.imwrite(frame_filename, vis_img)
        
        frames.append(vis_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            pipeline.save_video(frames, "output7.mp4", fps=15)
            break
        
        if tracking_data['detected']:
            x = tracking_data['rel_x']
            y = tracking_data['rel_y']
            z = tracking_data['z']
            angle = tracking_data['angle']
            # confidence = tracking_data['confidence']

            print(f"Tracking Data: x={x}, y={y}, z={z}, angle={angle}")

pipeline.stop()
cv2.destroyAllWindows()