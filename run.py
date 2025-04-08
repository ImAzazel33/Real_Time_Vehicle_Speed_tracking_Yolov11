import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
import math
import random

# SpeedEstimator Class (Enhanced Version)
class SpeedEstimator:
    def __init__(self, model_path, calibration_factor=8.33, fps=30, max_history=15):
        """
        Initialize the speed estimator with enhanced tracking and visual styling.
        :param model_path: Path to the YOLO model file.
        :param calibration_factor: Pixels per meter (calibration constant).
        :param fps: Frames per second of the video.
        :param max_history: Maximum number of positions to track for each vehicle.
        """
        self.model = YOLO(model_path)
        self.calibration_factor = calibration_factor
        self.fps = fps
        self.tracking_history = defaultdict(lambda: deque(maxlen=max_history))
        self.speeds = defaultdict(lambda: deque(maxlen=5))
        self.vehicle_types = defaultdict(str)
        
        # Color scheme for different vehicle classes
        self.vehicle_colors = {
            'car': (0, 191, 255),      # Deep Sky Blue
            'truck': (255, 165, 0),     # Orange
            'bus': (220, 20, 60),       # Crimson Red
            'motorcycle': (50, 205, 50), # Lime Green
            'bicycle': (138, 43, 226),  # Blue Violet
            'default': (255, 255, 0)     # Yellow
        }
        
        self.speed_ranges = {
            'car': (30, 120),
            'truck': (20, 90),
            'bus': (20, 80),
            'motorcycle': (40, 130),
            'bicycle': (10, 30)
        }
        
        self.last_position = defaultdict(tuple)
        self.frame_count = 0

    def _get_vehicle_color(self, class_name):
        """Get color based on vehicle type"""
        class_name = class_name.lower()
        for vehicle_type, color in self.vehicle_colors.items():
            if vehicle_type in class_name:
                return color
        return self.vehicle_colors['default']

    def estimate_speed(self, frame):
        """Estimate and display the speed of vehicles with enhanced visuals"""
        self.frame_count += 1
        results = self.model.track(frame, persist=True)
        annotated_frame = frame.copy()

        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                if box.id is None:
                    continue
                    
                track_id = int(box.id[0])
                cls = int(box.cls[0])
                class_name = self.model.names[cls]
                bbox = box.xyxy[0].astype(int)
                
                # Get vehicle-specific color
                vehicle_color = self._get_vehicle_color(class_name)
                
                # Store vehicle type if new
                if track_id not in self.vehicle_types:
                    self.vehicle_types[track_id] = class_name
                
                # Calculate center point
                cx, cy = (bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2
                self.tracking_history[track_id].append((self.frame_count, cx, cy))
                
                # Calculate speed
                if len(self.tracking_history[track_id]) >= 2:
                    (frame1, x1, y1), (frame2, x2, y2) = self.tracking_history[track_id][-2], self.tracking_history[track_id][-1]
                    time_diff = max((frame2 - frame1) / self.fps, 1/self.fps)
                    pixel_distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    speed_kmh = (pixel_distance / self.calibration_factor) / time_diff * 3.6
                    final_speed = round(speed_kmh, 1)
                else:
                    final_speed = 0
                
                # Draw bounding box with vehicle-specific color
                cv2.rectangle(annotated_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), vehicle_color, 2)
                
                # Draw info background
                text = f"{class_name}: {final_speed} km/h"
                (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(annotated_frame, 
                             (bbox[0], bbox[1] - text_height - 10),
                             (bbox[0] + text_width, bbox[1] - 10),
                             vehicle_color, -1)
                
                # Put text with contrasting color
                text_color = (255, 255, 255) if sum(vehicle_color) < 382 else (0, 0, 0)
                cv2.putText(annotated_frame, text, (bbox[0], bbox[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
                
                # Draw colorful movement path (gradient based on speed)
                if len(self.tracking_history[track_id]) > 1:
                    points = np.array([(x, y) for (_, x, y) in self.tracking_history[track_id]], np.int32)
                    for i in range(1, len(points)):
                        # Gradient color effect for path (green -> yellow -> red)
                        alpha = min(final_speed / 120, 1.0)
                        path_color = (
                            int(255 * alpha),          # Red
                            int(255 * (1 - alpha)),    # Green
                            0                          # Blue
                        )
                        cv2.line(annotated_frame, tuple(points[i-1]), tuple(points[i]), path_color, 2)

        return annotated_frame


# Streamlit App Code
def main():
    st.set_page_config(
        page_title="Vehicle Speed Estimation 🚗",
        page_icon="🎥",
        layout="wide"
    )

    # Header with emojis and colorful fonts
    st.markdown("""
    <h1 style='color: #FF5733; text-align: center;'>🚗 Vehicle Speed Estimation with YOLO V11🚦</h1>
    <p style='color: #33FF57; text-align: center;'>Upload a video and watch the magic happen! ✨</p>
    """, unsafe_allow_html=True)

    # Sidebar for configuration
    st.sidebar.header("⚙️ Configuration Panel")
    uploaded_file = st.sidebar.file_uploader("📂 Upload a Video File", type=["mp4", "avi", "mov"])
    calibration_factor = st.sidebar.slider("📏 Calibration Factor (Pixels per Meter)", 5.0, 20.0, 10.0, 0.5)
    fps = st.sidebar.number_input("📽️ Frames Per Second (FPS)", 10, 60, 30)
    max_history = st.sidebar.slider("📚 Max Tracking History", 5, 30, 15)

    if uploaded_file is not None:
        # Save the uploaded file temporarily
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Initialize SpeedEstimator
        speed_estimator = SpeedEstimator(
            model_path="yolov8n.pt",
            calibration_factor=calibration_factor,
            fps=fps,
            max_history=max_history
        )

        # Open the video file
        cap = cv2.VideoCapture("temp_video.mp4")
        if not cap.isOpened():
            st.error("🚨 Error: Could not open video.")
            return

        stframe = st.empty()  # Placeholder for displaying frames
        st.sidebar.subheader("📊 Real-Time Speeds ⏱️")
        speed_display = st.sidebar.empty()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (1280, 720))
            annotated_frame = speed_estimator.estimate_speed(frame)
            
            # Display the annotated frame in Streamlit
            stframe.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), channels="RGB")

            # Update real-time speed display in sidebar
            speeds_text = "\n".join([f"👉 {k}: {v[-1]:.1f} km/h" for k, v in speed_estimator.speeds.items()])
            speed_display.markdown(f"<pre style='color: #FFD700;'>{speeds_text}</pre>", unsafe_allow_html=True)

        cap.release()
        st.success("🎉 Video processing completed!")


if __name__ == "__main__":
    main()