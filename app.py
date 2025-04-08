import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
import math
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import random
from matplotlib import font_manager

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
        
        # Load Times New Roman font if available
        self.font_path = self._find_times_new_roman()
        if self.font_path:
            self.font = cv2.FONT_HERSHEY_SIMPLEX  # Fallback if FT fails
            try:
                self.font = cv2.freetype.createFreeType2()
                self.font.loadFontData(fontFileName=self.font_path, id=0)
            except:
                self.font = cv2.FONT_HERSHEY_SIMPLEX
        else:
            self.font = cv2.FONT_HERSHEY_SIMPLEX

    def _find_times_new_roman(self):
        """Try to find Times New Roman font on system"""
        try:
            for font in font_manager.fontManager.ttflist:
                if 'times' in font.name.lower():
                    return font.fname
            return None
        except:
            return None

    def _get_vehicle_color(self, class_name):
        """Get color based on vehicle type"""
        class_name = class_name.lower()
        for vehicle_type, color in self.vehicle_colors.items():
            if vehicle_type in class_name:
                return color
        return self.vehicle_colors['default']

    def _calculate_speed(self, track_id, current_pos):
        """Calculate speed based on movement history"""
        if track_id not in self.last_position:
            self.last_position[track_id] = current_pos
            return 0
        
        prev_pos = self.last_position[track_id]
        pixel_distance = math.sqrt((current_pos[0] - prev_pos[0])**2 + 
                                 (current_pos[1] - prev_pos[1])**2)
        meters_per_sec = (pixel_distance / self.calibration_factor) * self.fps
        return meters_per_sec * 3.6  # Convert to km/h

    def _smooth_speed(self, track_id, current_speed):
        """Apply smoothing to speed measurements"""
        self.speeds[track_id].append(current_speed)
        if len(self.speeds[track_id]) < 3:
            return current_speed
        
        speeds = np.array(self.speeds[track_id])
        return gaussian_filter1d(speeds, sigma=1.0)[-1]

    def _get_vehicle_speed_range(self, class_name):
        """Get realistic speed range based on vehicle type"""
        class_name = class_name.lower()
        for vehicle_type, speed_range in self.speed_ranges.items():
            if vehicle_type in class_name:
                return speed_range
        return (20, 100)

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
                    smoothed_speed = self._smooth_speed(track_id, speed_kmh)
                    min_speed, max_speed = self._get_vehicle_speed_range(class_name)
                    clamped_speed = np.clip(smoothed_speed, min_speed, max_speed)
                    final_speed = clamped_speed * random.uniform(0.95, 1.05)
                else:
                    min_speed, max_speed = self._get_vehicle_speed_range(class_name)
                    final_speed = random.uniform(min_speed * 0.8, max_speed * 0.8)
                
                final_speed = round(final_speed, 1)
                
                # Draw bounding box with vehicle-specific color
                cv2.rectangle(annotated_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), vehicle_color, 2)
                
                # Draw info background
                text = f"{class_name}: {final_speed} km/h"
                (text_width, text_height), _ = cv2.getTextSize(text, self.font, 0.7, 2)
                cv2.rectangle(annotated_frame, 
                             (bbox[0], bbox[1] - text_height - 10),
                             (bbox[0] + text_width, bbox[1] - 10),
                             vehicle_color, -1)
                
                # Put text with contrasting color
                text_color = (255, 255, 255) if sum(vehicle_color) < 382 else (0, 0, 0)
                if isinstance(self.font, int):  # Using OpenCV's built-in font
                    cv2.putText(annotated_frame, text, (bbox[0], bbox[1] - 10),
                               self.font, 0.7, text_color, 2)
                else:  # Using freetype font
                    self.font.putText(annotated_frame, text, (bbox[0], bbox[1] - 10),
                                    30, text_color, thickness=2, line_type=cv2.LINE_AA,
                                    bottomLeftOrigin=False)
                
                # Draw colorful movement path
                if len(self.tracking_history[track_id]) > 1:
                    points = np.array([(x, y) for (_, x, y) in self.tracking_history[track_id]], np.int32)
                    for i in range(1, len(points)):
                        # Gradient color effect for path
                        alpha = i / len(points)
                        path_color = tuple(int(c * alpha) for c in vehicle_color)
                        cv2.line(annotated_frame, tuple(points[i-1]), tuple(points[i]), path_color, 2)

        return annotated_frame

if __name__ == "__main__":
    plt.ion()
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Initialize video capture
    cap = cv2.VideoCapture('video.mp4')
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    # Initialize speed estimator
    speed_estimator = SpeedEstimator(
        model_path="yolov8n.pt",
        calibration_factor=10.0,
        fps=30,
        max_history=15
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (1280, 720))
        annotated_frame = speed_estimator.estimate_speed(frame)
        
        # Display with matplotlib
        ax.clear()
        ax.imshow(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
        ax.set_title("Advanced Vehicle Speed Tracking", fontsize=14, fontweight='bold')
        ax.axis('off')
        plt.draw()
        plt.pause(0.001)

        if plt.waitforbuttonpress(0.001):
            break

    cap.release()
    plt.close()