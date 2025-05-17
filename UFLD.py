import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image
from transformers import pipeline
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import matplotlib.pyplot as plt
import pandas as pd

# Clone UFLD repository (only needed if not already cloned)
repo_path = "/kaggle/working/Ultrafast-Lane-Detection-Inference-Pytorch-"
if not os.path.exists(repo_path):
    os.system("git clone https://github.com/ibaiGorordo/Ultrafast-Lane-Detection-Inference-Pytorch-.git")
    print("✅ Cloned Ultrafast-Lane-Detection repository")

# Define paths
detector_path = os.path.join(repo_path, "ultrafastLaneDetector")
model_path = os.path.join(detector_path, "model")
simple_file_path = "/kaggle/working/ultrafastLaneDetector.py"
model_file_path = "/kaggle/working/model.py"

# Write model.py
model_content = """
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class parsingNet(nn.Module):
    def __init__(self, pretrained=False, backbone='18', cls_dim=(201, 18, 4), use_aux=False):
        super(parsingNet, self).__init__()
        self.cls_dim = cls_dim
        self.use_aux = use_aux
        
        if backbone == '18':
            self.model = models.resnet18(pretrained=pretrained)
        else:
            raise NotImplementedError(f"Backbone {backbone} not supported")
        
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Identity()
        
        self.cls = nn.Sequential(
            nn.Linear(1800, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, cls_dim[0] * cls_dim[1] * cls_dim[2])
        )
        
        self.pool = nn.Conv2d(512, cls_dim[2], kernel_size=1)
        
    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        
        target_features = 1800
        channels = x.size(1)
        spatial_size = int((target_features / channels) ** 0.5) + 1
        if spatial_size < 1:
            spatial_size = 1
        x = F.adaptive_avg_pool2d(x, (spatial_size, spatial_size))
        
        x = x.view(x.size(0), -1)
        if x.size(1) > target_features:
            x = x[:, :target_features]
        elif x.size(1) < target_features:
            x = F.pad(x, (0, target_features - x.size(1)))
        
        cls_out = self.cls(x)
        cls_out = cls_out.view(-1, self.cls_dim[2], self.cls_dim[1], self.cls_dim[0])
        
        if self.use_aux:
            pool_out = self.pool(x.view(x.size(0), 512, 1, 1))
            return cls_out, pool_out
        return cls_out
"""

# Write ultrafastLaneDetector.py
ufld_content = """
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultrafastLaneDetector.model import parsingNet

class ModelType:
    CULANE = 0
    TUSIMPLE = 1

class ModelConfig:
    def __init__(self):
        self.num_points = 18
        self.num_lanes = 4
        self.prob_th = 0.3
        self.img_w = 1280
        self.img_h = 720
        self.griding_num = 200
        self.backbone = '18'
        self.ort = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class UltrafastLaneDetector:
    def __init__(self, model_path, model_type=0, use_gpu=True):
        self.model_type = model_type
        self.cfg = ModelConfig()
        self.model = parsingNet(
            pretrained=False,
            backbone=self.cfg.backbone,
            cls_dim=(self.cfg.griding_num + 1, self.cfg.num_points, self.cfg.num_lanes),
            use_aux=False
        ).to(self.cfg.device)
        
        state_dict = torch.load(model_path, map_location=self.cfg.device)
        if 'model' in state_dict:
            state_dict = state_dict['model']
            print(f"✅ Loaded nested state_dict['model'] with keys: {list(state_dict.keys())[:10]}")
        elif all(k.startswith('model.') for k in state_dict.keys()):
            state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
            print(f"✅ Stripped 'model.' prefix from state_dict keys: {list(state_dict.keys())[:10]}")
        
        try:
            self.model.load_state_dict(state_dict, strict=False)
            print("✅ Successfully loaded state_dict (strict=False)")
        except Exception as e:
            print(f"❌ Failed to load state_dict: {e}")
            raise
        self.model.eval()

    def detect_lanes(self, image, draw_points=True):
        input_tensor = self.prepare_input(image)
        output = self.inference(input_tensor)
        self.lanes_points, self.lanes_detected = self.process_output(output, self.cfg)
        output_img = self.draw_lanes(image, self.lanes_points, self.lanes_detected, draw_points)
        fps = -1
        return output_img, fps

    def prepare_input(self, img):
        if img.shape[:2] != (self.cfg.img_h, self.cfg.img_w):
            img = cv2.resize(img, (self.cfg.img_w, self.cfg.img_h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose((2, 0, 1)).astype(np.float32) / 255.0
        img = torch.from_numpy(img).float().to(self.cfg.device)
        return img.unsqueeze(0)

    def inference(self, input_tensor):
        with torch.no_grad():
            output = self.model(input_tensor)
        return output

    def process_output(self, output, cfg):
        processed_output = output[0].reshape(-1, cfg.num_points * 3)
        lanes_points = []
        lanes_detected = []

        for lane_num in range(cfg.num_lanes):
            lane_points = []
            lane_detected = False
            scores = F.softmax(processed_output[lane_num * cfg.num_points: (lane_num + 1) * cfg.num_points, :2], dim=1)[:, 1]
            ranges = processed_output[lane_num * cfg.num_points: (lane_num + 1) * cfg.num_points, 2]
            max_idx = torch.argmax(scores)
            if scores[max_idx] > cfg.prob_th:
                lane_detected = True
                selected_indices = torch.where(scores >= cfg.prob_th)[0]
                for idx in selected_indices:
                    x = (ranges[idx] * cfg.img_w).cpu().numpy()
                    y = ((cfg.num_points - idx - 1) * cfg.img_h / (cfg.num_points - 1)).cpu().numpy()
                    lane_points.append([float(x), float(y)])
            lanes_points.append(lane_points)
            lanes_detected.append(lane_detected)
            print(f"Lane {lane_num + 1}: Detected={lane_detected}, Points={len(lane_points)}")

        return lanes_points, np.array(lanes_detected)

    def draw_lanes(self, img, lanes_points, lanes_detected, draw_points=True):
        output_img = img.copy()
        lane_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
        for lane_num, (lane_points, lane_detected) in enumerate(zip(lanes_points, lanes_detected)):
            if lane_detected:
                for point in lane_points:
                    if draw_points:
                        cv2.circle(output_img, (int(point[0]), int(point[1])), 3, lane_colors[lane_num], -1)
        return output_img

    def get_ego_lane_index(self):
        return 1
"""

# Write files
os.makedirs(model_path, exist_ok=True)
with open(model_file_path, "w") as f:
    f.write(model_content)
with open(simple_file_path, "w") as f:
    f.write(ufld_content)
print(f"✅ Wrote model.py to {model_file_path}")
print(f"✅ Wrote ultrafastLaneDetector.py to {simple_file_path}")

# Add repo path to sys.path
sys.path.append(repo_path)
from ultrafastLaneDetector import UltrafastLaneDetector, ModelType
from ultrafastLaneDetector.model import parsingNet
print("✅ Successfully imported UltrafastLaneDetector and parsingNet")

# Load UFLD model
ufld_model_path = "/kaggle/input/culane/pytorch/default/1/culane_18.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not os.path.exists(ufld_model_path):
    raise FileNotFoundError(f"❌ culane_18.pth not found at {ufld_model_path}")
lane_detector = UltrafastLaneDetector(ufld_model_path, model_type=ModelType.CULANE, use_gpu=(device.type == "cuda"))
lane_detector.model.to(device)
print("✅ UFLD Model loaded")

# Load YOLO and DeepSort
yolo_model = YOLO("yolov8m.pt")
deep_sort = DeepSort(embedder="mobilenet", max_age=30, nms_max_overlap=0.5)

# Load MiDaS for depth estimation
depth_pipeline = pipeline("depth-estimation", model="intel/dpt-large")

# Camera input
cap = cv2.VideoCapture(0)  # Use default camera (index 0)
if not cap.isOpened():
    raise RuntimeError("❌ Failed to open camera")
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
print(f"Camera opened: {frame_width}x{frame_height} @ {fps} FPS")

# Camera parameters
focal_length = frame_width * 0.8
cx, cy = frame_width / 2, frame_height / 2

# Simulate 2D LiDAR
def simulate_2d_lidar(frame_idx, max_range=50, num_points=360):
    angles = np.linspace(0, 2 * np.pi, num_points)
    distances = np.random.uniform(10, max_range, num_points)
    return np.array([[d * np.cos(a), d * np.sin(a)] for a, d in zip(angles, distances)])

# Compute 3D lane points
def compute_lane_points_3d(lanes_points, depth_map, focal_length, cx, cy):
    height, width = depth_map.shape
    lane_points_3d = []
    for lane in lanes_points:
        lane_3d = []
        for point in lane:
            u, v = int(point[0]), int(point[1])
            if 0 <= u < width and 0 <= v < height:
                z = depth_map[v, u]
                if z < 0.1:
                    z = 0.1
                x = (u - cx) * z / focal_length
                y = (v - cy) * z / focal_length
                lane_3d.append([x, y, z])
        lane_points_3d.append(lane_3d)
    return lane_points_3d

# Create BEV
def create_birds_eye_view(lidar_points_2d, detected_objects, lane_points_3d, x_range=(-20, 50), y_range=(-30, 30), grid_size=0.1):
    mask = (lidar_points_2d[:, 0] >= x_range[0]) & (lidar_points_2d[:, 0] <= x_range[1]) & \
           (lidar_points_2d[:, 1] >= y_range[0]) & (lidar_points_2d[:, 1] <= y_range[1])
    points_2d = lidar_points_2d[mask]
    x_bins = int((x_range[1] - x_range[0]) / grid_size)
    y_bins = int((y_range[1] - y_range[0]) / grid_size)
    bev_map = np.zeros((x_bins, y_bins))
    for point in points_2d:
        x, y = point[:2]
        x_idx = int((x - x_range[0]) / grid_size)
        y_idx = int((y - y_range[0]) / grid_size)
        if 0 <= x_idx < x_bins and 0 <= y_idx < y_bins:
            bev_map[x_idx, y_idx] += 1
    bev_map = np.log1p(bev_map)
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(bev_map.T, cmap='viridis', origin='lower', extent=[x_range[0], x_range[1], y_range[0], y_range[1]])
    plt.scatter([0], [0], color='red', s=100, label='Vehicle', marker='^')
    for obj in detected_objects:
        x, y = obj["center"][:2]
        color = [c/255 for c in obj["color"]]
        plt.scatter(x, y, color=color, s=200, alpha=0.7)
        plt.text(x, y+1, f"{obj['class']} ({obj['confidence']*100:.1f}%)", color='white')
    for i, lane in enumerate(lane_points_3d):
        if len(lane) > 0:
            lane_np = np.array(lane)
            plt.plot(lane_np[:, 0], lane_np[:, 1], color=[c/255 for c in [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)][i % 4]], linewidth=2, label=f'Lane {i+1}')
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.title("Bird's-Eye View with Lanes and Objects")
    plt.legend()
    plt.grid(True)
    fig.canvas.draw()
    bev_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    bev_img = bev_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return bev_img

# Dummy path planning (replace with actual implementation)
def plan_path(lanes_points, lanes_detected, detected_objects, lidar_points_2d):
    return ["Maintain lane", "Check surroundings"]

# Process camera feed
frame_idx = 0
print("✅ Processing camera feed... Press 'q' to quit")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to read frame from camera")
        break
    frame_idx += 1
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize and preprocess image
    image = cv2.resize(image, (1280, 720))
    image = cv2.convertScaleAbs(image, alpha=1.5, beta=50)

    # Lane detection
    try:
        lane_img, fps = lane_detector.detect_lanes(image)
        lanes_points = lane_detector.lanes_points
        lanes_detected = lane_detector.lanes_detected
        print(f"Frame {frame_idx} Lanes Detected: {lanes_detected}")
    except Exception as e:
        print(f"⚠ Lane detection failed: {e}")
        lane_img = image.copy()
        lanes_points = [[] for _ in range(4)]
        lanes_detected = np.array([False] * 4)

    # Depth estimation
    depth_result = depth_pipeline(Image.fromarray(image))
    depth_map = np.array(depth_result["depth"])
    depth_map = depth_map / np.max(depth_map) * 50

    # Compute 3D points
    height, width = depth_map.shape
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    z = depth_map
    z[z < 0.1] = 0.1
    x = (u - cx) * z / focal_length
    y = (v - cy) * z / focal_length
    points_3d = np.stack([x, y, z], axis=-1).reshape(-1, 3)

    # Compute 3D lane points
    lane_points_3d = compute_lane_points_3d(lanes_points, depth_map, focal_length, cx, cy)

    # Simulate LiDAR
    lidar_points_2d = simulate_2d_lidar(frame_idx)

    # YOLO and DeepSort
    yolo_results = yolo_model(image, imgsz=1280, conf=0.05)
    detections = []
    tracked_yolo_boxes = []
    for det in yolo_results[0].boxes:
        x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
        conf = float(det.conf[0])
        cls_id = int(det.cls[0])
        class_name = yolo_model.names[cls_id]
        if conf < 0.2:
            continue
        w, h = x2 - x1, y2 - y1
        detections.append(([x1, y1, w, h], conf, cls_id))
        tracked_yolo_boxes.append({
            "bbox": [x1, y1, w, h],
            "cls_id": cls_id,
            "center": (x1 + w // 2, y1 + h // 2),
            "class_name": class_name,
            "conf": conf
        })
    tracks = deep_sort.update_tracks(detections, frame=image)
    detected_objects = []
    img_with_detections = lane_img.copy()
    for track in tracks:
        if not track.is_confirmed():
            continue
        x1, y1, x2, y2 = track.to_ltrb()
        w, h = x2 - x1, y2 - y1
        cx = int(x1 + w / 2)
        matched = None
        for yolo_box in tracked_yolo_boxes:
            yolo_cx, yolo_cy = yolo_box["center"]
            if abs(cx - yolo_cx) < 30:
                matched = yolo_box
                break
        if not matched:
            continue
        x, y, w, h = matched["bbox"]
        cls_id = matched["cls_id"]
        class_name = matched["class_name"]
        conf = matched["conf"]
        color = (0, 255, 255) if class_name == "car" else (255, 255, 0) if class_name == "truck" else (0, 255, 0)
        obj_points = []
        for u in range(int(x), int(x+w)):
            for v in range(int(y), int(y+h)):
                if 0 <= v < height and 0 <= u < width:
                    idx = v * width + u
                    obj_points.append(points_3d[idx])
        if obj_points:
            obj_center = np.mean(obj_points, axis=0)
            detected_objects.append({
                "class": class_name,
                "confidence": conf,
                "center": obj_center,
                "color": color,
                "lane_idx": 1
            })
            cv2.rectangle(img_with_detections, (x, y), (x+w, y+h), color, 2)
            cv2.putText(img_with_detections, f"{class_name}: {obj_center[0]:.2f}m", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Path planning
    path_suggestions = plan_path(lanes_points, lanes_detected, detected_objects, lidar_points_2d)
    for i, suggestion in enumerate(path_suggestions):
        cv2.putText(img_with_detections, suggestion, (10, 30 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Create BEV
    bev_img = create_birds_eye_view(lidar_points_2d, detected_objects, lane_points_3d)

    # Display outputs
    cv2.imshow("Perspective View", cv2.cvtColor(img_with_detections, cv2.COLOR_RGB2BGR))
    cv2.imshow("Bird's-Eye View", cv2.cvtColor(bev_img, cv2.COLOR_RGB2BGR))

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
print("✅ Camera feed processing stopped")