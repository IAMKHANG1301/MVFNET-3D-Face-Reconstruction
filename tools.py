import numpy as np
import scipy.io as io
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import sys
import math
import torch
import os
import trimesh
import matplotlib.pyplot as plt # SỬA: Thêm import
from scipy.spatial import ConvexHull # SỬA: Thêm import
from scipy.interpolate import interp1d # SỬA: Thêm import
from ultralytics import YOLO
from gfpgan import GFPGANer

# --- Cấu hình mô hình bổ trợ ---
# Tải detector YOLOv11-Face
face_detector = YOLO('weights/yolo11n-face.pt') 

# Khởi tạo bộ làm nét GFPGAN (chạy trên GPU nếu có)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
face_restorer = GFPGANer(
    model_path='weights/GFPGANv1.4.pth',
    upscale=1,
    arch='clean',
    channel_multiplier=2,
    device=device
)

# Load dữ liệu 3DMM
model_shape = io.loadmat('data/Model_Shape.mat')
kpt_index = np.reshape(model_shape['keypoints'], 68).astype(np.int32) - 1
model_exp = io.loadmat('data/Model_Expression.mat')
data = io.loadmat('data/sigma_exp.mat')
pose_mean = np.array([0,0,0,112,112,0,0]).astype(np.float32)
pose_std = np.array([math.pi/2.0,math.pi/2.0,math.pi/2.0,56,56,1,224.0 / (2 * 180000.0)]).astype(np.float32)

# --- HÀM CẢI TIẾN ---

def enhance_face(image_pil):
    """Sử dụng GFPGAN để làm nét khuôn mặt với cơ chế an toàn"""
    img_np = np.array(image_pil)
    try:
        # has_aligned=False vì ảnh crop từ YOLO chưa xoay thẳng chuẩn
        _, _, restored_img = face_restorer.enhance(
            img_np, has_aligned=False, only_center_face=True, paste_back=True
        )
        if restored_img is not None:
            return Image.fromarray(restored_img)
        return image_pil
    except Exception as e:
        print(f"  [!] Warning GFPGAN: {e}")
        return image_pil

def crop_image_yolo(image, res=224):
    """Crop khuôn mặt bằng YOLOv11-Face"""
    results = face_detector(np.array(image), conf=0.5, verbose=False)
    if len(results[0].boxes) == 0:
        return image.resize((res, res), Image.BICUBIC)
    
    box = results[0].boxes.xyxy[0].cpu().numpy()
    x1, y1, x2, y2 = box
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    size = int(max(x2 - x1, y2 - y1) * 1.2)
    
    left, top = int(center_x - size/2), int(center_y - size/2)
    crop_img = image.crop((left, top, left + size, top + size))
    return crop_img.resize((res, res), Image.BICUBIC)

def refine_mesh(vertices, triangles):
    """Làm mượt mesh bằng Laplacian Smoothing"""
    mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
    refined_vertices = trimesh.smoothing.filter_laplacian(mesh, iterations=2)
    return refined_vertices.vertices

# --- HÀM TOÁN HỌC & CHUYỂN ĐỔI ---

def angle_to_rotation(angles):
    phi, gamma, theta = angles
    R_x = np.array([[1, 0, 0], [0, math.cos(phi), math.sin(phi)], [0, -math.sin(phi), math.cos(phi)]])
    R_y = np.array([[math.cos(gamma), 0, -math.sin(gamma)], [0, 1, 0], [math.sin(gamma), 0, math.cos(gamma)]])
    R_z = np.array([[math.cos(theta), math.sin(theta), 0], [-math.sin(theta), math.cos(theta), 0], [0, 0, 1]])
    return np.matmul(np.matmul(R_x, R_y), R_z)

def preds_to_pose(preds):
    pose = preds * pose_std + pose_mean
    return angle_to_rotation(pose[:3]), pose[3:5], pose[6]

def preds_to_shape(preds):
    alpha = np.reshape(preds[:199], [199,1]) * np.reshape(model_shape['sigma'], [199,1])
    beta = np.reshape(preds[199:228], [29, 1]) * 1.0/(1000.0 * np.reshape(data['sigma_exp'], [29, 1]))
    face_shape = np.matmul(model_shape['w'], alpha) + np.matmul(model_exp['w_exp'], beta) + model_shape['mu_shape']
    face_shape = face_shape.reshape(-1, 3)
    
    # Tính Landmark cho 3 view
    kpts = []
    for i in range(3):
        start = 228 + (i * 7)
        R, t, s = preds_to_pose(preds[start : start+7])
        kpt = np.matmul(face_shape[kpt_index], s*R[:2].transpose()) + np.reshape(t, [1,2])
        kpt[:, 1] = 224 - kpt[:, 1]
        kpts.append(kpt)
        
    return [face_shape, model_shape['tri'].astype(np.int64).transpose() - 1] + kpts

def sample_texture_fusion(face_shape, triangles, images, preds):
    """Hàm lấy màu hòa trộn từ 3 góc nhìn (Đã mở khóa và sửa lỗi)"""
    h, w = 224, 224
    num_verts = face_shape.shape[0]
    
    # Tính pháp tuyến (Normals)
    mesh = trimesh.Trimesh(vertices=face_shape, faces=triangles)
    vertex_normals = mesh.vertex_normals

    view_directions = [
        np.array([0, 0, 1]),    # Front
        np.array([-0.8, 0, 0.6]), # Left
        np.array([0.8, 0, 0.6])   # Right
    ]
    
    final_colors = np.zeros((num_verts, 3), dtype=np.float32)
    total_weights = np.zeros((num_verts, 1), dtype=np.float32)

    for i, img in enumerate(images):
        img_np = np.array(img.resize((224, 224))).astype(np.float32)
        start_idx = 228 + (i * 7)
        R, t2d, s = preds_to_pose(preds[start_idx : start_idx + 7])

        projected = np.matmul(face_shape, s * R[:2].T) + t2d
        coords_x = np.clip(projected[:, 0], 0, w - 1).astype(np.int32)
        coords_y = np.clip(224 - projected[:, 1], 0, h - 1).astype(np.int32)

        weight = np.maximum(0.01, np.sum(vertex_normals * view_directions[i], axis=1, keepdims=True))
        weight = np.power(weight, 1.5)

        final_colors += img_np[coords_y, coords_x] * weight
        total_weights += weight

    final_colors /= (total_weights + 1e-6)
    return np.clip(final_colors, 0, 255).astype(np.uint8)

# --- HÀM LƯU FILE & ĐÁNH GIÁ ---

def write_ply(filename, points=None, mesh=None, colors=None, as_text=True):
    # (Giữ nguyên logic của bạn nhưng thêm check colors)
    if not filename.endswith('ply'): filename += '.ply'
    
    num_verts = len(points)
    num_faces = len(mesh)
    
    with open(filename, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {num_verts}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if colors is not None:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element face {num_faces}\n")
        f.write("property list uchar int vertex_indices\nend_header\n")
        
        for i in range(num_verts):
            line = f"{points[i,0]} {points[i,1]} {points[i,2]}"
            if colors is not None:
                line += f" {int(colors[i,0])} {int(colors[i,1])} {int(colors[i,2])}"
            f.write(line + "\n")
            
        for i in range(num_faces):
            f.write(f"3 {mesh[i,0]} {mesh[i,1]} {mesh[i,2]}\n")
    return True

def calculate_nme(pred_landmarks, gt_landmarks):
    # (Giữ nguyên hàm tính NME của bạn)
    min_xy = np.min(gt_landmarks, axis=0)
    max_xy = np.max(gt_landmarks, axis=0)
    bbox_size = np.sqrt(np.prod(max_xy[:2] - min_xy[:2]))
    error = np.mean(np.linalg.norm(pred_landmarks[:, :2] - gt_landmarks[:, :2], axis=1))
    return error / bbox_size

def plot_ced_curve(nme_list, save_path='result/ced_curve.png'):
    # (Hàm này giờ đã có plt nên sẽ không lỗi)
    errors = np.sort(nme_list)
    ced = np.arange(1, len(errors) + 1) / len(errors)
    plt.figure()
    plt.plot(errors, ced)
    plt.savefig(save_path)
    plt.close()