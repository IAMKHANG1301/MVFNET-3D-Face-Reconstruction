import face_alignment
import numpy as np
import scipy.io as io
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import sys
import math
import matplotlib.pyplot as plt
import os

model_shape = io.loadmat('data/Model_Shape.mat')
kpt_index = np.reshape(model_shape['keypoints'], 68).astype(np.int32) - 1
model_exp = io.loadmat('data/Model_Expression.mat')
data = io.loadmat('data/sigma_exp.mat')
pose_mean = np.array([0,0,0,112,112,0,0]).astype(np.float32)
pose_std = np.array([math.pi/2.0,math.pi/2.0,math.pi/2.0,56,56,1,224.0 / (2 * 180000.0)]).astype(np.float32)

def angle_to_rotation(angles):
    phi = angles[0]
    gamma = angles[1]
    theta = angles[2]
    
    R_x = np.eye(3)
    R_x[1, 1] = math.cos(phi)
    R_x[1, 2] = math.sin(phi)
    R_x[2, 1] = - math.sin(phi)
    R_x[2, 2] = math.cos(phi)

    R_y = np.eye(3)
    R_y[0, 0] = math.cos(gamma)
    R_y[0, 2] = - math.sin(gamma)
    R_y[2, 0] = math.sin(gamma)
    R_y[2, 2] = math.cos(gamma)

    R_z = np.eye(3)
    R_z[0, 0] = math.cos(theta)
    R_z[0, 1] = math.sin(theta)
    R_z[1, 0] = - math.sin(theta)
    R_z[1, 1] = math.cos(theta)

    return np.matmul(np.matmul(R_x, R_y), R_z)

def preds_to_pose(preds):
    pose = preds * pose_std + pose_mean
    R = angle_to_rotation(pose[:3])
    t2d = pose[3:5]
    s = pose[6]
    return R, t2d, s

def preds_to_shape(preds):
    # paras = torch.mul(preds[:228, :], label_std[:199+29, :])
    alpha = np.reshape(preds[:199], [199,1]) * np.reshape(model_shape['sigma'], [199,1])
    beta = np.reshape(preds[199:228], [29, 1]) * 1.0/(1000.0 * np.reshape(data['sigma_exp'], [29, 1]))
    face_shape = np.matmul(model_shape['w'], alpha) + np.matmul(model_exp['w_exp'], beta) + model_shape['mu_shape']
    face_shape = face_shape.reshape(-1, 3)
    
    R, t, s = preds_to_pose(preds[228:228+7])
    kptA = np.matmul(face_shape[kpt_index], s*R[:2].transpose()) + np.repeat(np.reshape(t,[1,2]), 68, axis=0) 
    kptA[:, 1] = 224 - kptA[:, 1]
    R, t, s = preds_to_pose(preds[228+7:228+14])
    kptB = np.matmul(face_shape[kpt_index], s*R[:2].transpose()) + np.repeat(np.reshape(t,[1,2]), 68, axis=0)
    kptB[:, 1] = 224 - kptB[:, 1]
    
    R, t, s = preds_to_pose(preds[228+14:])
    kptC = np.matmul(face_shape[kpt_index], s*R[:2].transpose()) + np.repeat(np.reshape(t,[1,2]), 68, axis=0)
    kptC[:, 1] = 224 - kptC[:, 1]
    return [face_shape, model_shape['tri'].astype(np.int64).transpose() - 1, kptA, kptB, kptC]
    
    
def crop_image(image, res=224):
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.THREE_D, flip_input=False)
    pts = fa.get_landmarks(np.array(image))
    if len(pts) < 1:
        assert "No face detected!"
    pts = np.array(pts[0]).astype(np.int32)
        
    h = image.size[1]
    w = image.size[0]
        # x-width-pts[0,:], y-height-pts[1,:]
    x_max = np.max(pts[:68, 0])
    x_min = np.min(pts[:68, 0])
    y_max = np.max(pts[:68, 1])
    y_min = np.min(pts[:68, 1])
    bbox = [y_min, x_min, y_max, x_max]
    # c (cy, cx)
    c = [bbox[2] - (bbox[2] - bbox[0]) / 2, bbox[3] - (bbox[3] - bbox[1]) / 2.0]
    c[0] = c[0] - (bbox[2] - bbox[0]) * 0.12
    s = (max(bbox[2] - bbox[0], bbox[3] - bbox[1]) * 1.5).astype(np.int32)
    old_bb = np.array([c[0] - s / 2, c[1] - s / 2, c[0] + s / 2, c[1] + s / 2]).astype(np.int32)
    crop_img = Image.new('RGB', (s, s))
    #crop_img = torch.zeros(image.shape[0], s, s, dtype=torch.float32)

    shift_x = 0 - old_bb[1]
    shift_y = 0 - old_bb[0]
    old_bb = np.array([max(0, old_bb[0]), max(0, old_bb[1]),
              min(h, old_bb[2]), min(w, old_bb[3])]).astype(np.int32)
    hb = old_bb[2] - old_bb[0]
    wb = old_bb[3] - old_bb[1]
    new_bb = np.array([max(0, shift_y), max(0, shift_x), max(0, shift_y) + hb, max(0, shift_x) + wb]).astype(np.int32)
    cache = image.crop((old_bb[1], old_bb[0], old_bb[3], old_bb[2]))
    crop_img.paste(cache, (new_bb[1], new_bb[0], new_bb[3], new_bb[2]))
    crop_img = crop_img.resize((res, res), Image.BICUBIC)
    return crop_img

def write_ply(filename, points=None, mesh=None, colors=None, as_text=True):
    points = pd.DataFrame(points, columns=["x", "y", "z"])
    mesh = pd.DataFrame(mesh, columns=["v1", "v2", "v3"])
    if colors is not None:
        colors = pd.DataFrame(colors, columns=["red", "green", "blue"])
        points = pd.concat([points, colors], axis=1)
    """
 
    Parameters
    ----------
    filename: str
        The created file will be named with this
    points: ndarray
    mesh: ndarray
    as_text: boolean
        Set the write mode of the file. Default: binary
 
    Returns
    -------
    boolean
        True if no problems
 
    """
    if not filename.endswith('ply'):
        filename += '.ply'

    # open in text mode to write the header
    with open(filename, 'w') as ply:
        header = ['ply']

        if as_text:
            header.append('format ascii 1.0')
        else:
            header.append('format binary_' + sys.byteorder + '_endian 1.0')

        if points is not None:
            header.extend(describe_element('vertex', points))
        if mesh is not None:
            mesh = mesh.copy()
            mesh.insert(loc=0, column="n_points", value=3)
            mesh["n_points"] = mesh["n_points"].astype("u1")
            header.extend(describe_element('face', mesh))

        header.append('end_header')

        for line in header:
            ply.write("%s\n" % line)

    if as_text:
        if points is not None:
            points.to_csv(filename, sep=" ", index=False, header=False, mode='a',
                          encoding='ascii')
        if mesh is not None:
            mesh.to_csv(filename, sep=" ", index=False, header=False, mode='a',
                        encoding='ascii')

    else:
        # open in binary/append to use tofile
        with open(filename, 'ab') as ply:
            if points is not None:
                points.to_records(index=False).tofile(ply)
            if mesh is not None:
                mesh.to_records(index=False).tofile(ply)

    return True

def describe_element(name, df):
    """ Takes the columns of the dataframe and builds a ply-like description
    Parameters
    ----------
    name: str
    df: pandas DataFrame
    Returns
    -------
    element: list[str]
    """
    property_formats = {'f': 'float', 'u': 'uchar', 'i': 'int'}
    element = ['element ' + name + ' ' + str(len(df))]

    if name == 'face':
        element.append("property list uchar int vertex_indices")

    else:
        for i in range(len(df.columns)):
            # get first letter of dtype to infer format
            f = property_formats[str(df.dtypes.iloc[i])[0]]
            element.append('property ' + f + ' ' + str(df.columns.values[i]))

    return element

def calculate_nme(pred_landmarks, gt_landmarks):
    """Tính chỉ số Normalized Mean Error (NME)"""
    # Tính kích thước bounding box từ ground truth
    min_xy = np.min(gt_landmarks, axis=0)
    max_xy = np.max(gt_landmarks, axis=0)
    bbox_size = np.sqrt(np.prod(max_xy[:2] - min_xy[:2]))
    
    # Tính sai số
    error = np.mean(np.linalg.norm(pred_landmarks[:, :2] - gt_landmarks[:, :2], axis=1))
    return error / bbox_size

def plot_ced_curve(nme_list, save_path='result/ced_curve.png'):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    errors = np.sort(nme_list)
    ced = np.arange(1, len(errors) + 1) / len(errors)
    
    plt.figure(figsize=(8, 6))
    plt.plot(errors, ced, linewidth=2.5, color='#e74c3c', label='MVF-Net (Our Model)')
    plt.xlim([0, 0.1])
    plt.ylim([0, 1.0])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.title('Cumulative Error Distribution (CED) on AFLW2000-3D', fontsize=14)
    plt.xlabel('Normalized Mean Error (NME)', fontsize=12)
    plt.ylabel('Fraction of Images', fontsize=12)
    plt.legend(loc='lower right', fontsize=12)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[*] Đã lưu biểu đồ CED Curve tại: {save_path}")

def plot_error_by_yaw(nme_list, yaw_list, save_path='result/error_by_yaw.png'):
    yaw_bins = {'[0, 30]': [], '[30, 60]': [], '[60, 90]': []}
    for nme, yaw in zip(nme_list, yaw_list):
        abs_yaw = abs(yaw) * (180.0 / np.pi) 
        if abs_yaw <= 30: yaw_bins['[0, 30]'].append(nme)
        elif abs_yaw <= 60: yaw_bins['[30, 60]'].append(nme)
        else: yaw_bins['[60, 90]'].append(nme)
            
    categories = list(yaw_bins.keys())
    means = [np.mean(yaw_bins[cat]) * 100 if yaw_bins[cat] else 0 for cat in categories]
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(categories, means, color=['#2ecc71', '#3498db', '#9b59b6'], width=0.5)
    plt.title('NME across Different Yaw Angles', fontsize=14)
    plt.xlabel('Yaw Angle (Degrees)', fontsize=12)
    plt.ylabel('Average NME (%)', fontsize=12)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.05, f'{yval:.2f}%', ha='center', va='bottom', fontweight='bold')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[*] Đã lưu biểu đồ NME by Yaw tại: {save_path}")



def sample_texture(face_shape, image, preds, view_idx=0):
    # Đảm bảo ảnh là PIL RGB và lấy pixel dạng numpy uint8 [0-255]
    img_np = np.array(image.convert('RGB'))
    h, w, _ = img_np.shape

    # Xác định vị trí Pose của góc nhìn tương ứng (0=Front, 1=Left, 2=Right)
    start_idx = 228 + (view_idx * 7)
    pose_slice = preds[start_idx : start_idx + 7]

    # Chuyển đổi tham số dự đoán thành ma trận xoay, tịnh tiến
    R, t2d, s = preds_to_pose(pose_slice)

    # Chiếu đỉnh 3D lên mặt phẳng 2D
    projected = np.matmul(face_shape, s * R[:2].T) + t2d
    
    # Đảo trục Y để khớp với hệ tọa độ ảnh (gốc tọa độ trên-trái)
    projected[:, 1] = 224 - projected[:, 1]

    # Lấy tọa độ pixel gần nhất
    coords = np.round(projected).astype(np.int32)
    
    # Cắt (clip) tọa độ để không vượt quá kích thước ảnh 224x224
    coords[:, 0] = np.clip(coords[:, 0], 0, w - 1)
    coords[:, 1] = np.clip(coords[:, 1], 0, h - 1)

    # Lấy màu RGB tại các tọa độ đã chiếu
    colors = img_np[coords[:, 1], coords[:, 0]]
    
    return colors.astype(np.uint8)
