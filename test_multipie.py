import torch
import os
import glob
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from model import VggEncoder
import tools
import time
import io
import gc
import face_alignment

# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN
# ==========================================
MULTI_PIE_DIR = './data/Multi-PIE' # Thư mục chứa ảnh Multi-PIE
MODEL_PATH = './data/net.pth'
NUM_SAMPLES = 4 # Số lượng người muốn vẽ
SAVE_DIR = './result/Multi-PIE'
# ==========================================

def render_3d_mesh_to_image(vertices, triangles, size=(224, 224), azim=0):
    """
    Render khối 3D MÀU XÁM TRƠN (Clay Render) kèm đổ bóng
    """
    fig = plt.figure(figsize=(3, 3), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    
    # CHUẨN HÓA TRỤC TỌA ĐỘ
    X = vertices[:, 0]        
    Y = vertices[:, 2]        
    Z = vertices[:, 1]       
    
    # Render lưới 3D màu xám nhạt kèm đổ bóng
    ax.plot_trisurf(X, Y, Z, triangles=triangles, color='#bdc3c7', edgecolor='none')
        
    # ĐIỀU KHIỂN CAMERA QUAY THEO GÓC YÊU CẦU
    ax.view_init(elev=0, azim=azim) 
    
    ax.set_box_aspect([1, 1, 1])
    ax.axis('off')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close(fig)
    buf.seek(0)
    
    img = Image.open(buf).convert('RGB')
    return img.resize(size, Image.Resampling.LANCZOS)

def crop_image_with_ai(image, fa_model, res=224):
    """Dùng AI tìm mặt, cắt ảnh và TÍNH LUÔN GÓC YAW bằng 3D Landmarks"""
    pts = fa_model.get_landmarks(np.array(image))
    if pts is None or len(pts) == 0:
        return None, None, None, None, 0.0
        
    # Lấy tọa độ 3D đầy đủ (68, 3) từ FaceAlignment
    pts_3d = np.array(pts[0])
    
    # ĐÃ FIX: Tính góc Yaw bằng mốc 3D của FaceAlignment (Chắc chắn có trục Z)
    dx = pts_3d[16, 0] - pts_3d[0, 0]
    dz = pts_3d[16, 2] - pts_3d[0, 2]
    yaw_deg = np.arctan2(dz, dx) * 180.0 / np.pi
    
    # Chuyển về 2D nguyên để cắt ảnh
    pts_2d = pts_3d.astype(np.int32)
    h, w = image.size[1], image.size[0]
    
    # Bounding Box
    x_max, x_min = np.max(pts_2d[:68, 0]), np.min(pts_2d[:68, 0])
    y_max, y_min = np.max(pts_2d[:68, 1]), np.min(pts_2d[:68, 1])
    
    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0 
    center_y = center_y + (y_max - y_min) * 0.12 
    
    size = max(x_max - x_min, y_max - y_min) * 1.5
    
    left = center_x - size / 2.0
    top = center_y - size / 2.0
    right = center_x + size / 2.0
    bottom = center_y + size / 2.0
    
    crop_img = image.crop((left, top, right, bottom))
    crop_img = crop_img.resize((res, res), Image.BICUBIC)
    
    # Trả về 5 biến: Ảnh đã cắt, tọa độ X, tọa độ Y, Scale, và Góc Yaw
    return crop_img, left, top, size, yaw_deg

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # 1. Khởi tạo mô hình MVF-Net
    model = VggEncoder()
    model = torch.nn.DataParallel(model).cuda()
    print(f"[*] Loading weights from: {MODEL_PATH}")
    ckpt = torch.load(MODEL_PATH, weights_only=True)
    model.load_state_dict(ckpt)
    model.eval()

    print(f"[*] Initializing FaceAlignment AI model to find Multi-PIE faces...")
    import warnings
    warnings.filterwarnings("ignore")
    fa_model = face_alignment.FaceAlignment(face_alignment.LandmarksType.THREE_D, flip_input=False)

    img_paths = glob.glob(os.path.join(MULTI_PIE_DIR, '**', '*.jpg'), recursive=True) + \
                glob.glob(os.path.join(MULTI_PIE_DIR, '**', '*.png'), recursive=True)
                
    if len(img_paths) == 0:
        print(f"[!] No images found at '{MULTI_PIE_DIR}'")
        return
    
    # Lấy ngẫu nhiên các ảnh để Visualize
    np.random.seed(42)
    selected_paths = np.random.choice(img_paths, NUM_SAMPLES, replace=False)
    
    print(f"[*] Starting Visualization generation for {NUM_SAMPLES} Multi-PIE images...")
    
    row_images = []
    
    for img_path in selected_paths:
        img = Image.open(img_path).convert('RGB')
        
        # Cắt ảnh và nhận trực tiếp góc Yaw
        img_crop, _, _, _, yaw_deg = crop_image_with_ai(img, fa_model)
        if img_crop is None:
            print(f"  -> Skipping image (No face detected by AI): {os.path.basename(img_path)}")
            continue

        img_tensor = transforms.functional.to_tensor(img_crop)
        
        # Kích hoạt chế độ Single-View
        input_tensor = torch.cat([img_tensor, img_tensor, img_tensor], 0).view(1, 9, 224, 224).cuda()

        with torch.no_grad():
            preds = model(input_tensor)

        preds_numpy = preds[0].cpu().numpy()
        faces3d = tools.preds_to_shape(preds_numpy)
        
        # Chỉnh lại size ảnh gốc để ghép chuẩn
        img_original_resized = img.resize((224, 224), Image.Resampling.LANCZOS)
        
        # Bù trừ góc xoay Camera (Camera Tracking)
        azim_left = 70 - yaw_deg    
        azim_front = 0 - yaw_deg    
        azim_right = -70 - yaw_deg  
        
        # Render thẳng 3 góc độ với lưới xám trơn cùng camera bù trừ
        mesh_left = render_3d_mesh_to_image(faces3d[0], faces3d[1], size=(224, 224), azim=azim_left)
        mesh_front = render_3d_mesh_to_image(faces3d[0], faces3d[1], size=(224, 224), azim=azim_front)
        mesh_right = render_3d_mesh_to_image(faces3d[0], faces3d[1], size=(224, 224), azim=azim_right)
        
        # Ghép thành 1 hàng ngang (Chiều rộng = 224 * 4)
        row = Image.new('RGB', (224 * 4, 224), color='white')
        row.paste(img_original_resized, (0, 0))
        row.paste(mesh_left, (224, 0))
        row.paste(mesh_front, (448, 0))
        row.paste(mesh_right, (672, 0))
        
        row_images.append(row)
        img.close()
        gc.collect()

    if not row_images:
        print("[!] No images processed successfully.")
        return

    # Thêm dải Tiêu đề ở trên cùng
    header = Image.new('RGB', (224 * 4, 40), color='#f0f0f0')
    draw = ImageDraw.Draw(header)
    
    draw.text((80, 10), "ORIGINAL", fill="black")
    draw.text((224 + 70, 10), "LEFT PROFILE", fill="black")
    draw.text((448 + 80, 10), "FRONTAL", fill="black")
    draw.text((672 + 70, 10), "RIGHT PROFILE", fill="black")

    # Ghép tất cả các hàng dọc lại thành 1 bức ảnh Final
    final_height = 40 + (224 * len(row_images))
    final_viz = Image.new('RGB', (224 * 4, final_height), color='white')
    
    final_viz.paste(header, (0, 0))
    y_offset = 40
    for row_img in row_images:
        final_viz.paste(row_img, (0, y_offset))
        y_offset += 224
        
    final_viz_path = os.path.join(SAVE_DIR, 'multipie_multi_angle_viz.png')
    final_viz.save(final_viz_path, dpi=(300, 300))
    print(f"\n[COMPLETED] Multi-PIE Visualization saved at: {final_viz_path}")

if __name__ == '__main__':
    main()