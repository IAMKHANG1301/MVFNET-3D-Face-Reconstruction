import torch
import os
import glob
import scipy.io as sio
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import tools
import time
import cv2  
import io
import gc
import face_alignment 
import argparse # Thêm thư viện này

# ==========================================
# CẤU HÌNH MẶC ĐỊNH
# ==========================================
AFLW_DIR = './data/AFLW2000'
NUM_VISUALIZE_SAMPLES = 4 
# ==========================================

def draw_landmarks_on_image(img_pil, pred_pts, gt_pts):
    # (Giữ nguyên hàm này của bạn)
    img_cv2 = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    for pt in gt_pts:
        cv2.circle(img_cv2, (int(pt[0]), int(pt[1])), 2, (0, 255, 255), -1)
    for pt in pred_pts:
        cv2.circle(img_cv2, (int(pt[0]), int(pt[1])), 2, (255, 0, 0), -1)
    return Image.fromarray(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))

def render_3d_mesh_to_image(vertices, triangles, size=(224, 224), azim=0):
    # (Giữ nguyên hàm này của bạn)
    fig = plt.figure(figsize=(3, 3), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    X, Y, Z = vertices[:, 0], vertices[:, 2], -vertices[:, 1]       
    ax.plot_trisurf(X, Y, Z, triangles=triangles, color='#bdc3c7', edgecolor='none')
    ax.view_init(elev=0, azim=azim) 
    ax.set_box_aspect([1, 1, 1])
    ax.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert('RGB')
    return img.resize(size, Image.Resampling.LANCZOS)

def crop_image_like_paper(image, fa_model, res=224):
    # (Giữ nguyên hàm này của bạn)
    pts = fa_model.get_landmarks(np.array(image))
    if pts is None or len(pts) == 0: return None, None, None, None 
    pts = np.array(pts[0]).astype(np.int32)
    h, w = image.size[1], image.size[0]
    x_max, x_min = np.max(pts[:68, 0]), np.min(pts[:68, 0])
    y_max, y_min = np.max(pts[:68, 1]), np.min(pts[:68, 1])
    bbox = [y_min, x_min, y_max, x_max]
    c = [bbox[2] - (bbox[2] - bbox[0]) / 2, bbox[3] - (bbox[3] - bbox[1]) / 2.0]
    c[0] = c[0] - (bbox[2] - bbox[0]) * 0.12
    s = (max(bbox[2] - bbox[0], bbox[3] - bbox[1]) * 1.5).astype(np.int32)
    old_bb = np.array([c[0] - s / 2, c[1] - s / 2, c[0] + s / 2, c[1] + s / 2]).astype(np.int32)
    crop_img = Image.new('RGB', (s, s))
    shift_x, shift_y = 0 - old_bb[1], 0 - old_bb[0]
    old_bb_clip = np.array([max(0, old_bb[0]), max(0, old_bb[1]), min(h, old_bb[2]), min(w, old_bb[3])]).astype(np.int32)
    hb, wb = old_bb_clip[2] - old_bb_clip[0], old_bb_clip[3] - old_bb_clip[1]
    new_bb = np.array([max(0, shift_y), max(0, shift_x), max(0, shift_y) + hb, max(0, shift_x) + wb]).astype(np.int32)
    cache = image.crop((old_bb_clip[1], old_bb_clip[0], old_bb_clip[3], old_bb_clip[2]))
    crop_img.paste(cache, (new_bb[1], new_bb[0], new_bb[3], new_bb[2]))
    return crop_img.resize((res, res), Image.BICUBIC), old_bb[1], old_bb[0], s

def main():
    # --- BƯỚC 0: XỬ LÝ ĐỐI SỐ DÒNG LỆNH ---
    parser = argparse.ArgumentParser(description='MVF-Net Evaluation on AFLW2000')
    parser.add_argument('--improved', action='store_true', help='Sử dụng mô hình ImprovedVggEncoder')
    parser.add_argument('--model_path', type=str, default=None, help='Đường dẫn file .pth')
    args = parser.parse_args()

    # --- BƯỚC 1: KHỞI TẠO MODEL VÀ NẠP TRỌNG SỐ ---
    if args.improved:
        from finetune_claude import ImprovedVggEncoder
        model = ImprovedVggEncoder()
        # Nếu không truyền path, lấy file checkpoint fine-tune mặc định
        model_path = args.model_path if args.model_path else './checkpoints/improved_mvfnet.pth'
        print("[*] Chế độ: IMPROVED MODEL (Attention + Multi-view Awareness)")
    else:
        from model import VggEncoder
        model = VggEncoder()
        # Nếu không truyền path, lấy file gốc của tác giả
        model_path = args.model_path if args.model_path else './data/net.pth'
        print("[*] Chế độ: ORIGINAL MODEL (Baseline)")

    print(f"[*] Loading weights from: {model_path}")
    checkpoint = torch.load(model_path, map_location='cuda')

    # Trích xuất state_dict
    if args.improved and isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Xử lý tiền tố 'module.' của DataParallel nếu có
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model = torch.nn.DataParallel(model).cuda()
    model.eval()

    # (Phần còn lại giữ nguyên logic xử lý của bạn)
    print(f"[*] Initializing FaceAlignment library...")
    import warnings
    warnings.filterwarnings("ignore") 
    fa_model = face_alignment.FaceAlignment(face_alignment.LandmarksType.THREE_D, flip_input=False)

    all_nme, all_yaw, all_viz_info = [], [], []
    img_paths = glob.glob(os.path.join(AFLW_DIR, '*.jpg'))
    if not img_paths:
        print(f"[!] Error: No images found in {AFLW_DIR}")
        return
        
    print(f"[*] Starting evaluation on {len(img_paths)} images...")
    start_time = time.time()

    for i, img_path in enumerate(img_paths):
        mat_path = img_path.replace('.jpg', '.mat')
        if not os.path.exists(mat_path): continue
        mat_data = sio.loadmat(mat_path)
        gt_landmarks = mat_data['pt3d_68'][:2, :].T 
        yaw_angle = mat_data['Pose_Para'][0][1]    

        img = Image.open(img_path).convert('RGB')
        img_crop, left_x, top_y, s = crop_image_like_paper(img, fa_model)
        if img_crop is None: continue

        img_tensor = transforms.functional.to_tensor(img_crop)
        input_tensor = torch.cat([img_tensor, img_tensor, img_tensor], 0).view(1, 9, 224, 224).cuda()

        with torch.no_grad():
            preds = model(input_tensor)
            # Nếu là Improved model, preds sẽ là tuple (out, confidences)
            if isinstance(preds, tuple):
                preds = preds[0]

        # preds_numpy = preds[0].cpu().numpy()
        # faces3d = tools.preds_to_shape(preds_numpy)
        # pred_landmarks = faces3d[2][:, :2].copy() 

        # vertices_3d = faces3d[0]
        # R, t2d, s_pose = tools.preds_to_pose(preds_numpy[228:235])
        # projected_all_vertices = np.matmul(vertices_3d, s_pose * R[:2].T) + t2d
        # projected_all_vertices[:, 1] = 224 - projected_all_vertices[:, 1]
        
        # kpt_inner = pred_landmarks[17:68]
        # dynamic_jaw = tools.dynamic_marching_jawline(projected_all_vertices, kpt_inner)
        # pred_landmarks[0:17] = dynamic_jaw

        # pred_scaled = pred_landmarks * (s / 224.0)
        # pred_scaled[:, 0] += left_x 
        # pred_scaled[:, 1] += top_y

        preds_numpy = preds[0].cpu().numpy()

        # CÁCH SỬA: Thay vì lấy từ faces3d, hãy lấy trực tiếp từ 136 giá trị đầu
        # Đây là các điểm landmark 2D (68 điểm x 2) mà Wing Loss đã tối ưu
        pred_landmarks_direct = preds_numpy[:136].reshape(68, 2)

        # Tiếp tục các bước ánh xạ ngược như cũ
        pred_scaled = pred_landmarks_direct * (s / 224.0)
        pred_scaled[:, 0] += left_x 
        pred_scaled[:, 1] += top_y

        nme = tools.calculate_nme(pred_scaled, gt_landmarks)
        all_nme.append(nme)
        all_yaw.append(yaw_angle)

        all_viz_info.append({
            'path': img_path, 'nme': nme, 'gt_pts': gt_landmarks,
            'pred_pts': pred_scaled, 'preds_vector': preds_numpy,
            'yaw': yaw_angle
        })

        img.close()
        if (i + 1) % 200 == 0:
            gc.collect() 
            print(f"  -> Processed {i + 1}/{len(img_paths)} samples... Current Mean NME: {np.mean(all_nme)*100:.2f}%")

    print(f"\n[DONE] Total runtime: {time.time() - start_time:.2f}s")
    print(f"[RESULT] Overall Mean NME: {np.mean(all_nme) * 100:.2f}%")
    
    save_dir = 'result/AFLW2000_Improved' if args.improved else 'result/AFLW2000_Original'
    os.makedirs(save_dir, exist_ok=True)
    tools.plot_ced_curve(all_nme, os.path.join(save_dir, 'ced_curve.png'))
    tools.plot_error_by_yaw(all_nme, all_yaw, os.path.join(save_dir, 'error_by_yaw.png'))

    # (Phần vẽ ảnh minh họa cuối script bạn giữ nguyên, 
    # chỉ cần đổi tên file save cuối cùng để không bị đè)
    # final_viz.save(os.path.join(save_dir, 'visualize_multi_angle.png'), dpi=(300, 300))

if __name__ == '__main__':
    main()