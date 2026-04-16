import torch
import os
import glob
import scipy.io as sio
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from model import VggEncoder
import tools
import time
import cv2  
import io
import gc
import face_alignment 

# ==========================================
# PATH CONFIGURATION
# ==========================================
AFLW_DIR = './data/AFLW2000'
MODEL_PATH = './data/net.pth'
NUM_VISUALIZE_SAMPLES = 4 
# ==========================================

def draw_landmarks_on_image(img_pil, pred_pts, gt_pts):
    """Vẽ Landmark dự đoán (Xanh) và GT (Vàng) lên ảnh"""
    img_cv2 = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    for pt in gt_pts:
        cv2.circle(img_cv2, (int(pt[0]), int(pt[1])), 2, (0, 255, 255), -1)
    for pt in pred_pts:
        cv2.circle(img_cv2, (int(pt[0]), int(pt[1])), 2, (255, 0, 0), -1)
    return Image.fromarray(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))

def render_3d_mesh_to_image(vertices, triangles, size=(224, 224), azim=0):
    """Render 3D mesh in Clay Gray style with corrected orientation"""
    fig = plt.figure(figsize=(3, 3), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    
    # FIXING COORDINATES: 
    # X stays horizontal. 
    # Depth goes to Y axis. 
    # -Y from vertices goes to Z axis (to make face stand upright)
    X = vertices[:, 0]        
    Y = vertices[:, 2]        
    Z = -vertices[:, 1]       
    
    ax.plot_trisurf(X, Y, Z, triangles=triangles, color='#bdc3c7', edgecolor='none')
        
    # CAMERA SETTINGS: 
    # elev=0 (eye level). azim matches the requested viewing angle.
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
    """AI-based face cropping following the paper's logic"""
    pts = fa_model.get_landmarks(np.array(image))
    if pts is None or len(pts) == 0:
        return None, None, None, None 
        
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
    old_bb_clip = np.array([max(0, old_bb[0]), max(0, old_bb[1]),
                            min(h, old_bb[2]), min(w, old_bb[3])]).astype(np.int32)
    
    hb, wb = old_bb_clip[2] - old_bb_clip[0], old_bb_clip[3] - old_bb_clip[1]
    new_bb = np.array([max(0, shift_y), max(0, shift_x), max(0, shift_y) + hb, max(0, shift_x) + wb]).astype(np.int32)

    cache = image.crop((old_bb_clip[1], old_bb_clip[0], old_bb_clip[3], old_bb_clip[2]))
    crop_img.paste(cache, (new_bb[1], new_bb[0], new_bb[3], new_bb[2]))
    return crop_img.resize((res, res), Image.BICUBIC), old_bb[1], old_bb[0], s

def main():
    # 1. Initialize Model
    model = VggEncoder()
    model = torch.nn.DataParallel(model).cuda()
    
    print(f"[*] Loading pretrained weights from: {MODEL_PATH}")
    ckpt = torch.load(MODEL_PATH, weights_only=True)
    model.load_state_dict(ckpt)
    model.eval()

    print(f"[*] Initializing FaceAlignment for preprocessing...")
    import warnings
    warnings.filterwarnings("ignore") 
    fa_model = face_alignment.FaceAlignment(face_alignment.LandmarksType.THREE_D, flip_input=False)

    all_nme, all_yaw, all_viz_info = [], [], []

    img_paths = glob.glob(os.path.join(AFLW_DIR, '*.jpg'))
    if not img_paths:
        print(f"[!] Error: No images found in {AFLW_DIR}")
        return
        
    print(f"[*] Evaluation started on {len(img_paths)} images...")
    start_time = time.time()

    for i, img_path in enumerate(img_paths):
        mat_path = img_path.replace('.jpg', '.mat')
        if not os.path.exists(mat_path): continue

        mat_data = sio.loadmat(mat_path)
        gt_landmarks = mat_data['pt3d_68'][:2, :].T 
        yaw_angle = mat_data['Pose_Para'][0][1]

        img = Image.open(img_path).convert('RGB')
        
        # 2. Preprocessing
        img_crop, left_x, top_y, s = crop_image_like_paper(img, fa_model)
        if img_crop is None: continue

        img_tensor = transforms.functional.to_tensor(img_crop)
        input_tensor = torch.cat([img_tensor, img_tensor, img_tensor], 0).view(1, 9, 224, 224).cuda()

        # 3. Inference
        with torch.no_grad():
            preds = model(input_tensor)

        preds_numpy = preds[0].cpu().numpy()
        faces3d = tools.preds_to_shape(preds_numpy)
        pred_landmarks = faces3d[2][:, :2] 

        # 4. Inverse mapping for NME calculation
        pred_scaled = pred_landmarks * (s / 224.0)
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
            print(f"  -> Processed {i + 1}/{len(img_paths)}... Current Mean NME: {np.mean(all_nme)*100:.2f}%")

    print(f"\n[COMPLETED] Total time: {time.time() - start_time:.2f}s")
    print(f"[RESULTS] Overall Mean NME: {np.mean(all_nme) * 100:.2f}%")
    
    os.makedirs('result/AFLW2000', exist_ok=True)
    tools.plot_ced_curve(all_nme, 'result/AFLW2000/ced_curve.png')
    tools.plot_error_by_yaw(all_nme, all_yaw, 'result/AFLW2000/error_by_yaw.png')

    # --- VISUALIZATION: 4 ROWS (Best-Worst) x 4 COLUMNS (Orig + 3D Angles) ---
    print(f"[*] Generating final multi-angle visualization report...")
    
    all_viz_info.sort(key=lambda x: x['nme'])
    samples = all_viz_info[:NUM_VISUALIZE_SAMPLES // 2] + all_viz_info[-NUM_VISUALIZE_SAMPLES // 2:]
    
    row_images = []
    label_bg = Image.new('RGB', (224, 40), color='#f0f0f0')
    
    for sample in samples:
        img_render = Image.open(sample['path']).convert('RGB')
        faces3d_render = tools.preds_to_shape(sample['preds_vector'])
        
        # Original Image with Landmarks
        img_with_pts = draw_landmarks_on_image(img_render, sample['pred_pts'], sample['gt_pts'])
        img_with_pts = img_with_pts.resize((224, 224), Image.Resampling.LANCZOS)
        
        # CAMERA TRACKING: Fix chaotic rotation
        # In Matplotlib 3D, azim=-90 is straight frontal when Y is depth.
        yaw_deg = sample['yaw'] * 180.0 / np.pi
        azim_front = -90 + yaw_deg
        azim_left = azim_front - 60   # Look from left side
        azim_right = azim_front + 60  # Look from right side
        
        # Clay Renders
        mesh_left = render_3d_mesh_to_image(faces3d_render[0], faces3d_render[1], azim=azim_left)
        mesh_front = render_3d_mesh_to_image(faces3d_render[0], faces3d_render[1], azim=azim_front)
        mesh_right = render_3d_mesh_to_image(faces3d_render[0], faces3d_render[1], azim=azim_right)
        
        # Label Info
        current_bg = label_bg.copy()
        draw = ImageDraw.Draw(current_bg)
        file_id = os.path.basename(sample['path'])[:8]
        draw.text((10, 5), f"ID: {file_id}", fill="black")
        draw.text((10, 20), f"NME: {sample['nme']*100:.2f}%", fill="#e74c3c")
        
        # Merging columns
        row = Image.new('RGB', (224 * 4, 264), color='white')
        row.paste(current_bg, (0, 0))            
        row.paste(img_with_pts, (0, 40))         
        row.paste(mesh_left, (224, 40))          
        row.paste(mesh_front, (448, 40))         
        row.paste(mesh_right, (672, 40))         
        row_images.append(row)
        
        img_render.close()
        del faces3d_render
        gc.collect()
    
    # Add Header Titles
    header = Image.new('RGB', (224 * 4, 40), color='#f0f0f0')
    draw = ImageDraw.Draw(header)
    draw.text((80, 10), "ORIGINAL", fill="black")
    draw.text((224 + 70, 10), "LEFT VIEW", fill="black")
    draw.text((448 + 80, 10), "FRONTAL", fill="black")
    draw.text((672 + 70, 10), "RIGHT VIEW", fill="black")

    final_viz = Image.new('RGB', (224 * 4, 40 + (264 * len(row_images))), color='white')
    final_viz.paste(header, (0, 0))
    y_offset = 40
    for row_img in row_images:
        final_viz.paste(row_img, (0, y_offset))
        y_offset += 264
        
    final_viz.save('result/AFLW2000/visualize_multi_angle.png', dpi=(300, 300))
    print(f"[*] Multi-angle report successfully saved to result/AFLW2000/")

if __name__ == '__main__':
    main()