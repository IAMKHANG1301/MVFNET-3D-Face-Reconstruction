import torch
import argparse
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
from collections import defaultdict

def render_3d_mesh_to_image(vertices, triangles, size=(224, 224), azim=0):
    # Khởi tạo khung vẽ 3D
    fig = plt.figure(figsize=(3, 3), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    
    # Hiệu chỉnh tọa độ đỉnh
    X = vertices[:, 0]         
    Y = vertices[:, 2]         
    Z = -vertices[:, 1]       
    
    # Vẽ bề mặt mesh màu xám
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

def main():
    parser = argparse.ArgumentParser(description="Multi-PIE Triplet Evaluation")
    parser.add_argument('--multi_pie_dir', type=str, default='./data/Multi-Pie', help='Path to Multi-Pie dataset')
    parser.add_argument('--save_dir', type=str, default='./result/Multi-PIE', help='Directory to save results')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of subjects to process')
    parser.add_argument('--textured', action='store_true', help='Enable texture sampling for PLY files')
    args = parser.parse_args()

    # Khởi tạo mô hình
    print("Loading MVFNet model...")
    model = VggEncoder()
    model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(torch.load('data/net.pth'))
    model.eval()

    # Quét tất cả ảnh trong thư mục
    img_pattern = os.path.join(args.multi_pie_dir, '**', '*.png')
    all_files = glob.glob(img_pattern, recursive=True)
    
    # Nhóm ảnh theo Subject_Session_Expression để tìm bộ 3 camera
    # Format: 001_01_01_041_05_crop_128.png -> Key: 001_01_01
    triplets = defaultdict(dict)
    camera_map = {'130': 'left', '051': 'front', '041': 'right'}
    
    for f in all_files:
        basename = os.path.basename(f)
        parts = basename.split('_')
        if len(parts) < 4: continue
        
        # ID đối tượng và Camera ID
        subject_key = "_".join(parts[:3]) 
        cam_id = parts[3]
        
        if cam_id in camera_map:
            triplets[subject_key][camera_map[cam_id]] = f

    # Lọc ra những nhóm có đủ 3 ảnh
    valid_keys = [k for k, v in triplets.items() if len(v) == 3]
    
    if not valid_keys:
        print(f"Error: No valid triplets (130, 051, 041) found in {args.multi_pie_dir}")
        return

    # Chọn ngẫu nhiên các đối tượng
    np.random.seed(42)
    selected_keys = np.random.choice(valid_keys, min(len(valid_keys), args.num_samples), replace=False)
    
    print(f"Processing {len(selected_keys)} subjects with full triplets...")
    row_images = []

    for idx, key in enumerate(selected_keys):
        sample_id = f"subject_{key}"
        sample_dir = os.path.join(args.save_dir, sample_id)
        debug_dir = os.path.join(sample_dir, 'debug')
        for d in [sample_dir, debug_dir]:
            if not os.path.exists(d): os.makedirs(d)

        paths = triplets[key]
        processed_tensors = []
        enhanced_imgs = {}

        # Tiền xử lý lần lượt: Front, Left, Right để đưa vào MVFNet
        for view in ['front', 'left', 'right']:
            img = Image.open(paths[view]).convert('RGB')
            
            # Sử dụng YOLO để chuẩn hóa vùng cắt
            img_crop, *coords = tools.crop_image_yolo(img)
            if img_crop is None: img_crop = img.resize((224, 224))
            
            # Làm nét ảnh bằng GFPGAN
            img_enhanced = tools.enhance_face(img_crop)
            img_enhanced.save(os.path.join(debug_dir, f"{view}_enhanced.jpg"))
            
            enhanced_imgs[view] = img_enhanced
            processed_tensors.append(transforms.functional.to_tensor(img_enhanced))

        # Chạy Inference MVFNet với đầu vào 3 ảnh (9 kênh)
        input_tensor = torch.cat(processed_tensors, 0).view(1, 9, 224, 224).cuda()
        with torch.no_grad():
            preds = model(input_tensor)

        preds_np = preds[0].cpu().numpy()
        vertices, triangles, *landmarks = tools.preds_to_shape(preds_np)
        vertices = tools.refine_mesh(vertices, triangles)

        # Lưu file .ply
        colors = None
        if args.textured:
            # Lấy màu từ bộ 3 ảnh đã làm nét
            imgs_list = [enhanced_imgs['front'], enhanced_imgs['left'], enhanced_imgs['right']]
            colors = tools.sample_texture_fusion(vertices, triangles, imgs_list, preds_np)
        
        tools.write_ply(os.path.join(sample_dir, "result.ply"), vertices, triangles, colors=colors)

        # Render 3 góc nhìn 3D
        mesh_l = render_3d_mesh_to_image(vertices, triangles, azim=-60)
        mesh_f = render_3d_mesh_to_image(vertices, triangles, azim=0)
        mesh_r = render_3d_mesh_to_image(vertices, triangles, azim=60)
        
        # In ảnh báo cáo: Lấy ảnh gốc là Front
        front_orig = enhanced_imgs['front'].resize((224, 224), Image.Resampling.LANCZOS)
        row = Image.new('RGB', (224 * 4, 224), color='white')
        row.paste(front_orig, (0, 0))
        row.paste(mesh_l, (224, 0))
        row.paste(mesh_f, (448, 0))
        row.paste(mesh_r, (672, 0))
        row_images.append(row)
        gc.collect()

    # Tạo Header và lưu ảnh tổng hợp
    header = Image.new('RGB', (224 * 4, 40), color='#f0f0f0')
    draw = ImageDraw.Draw(header)
    draw.text((60, 10), "ORIGINAL (FRONT-ENHANCED)", fill="black")
    draw.text((224 + 70, 10), "3D LEFT VIEW", fill="black")
    draw.text((448 + 80, 10), "3D FRONTAL", fill="black")
    draw.text((672 + 70, 10), "3D RIGHT VIEW", fill="black")

    final_viz = Image.new('RGB', (224 * 4, 40 + (224 * len(row_images))), color='white')
    final_viz.paste(header, (0, 0))
    for i, r in enumerate(row_images):
        final_viz.paste(r, (0, 40 + i * 224))
        
    final_viz.save(os.path.join(args.save_dir, 'multipie_multi_angle_viz.png'))
    print(f"\n[SUCCESS] Report saved in {args.save_dir}")

if __name__ == '__main__':
    main()