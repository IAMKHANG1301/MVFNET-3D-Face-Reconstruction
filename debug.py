import torch
import argparse
import os
import time
from PIL import Image
import torchvision.transforms as transforms
from model import VggEncoder # Kiến trúc mạng chính của dự án
import tools # Bộ công cụ xử lý YOLO, GFPGAN và Mesh 3D

def main():
    # Khởi tạo bộ nhận tham số dòng lệnh
    parser = argparse.ArgumentParser(description="Full Debug Pipeline with Automatic Texturing")
    parser.add_argument('--image_path', type=str, required=True, help='Path to directory containing front.jpg, left.jpg, right.jpg')
    parser.add_argument('--save_dir', type=str, default='./result_debug', help='Directory to save all checkpoints')
    args = parser.parse_args()

    # Thiết lập 5 thư mục lưu trữ cho 5 giai đoạn checkpoint khác nhau
    dirs = {
        'crop': os.path.join(args.save_dir, '1_crop'),
        'enhance': os.path.join(args.save_dir, '2_enhanced'),
        'raw': os.path.join(args.save_dir, '3_raw_mesh'),
        'refined': os.path.join(args.save_dir, '4_refined_mesh'),
        'textured': os.path.join(args.save_dir, '5_textured_mesh')
    }
    
    # Tạo các thư mục vật lý trên đĩa cứng
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)

    # Tải mô hình MVFNet lên bộ nhớ GPU
    print("Initializing model for full debug...")
    model = VggEncoder()
    model = torch.nn.DataParallel(model).cuda()
    # Tải trọng số với tùy chọn bảo mật weights_only=True
    model.load_state_dict(torch.load('data/mvfnet_finetuned_best.pth', weights_only=True))
    model.eval()

    # Danh sách các tệp tin ảnh đầu vào cần thiết
    views = ['front.jpg', 'left.jpg', 'right.jpg']
    processed_tensors = []
    final_pil_imgs = []

    print("--- STARTING STEP-BY-STEP DEBUG ---")

    # Duyệt qua từng góc nhìn để thực hiện tiền xử lý
    for view_name in views:
        view_key = view_name.split('.')[0]
        img_path = os.path.join(args.image_path, view_name)
        if not os.path.exists(img_path):
            print(f"  [!] Missing view: {view_name}")
            continue

        # Mở ảnh và chuyển sang hệ màu RGB
        img = Image.open(img_path).convert('RGB')
        
        # Checkpoint 1: Cắt ảnh khuôn mặt bằng YOLOv11
        # Sử dụng dấu * để gom các tọa độ (left, top, size) không dùng tới
        img_cropped, *coords = tools.crop_image_yolo(img)
        if img_cropped is None:
            print(f"  [!] Face detection failed for {view_name}")
            continue
        img_cropped.save(os.path.join(dirs['crop'], f"{view_key}_crop.jpg"))
        print(f"  [+] Checkpoint 1 (Crop) saved for: {view_key}")

        # Checkpoint 2: Phục hồi chi tiết khuôn mặt bằng GFPGAN
        img_enhanced = tools.enhance_face(img_cropped)
        img_enhanced.save(os.path.join(dirs['enhance'], f"{view_key}_enhanced.jpg"))
        print(f"  [+] Checkpoint 2 (Enhance) saved for: {view_key}")
        
        # Lưu trữ ảnh phục vụ cho bước dán texture và tensor cho mô hình
        final_pil_imgs.append(img_enhanced)
        processed_tensors.append(transforms.functional.to_tensor(img_enhanced))

    # Kiểm tra điều kiện đủ 3 ảnh trước khi chạy mô hình 3D
    if len(processed_tensors) < 3:
        print("Error: Debugging requires all 3 views (front, left, right).")
        return

    # Chạy quy trình dự đoán tham số 3DMM
    print("Running 3D inference...")
    input_tensor = torch.cat(processed_tensors, 0).view(1, 9, 224, 224).cuda()
    with torch.no_grad():
        preds = model(input_tensor)
    
    # Chuyển kết quả sang định dạng numpy để xử lý hình học
    preds_np = preds[0].detach().cpu().numpy()
    # Trích xuất dữ liệu đỉnh (vertices) và mặt (triangles) thô
    vertices, triangles, kptA, kptB, kptC = tools.preds_to_shape(preds_np)

    # Checkpoint 3: Lưu Mesh thô chưa qua xử lý làm mịn
    tools.write_ply(os.path.join(dirs['raw'], 'raw_mesh.ply'), vertices, triangles)
    print("  [+] Checkpoint 3 (Raw Mesh) saved.")

    # Checkpoint 4: Làm mượt Mesh bằng thuật toán Laplacian
    vertices_refined = tools.refine_mesh(vertices, triangles)
    tools.write_ply(os.path.join(dirs['refined'], 'refined_mesh.ply'), vertices_refined, triangles)
    print("  [+] Checkpoint 4 (Refined Mesh) saved.")

    # Checkpoint 5: Lấy mẫu màu và lưu Mesh hoàn chỉnh (Luôn thực hiện)
    print("Applying automatic texture sampling...")
    # Kết hợp màu sắc từ 3 góc nhìn đã làm nét
    colors = tools.sample_texture_fusion(vertices_refined, triangles, final_pil_imgs, preds_np)
    # Xuất file .ply cuối cùng có đầy đủ màu sắc
    tools.write_ply(os.path.join(dirs['textured'], 'textured_mesh.ply'), vertices_refined, triangles, colors=colors)
    print("  [+] Checkpoint 5 (Textured Mesh) saved.")

    print(f"\nAll debug checkpoints are successfully stored in: {args.save_dir}")

if __name__ == '__main__':
    main()