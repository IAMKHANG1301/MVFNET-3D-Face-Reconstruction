import torch  
import argparse  
import os 
import time  
from PIL import Image 
import torchvision.transforms as transforms 
from model import VggEncoder  
import tools 

def main():
    # Nhận các tham số đầu vào như đường dẫn ảnh và nơi lưu
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True, help='Path to directory containing front.jpg, left.jpg, right.jpg')
    parser.add_argument('--save_dir', type=str, default='./result', help='Output directory')
    # Thêm lựa chọn dán màu cho lưới 3D
    parser.add_argument('--textured', action='store_true', help='Enable texture sampling for the output mesh')
    args = parser.parse_args()

    # Thiết lập đường dẫn thư mục cho quá trình kiểm tra ảnh
    debug_dir = os.path.join(args.save_dir, 'debug')
    crop_dir = os.path.join(debug_dir, '1_cropped')
    enhance_dir = os.path.join(debug_dir, '2_enhanced')
    
    # Tạo tất cả thư mục cần thiết nếu chưa có
    for d in [args.save_dir, crop_dir, enhance_dir]:
        if not os.path.exists(d):
            os.makedirs(d)

    # Khởi tạo mô hình và chạy trên GPU
    print("Loading MVFNet model...")
    model = VggEncoder()
    model = torch.nn.DataParallel(model).cuda()
    # Tải trọng số huấn luyện tốt nhất cho mô hình
    model.load_state_dict(torch.load('data/net.pth'))
    model.eval()

    # Danh sách 3 tên file cần thiết cho mô hình đa góc nhìn
    views = ['front.jpg', 'left.jpg', 'right.jpg']
    processed_tensors = []
    final_pil_imgs = []

    print("Starting image preprocessing pipeline...")

    # Duyệt qua từng ảnh góc nhìn (trực diện, trái, phải)
    for view_name in views:
        view_key = view_name.split('.')[0]
        img_path = os.path.join(args.image_path, view_name)
        
        # Kiểm tra file ảnh có tồn tại hay không
        if not os.path.exists(img_path):
            print(f"  [!] Skipping: {img_path} not found.")
            continue

        # Mở ảnh và chuyển sang RGB
        img = Image.open(img_path).convert('RGB')
        
        # Cắt khuôn mặt chuẩn xác bằng YOLOv11
        img_cropped = tools.crop_image_yolo(img)
        # Lưu ảnh sau khi cắt vào thư mục debug
        img_cropped.save(os.path.join(crop_dir, f"{view_key}_crop.jpg"))
        print(f"  [+] Saved Crop Checkpoint: {view_key}")

        # Làm nét ảnh khuôn mặt bằng GFPGAN
        img_enhanced = tools.enhance_face(img_cropped)
        # Lưu ảnh đã làm nét vào thư mục debug
        img_enhanced.save(os.path.join(enhance_dir, f"{view_key}_enhanced.jpg"))
        print(f"  [+] Saved Enhancement Checkpoint: {view_key}")
        
        # Lưu ảnh PIL cuối cùng để dán texture và tensor để tính toán
        final_pil_imgs.append(img_enhanced)
        processed_tensors.append(transforms.functional.to_tensor(img_enhanced))

    # Yêu cầu bắt buộc phải có đủ 3 ảnh mới chạy được MVFNet
    if len(processed_tensors) < 3:
        print("Error: 3 views (front, left, right) are required for MVFNet.")
        return

    # Kết hợp 3 ảnh thành một khối dữ liệu (batch) để đưa vào mô hình
    print("Running MVFNet inference...")
    input_tensor = torch.cat(processed_tensors, 0).view(1, 9, 224, 224).cuda()
    
    # Dự đoán hình dạng 3D từ các ảnh đa góc nhìn
    with torch.no_grad():
        start_time = time.time()
        preds = model(input_tensor)
        print(f"MVFNet processing finished in: {time.time() - start_time:.4f}s")

    # Xử lý kết quả dự đoán thành định dạng Mesh
    preds_np = preds[0].detach().cpu().numpy()
    # Trích xuất đỉnh, mặt và các điểm mốc (landmarks)
    vertices, triangles, kptA, kptB, kptC = tools.preds_to_shape(preds_np)
    
    # Sử dụng bộ lọc Laplacian để tinh chỉnh bề mặt Mesh
    vertices = tools.refine_mesh(vertices, triangles)

    # Thiết lập tùy chọn lưu file có màu hoặc không màu
    colors = None
    output_filename = 'mesh_model.ply'
    
    # Thực hiện lấy mẫu màu từ cả 3 ảnh nếu được yêu cầu
    if args.textured:
        print("Sampling texture fusion from images...")
        # Kết hợp màu sắc từ các góc nhìn khác nhau
        colors = tools.sample_texture_fusion(vertices, triangles, final_pil_imgs, preds_np)
        output_filename = 'textured_model.ply'

    # Lưu file kết quả .ply để xem trên các phần mềm 3D
    save_path = os.path.join(args.save_dir, output_filename)
    tools.write_ply(save_path, vertices, triangles, colors=colors)
    
    # Thông báo hoàn tất quy trình
    print(f"\nSuccess! 3D file saved at: {save_path}")
    print(f"Debug images available in: {debug_dir}")

if __name__ == '__main__':
    main()