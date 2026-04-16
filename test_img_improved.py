import torch  # Nhập thư viện PyTorch
import argparse  # Nhận tham số dòng lệnh
import os  # Quản lý file
import time  # Đo thời gian
from PIL import Image  # Xử lý ảnh
import torchvision.transforms as transforms  # Chuyển đổi dữ liệu ảnh
from model import VggEncoder  # Mô hình MVFNet
import tools  # Công cụ hỗ trợ

def main():
    # Nhận tham số đầu vào
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True, help='Path to images')
    parser.add_argument('--save_dir', type=str, default='./result', help='Output directory')
    parser.add_argument('--textured', action='store_true', help='Enable texture for final mesh')
    args = parser.parse_args()

    # Chỉ tạo thư mục lưu kết quả chính
    os.makedirs(args.save_dir, exist_ok=True)

    # Tải mô hình MVFNet
    print("Loading MVFNet model...")
    model = VggEncoder()
    model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(torch.load('data/net.pth', weights_only=True))
    model.eval()

    views = ['front.jpg', 'left.jpg', 'right.jpg']
    processed_tensors = []
    final_pil_imgs = []

    print("Processing images...")

    # Tiền xử lý nhanh không lưu checkpoint
    for view_name in views:
        img_path = os.path.join(args.image_path, view_name)
        if not os.path.exists(img_path): continue

        img = Image.open(img_path).convert('RGB')
        
        # Thực hiện Crop và Enhance trực tiếp trong bộ nhớ
        img_cropped, *coords = tools.crop_image_yolo(img)
        img_enhanced = tools.enhance_face(img_cropped)
        
        final_pil_imgs.append(img_enhanced)
        processed_tensors.append(transforms.functional.to_tensor(img_enhanced))

    if len(processed_tensors) < 3:
        print("Error: Required 3 images (front, left, right).")
        return

    # Thực hiện dự đoán hình dạng 3D
    print("Running MVFNet inference...")
    input_tensor = torch.cat(processed_tensors, 0).view(1, 9, 224, 224).cuda()
    with torch.no_grad():
        preds = model(input_tensor)

    # Xử lý Mesh và làm mượt
    preds_np = preds[0].detach().cpu().numpy()
    vertices, triangles, *landmarks = tools.preds_to_shape(preds_np)
    vertices = tools.refine_mesh(vertices, triangles)

    # Thiết lập file đầu ra tùy theo chế độ màu sắc
    colors = None
    output_filename = 'mesh_only.ply'
    
    if args.textured:
        print("Sampling texture...")
        colors = tools.sample_texture_fusion(vertices, triangles, final_pil_imgs, preds_np)
        output_filename = 'textured_result.ply'

    # Lưu kết quả cuối cùng duy nhất
    save_path = os.path.join(args.save_dir, output_filename)
    tools.write_ply(save_path, vertices, triangles, colors=colors)
    
    print(f"\nSuccess! Final 3D model saved at: {save_path}")

if __name__ == '__main__':
    main()