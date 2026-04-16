import torch  
import os  
import numpy as np 
from PIL import Image  
import torchvision.transforms as transforms  
import tools 
from model import VggEncoder  
import argparse  
import time  

def process_group(input_path, out_dir, model_path, use_texture=False):
    # Kiểm tra và sử dụng GPU nếu có, nếu không thì dùng CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Khởi tạo kiến trúc mô hình VggEncoder
    print("Initializing MVFNet model...")
    model = VggEncoder()
    # Chạy mô hình trên nhiều GPU song song nếu có thể
    model = torch.nn.DataParallel(model).to(device)
    # Tải trọng số đã huấn luyện vào mô hình
    model.load_state_dict(torch.load(model_path, map_location=device))
    # Chuyển mô hình sang chế độ đánh giá (không huấn luyện)
    model.eval()

    # Tạo thư mục lưu kết quả nếu chưa tồn tại
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Đọc ảnh gốc và chuyển sang định dạng màu RGB
    print("Detecting faces in input image...")
    full_img = Image.open(input_path).convert('RGB')
    # Chuyển ảnh sang dạng mảng numpy để YOLO xử lý
    img_array = np.array(full_img)

    # Sử dụng YOLOv11-Face để tìm kiếm các khuôn mặt trong ảnh
    results = tools.face_detector(img_array, conf=0.5, verbose=False)
    # Trích xuất tọa độ các khung bao (bounding boxes)
    boxes = results[0].boxes.xyxy.cpu().numpy()

    # Thông báo lỗi nếu không tìm thấy ai
    if len(boxes) == 0:
        print("Error: No faces detected in the image.")
        return

    # In ra số lượng người tìm thấy và trạng thái chế độ dán màu
    print(f"Found {len(boxes)} person(s). Texture mode: {'ON' if use_texture else 'OFF'}")

    # Duyệt qua từng khuôn mặt được phát hiện
    for i, box in enumerate(boxes):
        person_id = f"person_{i+1}"
        # Thiết lập đường dẫn thư mục cho từng cá nhân
        person_dir = os.path.join(out_dir, person_id)
        debug_dir = os.path.join(person_dir, 'debug')
        # Tạo cây thư mục lưu trữ ảnh trung gian
        for d in [person_dir, debug_dir]:
            if not os.path.exists(d): os.makedirs(d)

        # Lấy tọa độ khung bao của người hiện tại
        print(f"[{person_id}] Cropping and enhancing face...")
        x1, y1, x2, y2 = box
        # Tính toán tâm và kích thước để cắt khuôn mặt
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
        # Sử dụng hệ số 1.5 để đảm bảo lấy đủ phần đầu cho mô hình
        size = int(max(x2 - x1, y2 - y1) * 1.5)
        
        # Cắt và thay đổi kích thước ảnh về chuẩn 224x224
        face_crop = full_img.crop((int(center_x - size/2), int(center_y - size/2), 
                                   int(center_x + size/2), int(center_y + size/2))).resize((224, 224), Image.BICUBIC)
        # Lưu ảnh đã cắt để kiểm tra (checkpoint 1)
        face_crop.save(os.path.join(debug_dir, "1_crop.jpg"))

        # Sử dụng GFPGAN để làm nét các chi tiết trên khuôn mặt
        face_enhanced = tools.enhance_face(face_crop)
        # Lưu ảnh đã làm nét (checkpoint 2)
        face_enhanced.save(os.path.join(debug_dir, "2_enhanced.jpg"))

        # Chuyển ảnh sang dạng tensor để đưa vào mô hình
        print(f"[{person_id}] Running 3D reconstruction...")
        img_tensor = transforms.functional.to_tensor(face_enhanced).to(device)
        # Tạo tensor đầu vào gồm 3 ảnh giống nhau cho 3 góc nhìn (Front/Left/Right)
        input_tensor = torch.cat([img_tensor, img_tensor, img_tensor], 0).unsqueeze(0)
        
        # Thực hiện dự đoán các tham số 3D mà không tính đạo hàm
        with torch.no_grad():
            preds = model(input_tensor)
        
        # Chuyển kết quả sang định dạng numpy để xử lý Mesh
        preds_np = preds[0].detach().cpu().numpy()
        # Tính toán tọa độ đỉnh (vertices) và các mặt (triangles)
        vertices, triangles, *others = tools.preds_to_shape(preds_np)
        
        # Làm mượt bề mặt lưới 3D để giảm răng cưa
        vertices = tools.refine_mesh(vertices, triangles)

        # Khởi tạo mặc định không có màu và tên file không màu
        colors = None
        ply_filename = f"mesh_{person_id}.ply"
        
        # Nếu người dùng bật chế độ dán màu
        if use_texture:
            print(f"[{person_id}] Sampling vertex colors...")
            # Tạo danh sách ảnh đầu vào phục vụ việc lấy mẫu màu
            enhanced_list = [face_enhanced, face_enhanced, face_enhanced]
            # Lấy màu từ ảnh dán lên các đỉnh của mô hình 3D
            colors = tools.sample_texture_fusion(vertices, triangles, enhanced_list, preds_np)
            ply_filename = f"textured_{person_id}.ply"

        # Lưu kết quả cuối cùng dưới định dạng .ply
        save_path = os.path.join(person_dir, ply_filename)
        tools.write_ply(save_path, vertices, triangles, colors=colors)
        
        print(f"[{person_id}] Success! File saved at: {save_path}")

if __name__ == "__main__":
    # Cài đặt bộ nhận tham số từ người dùng
    parser = argparse.ArgumentParser(description="Multi-face 3D Reconstruction Pipeline")
    parser.add_argument('--input', type=str, required=True, help='Path to the group image')
    parser.add_argument('--out', type=str, default='./result_group', help='Output directory')
    parser.add_argument('--textured', action='store_true', help='Enable texture sampling for the mesh')
    args = parser.parse_args()

    # Đường dẫn đến file trọng số của MVFNet
    model_ckpt = 'data/net.pth'
    
    # Ghi nhận thời gian bắt đầu và chạy chương trình chính
    start_time = time.time()
    process_group(args.input, args.out, model_ckpt, use_texture=args.textured)
    # Thông báo tổng thời gian hoàn thành
    print(f"\nTotal processing time: {time.time() - start_time:.2f} seconds")