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

# Thiết lập đường dẫn dữ liệu và trọng số
AFLW_DIR = './data/AFLW2000'
MODEL_PATH = './data/net.pth'

def main():
    # Khởi tạo mô hình mạng mạng VggEncoder
    model = VggEncoder()
    # Sử dụng song song GPU để tăng tốc độ tính toán
    model = torch.nn.DataParallel(model).cuda()
    
    # Tải trọng số đã huấn luyện vào mô hình
    print(f"Loading pretrained weights: {MODEL_PATH}")
    ckpt = torch.load(MODEL_PATH, weights_only=True)
    model.load_state_dict(ckpt)
    # Thiết lập chế độ đánh giá cho mô hình
    model.eval()

    # Khởi tạo các danh sách lưu trữ kết quả đánh giá
    all_nme, all_yaw = [], []

    # Lấy danh sách tất cả file ảnh trong thư mục AFLW2000
    img_paths = glob.glob(os.path.join(AFLW_DIR, '*.jpg'))
    
    print(f"Starting NME-51 evaluation on {len(img_paths)} images...")
    start_time = time.time()

    # Duyệt qua từng ảnh trong tập dữ liệu
    for i, img_path in enumerate(img_paths):
        # Xác định đường dẫn file nhãn tương ứng
        mat_path = img_path.replace('.jpg', '.mat')
        if not os.path.exists(mat_path): continue

        # Đọc dữ liệu tọa độ thực tế (Ground Truth) và góc quay (Yaw)
        mat_data = sio.loadmat(mat_path)
        gt_landmarks = mat_data['pt3d_68'][:2, :].T 
        yaw_angle = mat_data['Pose_Para'][0][1]

        # Mở ảnh và chuyển về định dạng màu RGB
        img = Image.open(img_path).convert('RGB')
        
        # Bước 1: Tiền xử lý ảnh bằng bộ phát hiện YOLOv11
        img_crop, left_x, top_y, s = tools.crop_image_yolo(img)
        if img_crop is None: continue

        # Bước 2: Làm nét ảnh bằng GFPGAN để tăng độ chính xác landmark
        img_enhanced = tools.enhance_face(img_crop)

        # Bước 3: Đưa ảnh qua mô hình MVFNet để dự đoán tham số 3D
        img_tensor = transforms.functional.to_tensor(img_enhanced)
        # Nhân bản ảnh trực diện thành 3 view theo yêu cầu đầu vào của mạng
        input_tensor = torch.cat([img_tensor, img_tensor, img_tensor], 0).view(1, 9, 224, 224).cuda()

        with torch.no_grad():
            preds = model(input_tensor)

        # Trích xuất Landmark từ kết quả dự đoán của mô hình
        preds_numpy = preds[0].cpu().numpy()
        # Lấy tọa độ landmark 3D chiếu lên mặt phẳng 2D
        res_3d = tools.preds_to_shape(preds_numpy)
        pred_landmarks = res_3d[2][:, :2] 

        # Bước 4: Ánh xạ ngược tọa độ từ ảnh 224x224 về ảnh gốc
        pred_scaled = pred_landmarks * (s / 224.0)
        pred_scaled[:, 0] += left_x 
        pred_scaled[:, 1] += top_y  

        # Bước 5: Tính toán sai số NME dựa trên 51 điểm nội quan
        # Hàm calculate_nme đã được sửa để bỏ qua 17 điểm viền hàm
        nme = tools.calculate_nme(pred_scaled, gt_landmarks)
        all_nme.append(nme)
        all_yaw.append(yaw_angle)

        img.close()
        # In tiến độ xử lý định kỳ mỗi 100 ảnh
        if (i + 1) % 100 == 0:
            gc.collect()
            print(f"  -> Processed {i + 1}/{len(img_paths)} | Current NME-51: {np.mean(all_nme)*100:.2f}%")

    # In kết quả tổng kết cuối cùng
    print(f"\n[COMPLETED] Total evaluation time: {time.time() - start_time:.2f}s")
    print(f"[RESULTS] Final Overall Mean NME-51: {np.mean(all_nme) * 100:.2f}%")
    
    # Tạo thư mục và vẽ các biểu đồ thống kê sai số
    os.makedirs('result/AFLW2000', exist_ok=True)
    # Vẽ đường cong CED cho 51 điểm nội quan
    tools.plot_ced_curve(all_nme, 'result/AFLW2000/nme51_ced_curve.png')
    # Vẽ biểu đồ sai số theo góc quay (Yaw)
    tools.plot_error_by_yaw(all_nme, all_yaw, 'result/AFLW2000/nme51_error_by_yaw.png')

    print(f"[*] Statistical charts saved to result/AFLW2000/")

if __name__ == '__main__':
    main()