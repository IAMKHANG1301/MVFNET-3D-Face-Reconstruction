==========================================================================
MVF-Net: Dự Hồi Quy Mô Hình Khuôn Mặt 3D Biến Dạng Đa Góc Nhìn
==========================================================================

Đây là mã nguồn thực nghiệm cho mô hình MVF-Net, được giới thiệu trong bài báo:
MVF-Net: Multi-View 3D Face Morphable Model Regression
Tác giả: Fanzi Wu, Linchao Bao, Yajing Chen, Yonggen Ling, Yibing Song, Songnan Li, King Ngi Ngan, Wei Liu. (CVPR 2019).

Mô hình này sử dụng mạng thần kinh tích chập (CNN) để hồi quy các tham số của Mô hình biến dạng khuôn mặt 3D (3DMM) từ ảnh đa góc nhìn bằng cách khai thác các ràng buộc hình học thông qua hàm mất mát căn chỉnh (view alignment loss).

1. YÊU CẦU HỆ THỐNG VÀ CÀI ĐẶT
--------------------------------------------------------------------------
Sử dụng công cụ quản lý gói 'uv' để đảm bảo tốc độ và tính ổn định trên Python 3.12+.

Bước 1: Cài đặt uv
pip install uv

Bước 2: Khởi tạo môi trường và cài đặt thư viện

# Tạo môi trường ảo
uv venv

# Kích hoạt môi trường ảo:
Windows: .venv\Scripts\activate
Linux/Mac: source .venv/bin/activate

# Cài đặt các thư viện phụ thuộc chính:
uv pip install -r requirement.txt

# Cài đặt bản vá BasicSR (tránh lỗi functional_tensor)
uv pip install git+https://github.com/XPixelGroup/BasicSR.git

2. CHUẨN BỊ DỮ LIỆU VÀ TRỌNG SỐ
--------------------------------------------------------------------------
Sắp xếp cấu trúc thư mục dự án như sau:

Thư mục data/:
- net.pth: Trọng số MVF-Net gốc CNN model từ [here](https://www.dropbox.com/s/7ds3aesjjmybjh9/net.pth?dl=0)
- Model_shape.mat, Model_Expression.mat, sigma_exp.mat từ [3DDFA](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm)
- AFLW2000/: Chứa ảnh và nhãn .mat của tập dữ liệu kiểm thử
Ngoài ra có thể tải full folder data mà nhóm em đã tích hợp sẵn từ đây: [here](https://drive.google.com/drive/folders/1ltjT1y9COBHCGQWTYLqnFznJ7Nf4J9EQ?usp=sharing)

Thư mục weights/:
- yolo11n-face.pt: Trọng số cho bộ phát hiện khuôn mặt YOLOv11 [here](https://huggingface.co/deepghs/yolo-face/resolve/b6b06ab1a58eba921209ee6431a79ffedc498eb1/yolov11n-face/model.pt)
- GFPGANv1.4.pth: Trọng số cho bộ làm nét khuôn mặt GFPGAN [here](https://share.google/gvt4fe8bAmTQ9znSk)

3. HƯỚNG DẪN KIỂM THỬ (INFERENCE)
--------------------------------------------------------------------------

3.1. Kiểm thử trên ảnh đơn Đa góc nhìn (Multi-view)
Sử dụng 3 ảnh (front, left, right) để dựng mô hình 3D hoàn chỉnh.

# Mesh thô (không màu):
uv run python test_img_improved.py --image_path ./data/imgs --save_dir ./result/mesh_only

# Mesh có Texture:
uv run python test_img_improved.py --image_path ./data/imgs --save_dir ./result/textured --textured

3.2. Kiểm thử trên ảnh nhóm (Group Photo)
Tự động phát hiện và dựng 3D cho từng người trong một tấm ảnh duy nhất.

uv run python process_group_final.py --input ./group.jpg --out ./result_group --textured

3.3. Đánh giá định lượng trên AFLW2000-3D
Đo lường sai số NME dựa trên 51 điểm landmark nội quan (Inner Landmarks) để đạt độ khách quan khoa học cao nhất.

uv run python test_aflw2000.py

Kết quả: Biểu đồ đường cong sai số tích lũy (CED Curve) và biểu đồ NME theo góc xoay (Yaw) được lưu tại result/AFLW2000/.

4. CÁC CẢI TIẾN KỸ THUẬT TÍCH HỢP
--------------------------------------------------------------------------
- YOLOv11-Face Detection: Tăng tốc độ phát hiện khuôn mặt đa đối tượng và ổn định ở góc nghiêng lớn.
- GFPGAN Enhancement: Làm nét khuôn mặt trước khi đưa vào MVF-Net, giúp giảm sai số NME.
- Laplacian Mesh Refinement: Làm mượt bề mặt lưới, giảm nhiễu và răng cưa.
- Textured Sampling (--textured): Hòa trộn màu sắc để tạo file .ply chân thực.

5. TRÍCH DẪN
--------------------------------------------------------------------------
@inproceedings{wu2019mvf,
  title={MVF-Net: Multi-View 3D Face Morphable Model Regression},
  author={Wu, Fanzi and Bao, Linchao and Chen, Yajing and Ling, Yonggen and Song, Yibing and Li, Songnan and Ngan, King Ngi Ngan and Liu, Wei},
  booktitle={CVPR},
  year={2019}
}

==========================================================================