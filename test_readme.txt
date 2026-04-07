==========================================================================
MVF-Net: Dự Hồi Quy Mô Hình Khuôn Mặt 3D Biến Dạng Đa Góc Nhìn
==========================================================================

Đây là mã nguồn thực nghiệm cho mô hình MVF-Net, được giới thiệu trong bài báo:
MVF-Net: Multi-View 3D Face Morphable Model Regression 
Tác giả: Fanzi Wu, Linchao Bao, Yajing Chen, Yonggen Ling, Yibing Song, Songnan Li, King Ngi Ngan, Wei Liu. (CVPR 2019).

Mô hình này sử dụng mạng thần kinh tích chập (CNN) để hồi quy các tham số của Mô hình biến dạng khuôn mặt 3D (3DMM) từ ảnh đa góc nhìn bằng cách khai thác các ràng buộc hình học thông qua hàm mất mát căn chỉnh (view alignment loss)[cite: 11, 12].

1. YÊU CẦU HỆ THỐNG VÀ CÀI ĐẶT
--------------------------------------------------------------------------
Sử dụng công cụ quản lý gói 'uv' để đảm bảo tốc độ và tính ổn định trên Python 3.x.

Bước 1: Cài đặt uv
pip install uv

Bước 2: Khởi tạo môi trường và cài đặt thư viện
# Tạo môi trường ảo
uv venv

# Kích hoạt môi trường ảo:
# Windows: .venv\Scripts\activate 
# Linux/Mac: source .venv/bin/activate

# Cài đặt các thư viện phụ thuộc chính:
uv pip install -r requirement.txt

# Tải thư viện face-alignment từ [https://github.com/1adrianb/face-alignment](https://github.com/1adrianb/face-alignment)
uv pip install git+https://github.com/1adrianb/face-alignment


2. CHUẨN BỊ DỮ LIỆU VÀ TRỌNG SỐ
--------------------------------------------------------------------------
Sắp xếp cấu trúc thư mục 'data/' như sau:

- Trọng số mô hình: Tải CNN model 'net.pth' CNN model từ [here](https://www.dropbox.com/s/7ds3aesjjmybjh9/net.pth?dl=0) đặt vào thư mục data/.
- Dữ liệu 3DMM: lấy 2 file 'Model_shape.mat' và 'Model_Expression.mat' từ [3DDFA](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm) và đặt vào thư mục data/.
- Tập dữ liệu Kiểm thử:
    + data/AFLW2000/: Chứa ảnh và nhãn .mat của tập AFLW2000-3D.
    + data/Multi-PIE/: Chứa ảnh chụp đa góc nhìn của các đối tượng.

3. HƯỚNG DẪN KIỂM THỬ (INFERENCE)
--------------------------------------------------------------------------
3.1. Kiểm thử định lượng trên AFLW2000-3D
Sử dụng chuẩn Cắt ảnh Tiêu chuẩn (Scale 1.58) để đạt sai số NME thấp nhất (~3.5% - 5%).
Lệnh chạy: 
python test_aflw2000.py

Kết quả: Biểu đồ đường cong sai số tích lũy (CED Curve) và ảnh báo cáo 4 cột (Gốc | Trái | Chính diện | Phải) được lưu tại result/AFLW2000/.

3.2. Kiểm thử định tính trên Multi-PIE
Sử dụng Camera Tracking để hiển thị khối 3D dưới dạng lưới xám (Clay Render) đa hướng khách quan.
Lệnh chạy: 
python test_multipie.py

Kết quả: Báo cáo so sánh sự nhất quán về hình dạng giữa các góc nhìn được lưu tại result/Multi-PIE/.

4. CÁC CẢI TIẾN KỸ THUẬT TÍCH HỢP
--------------------------------------------------------------------------
- Cắt ảnh Tiêu chuẩn (Standard Cropping): Hiệu chỉnh tâm Y dịch xuống cằm (+0.14 * kích thước) và hệ số Tỷ lệ 1.58 giúp bao phủ trọn vẹn đặc trưng khuôn mặt, giảm NME từ 11% xuống dưới 5%.
- Hiệu chỉnh Tọa độ (Coordinate Correction): Ánh xạ lại trục tọa độ (Z_plot = -Y_vertex) giúp khuôn mặt đứng thẳng, không bị ngược khi dựng hình (render) bằng Matplotlib.
- Theo dõi Camera (Camera Tracking): Tự động bù trừ góc xoay ngang (Yaw) dự đoán vào Camera (azim = -90 + yaw_deg), giúp xuất ảnh báo cáo đúng hướng quan sát Trái/Chính diện/Phải.

5. TRÍCH DẪN
--------------------------------------------------------------------------
@inproceedings{wu2019mvf,
  title={MVF-Net: Multi-View 3D Face Morphable Model Regression},
  author={Wu, Fanzi and Bao, Linchao and Chen, Yajing and Ling, Yonggen and Song, Yibing and Li, Songnan and Ngan, King Ngi and Liu, Wei},
  booktitle={CVPR},
  year={2019}
}
==========================================================================