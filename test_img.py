import torch
import argparse
import os
from PIL import Image
import tools
import torchvision.transforms as transforms
from model import VggEncoder
import time

parser = argparse.ArgumentParser()
parser.add_argument('--image_path', type=str, default=None, help='path to load images. It should include image name with: front|left|right')
parser.add_argument('--save_dir', type=str, default='./result', help='path to save 3D face shapes')

options = parser.parse_args()
crop_opt = True # Thay đổi thành True nếu bạn muốn cắt ảnh

# 1. Tải 3 góc nhìn ảnh
imgA = Image.open(os.path.join(options.image_path, 'front.jpg')).convert('RGB')
imgB = Image.open(os.path.join(options.image_path, 'left.jpg')).convert('RGB')
imgC = Image.open(os.path.join(options.image_path, 'right.jpg')).convert('RGB')

# 2. Cắt ảnh nếu cần thiết
if crop_opt:
    imgA = tools.crop_image(imgA)
    imgB = tools.crop_image(imgB)
    imgC = tools.crop_image(imgC)

# 3. Chuyển đổi sang Tensor (Giữ nguyên logic cat 9 kênh)
# Lưu ý: tA, tB, tC là các Tensor, còn imgA, imgB, imgC vẫn giữ định dạng PIL để lấy màu sau này
tA = transforms.functional.to_tensor(imgA)
tB = transforms.functional.to_tensor(imgB)
tC = transforms.functional.to_tensor(imgC)

# 4. Khởi tạo và tải mô hình
model = VggEncoder()
model = torch.nn.DataParallel(model).cuda() 
ckpt = torch.load('data/net.pth')
model.load_state_dict(ckpt)
model.eval()

# 5. Chạy Inference
input_tensor = torch.cat([tA, tB, tC], 0).view(1, 9, 224, 224).cuda()
start = time.time()
with torch.no_grad():
    preds = model(input_tensor)
print(f"Inference time: {time.time() - start:.4f}s")

# 6. Hậu xử lý Mesh
preds_np = preds[0].detach().cpu().numpy()
# Trích xuất vertices và triangles
faces3d = tools.preds_to_shape(preds_np)
vertices = faces3d[0]
triangles = faces3d[1]

# --- PHẦN BỔ SUNG: SAMPLING TEXTURE ---

print("[*] Đang thực hiện lấy mẫu màu (Sampling Texture)...")
# Sử dụng hàm fusion để kết hợp màu từ cả 3 ảnh đã crop
final_pil_imgs = [imgA, imgB, imgC]
vertex_colors = tools.sample_texture_fusion(vertices, triangles, final_pil_imgs, preds_np)

# Tạo thư mục lưu nếu chưa có
if not os.path.exists(options.save_dir):
    os.makedirs(options.save_dir)

# 7. Lưu file PLY có đầy đủ màu sắc
save_path = os.path.join(options.save_dir, 'shape_textured.ply')
tools.write_ply(save_path, vertices, triangles, colors=vertex_colors)

print(f"[SUCCESS] Đã lưu kết quả dán màu tại: {save_path}")