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
crop_opt = True # change to True if you want to crop the image
imgA = Image.open(os.path.join(options.image_path, 'front.jpg')).convert('RGB')
imgB = Image.open(os.path.join(options.image_path, 'left.jpg')).convert('RGB')
imgC = Image.open(os.path.join(options.image_path, 'right.jpg')).convert('RGB')
if crop_opt:
    imgA = tools.crop_image(imgA)
    imgB = tools.crop_image(imgB)
    imgC = tools.crop_image(imgC)
imgA = transforms.functional.to_tensor(imgA)
imgB = transforms.functional.to_tensor(imgB)
imgC = transforms.functional.to_tensor(imgC)
model = VggEncoder()
model = torch.nn.DataParallel(model).cuda() 
# model = model.to('cpu')
ckpt = torch.load('data/net.pth')
model.load_state_dict(ckpt)
#print model
input_tensor = torch.cat([imgA, imgB, imgC], 0).view(1, 9, 224, 224).cuda()
# input_tensor = torch.cat([imgA, imgB, imgC], 0).view(1, 9, 224, 224).to('cpu')
start = time.time()
preds = model(input_tensor)
print(time.time() -start)
# faces3d = tools.preds_to_shape(preds[0].detach().cpu().numpy())
# tools.write_ply(os.path.join(options.save_dir, 'shape.ply'), faces3d[0], faces3d[1])

preds_np = preds[0].detach().cpu().numpy()

# Tạo hình khối 3D
faces3d = tools.preds_to_shape(preds_np)
vertices = faces3d[0]
triangles = faces3d[1]

# GỌI HÀM LẤY MÀU: Truyền vào ảnh imgA đã crop
vertex_colors = tools.sample_texture(vertices, imgA, preds_np, view_idx=0)

# Lưu file PLY với tham số colors
if not os.path.exists(options.save_dir):
    os.makedirs(options.save_dir)

save_path = os.path.join(options.save_dir, 'shape_textured.ply')
tools.write_ply(save_path, vertices, triangles, colors=vertex_colors)