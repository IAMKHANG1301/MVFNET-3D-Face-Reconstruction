import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from model import VggEncoder
import argparse
import time

# 1. Dataset class cho 300W-LP (Cấu trúc Triplet: Front, Left, Right)
class ThreeHundredWLPDataset(Dataset):
    def __init__(self, data_root, transform=None):
        self.data_root = data_root
        self.transform = transform
        # Trong thực tế, bạn cần một file list hoặc liệt kê thư mục chứa các bộ triplet
        # Bài báo lấy 140k bộ triplet từ 300W-LP
        self.triplet_list = self._load_triplets()

    def _load_triplets(self):
        # Giả sử bạn có file list chứa đường dẫn: front_path, left_path, right_path, label_path
        # Bạn cần chuẩn bị dữ liệu này từ tập 300W-LP
        return [] 

    def __len__(self):
        return len(self.triplet_list)

    def __getitem__(self, idx):
        paths = self.triplet_list[idx]
        
        # Load 3 ảnh
        imgA = Image.open(paths['front']).convert('RGB')
        imgB = Image.open(paths['left']).convert('RGB')
        imgC = Image.open(paths['right']).convert('RGB')
        
        if self.transform:
            imgA = self.transform(imgA)
            imgB = self.transform(imgB)
            imgC = self.transform(imgC)
        
        # Gộp ảnh thành tensor (9, 224, 224) giống trong test_img.py
        imgs = torch.cat([imgA, imgB, imgC], dim=0)

        # Load nhãn (3DMM params và 3 Pose)
        # x_3dmm: 228 dim, pose: 7 dim mỗi view
        label_3dmm = torch.tensor(paths['label_3dmm'], dtype=torch.float32)
        poseA = torch.tensor(paths['poseA'], dtype=torch.float32)
        poseB = torch.tensor(paths['poseB'], dtype=torch.float32)
        poseC = torch.tensor(paths['poseC'], dtype=torch.float32)
        
        labels = torch.cat([label_3dmm, poseA, poseB, poseC], dim=0)
        
        return imgs, labels

# 2. Hàm Loss có giám sát (L_sup)
def compute_loss(preds, targets, l_3dmm, l_pose):
    # Tách tham số từ đầu ra (228 3DMM + 7*3 Pose = 249)
    pred_3dmm = preds[:, :228]
    pred_poses = preds[:, 228:]
    
    target_3dmm = targets[:, :228]
    target_poses = targets[:, 228:]
    
    # L2 loss cho 3DMM và Pose
    loss_3dmm = nn.MSELoss()(pred_3dmm, target_3dmm)
    loss_pose = nn.MSELoss()(pred_poses, target_poses)
    
    # Bài báo sử dụng các trọng số lambda để cân bằng
    # Lambda mặc định: 3DMM=1, Pose=10
    total_loss = l_3dmm * loss_3dmm + l_pose * loss_pose
    
    return total_loss, loss_3dmm, loss_pose

# 3. Hàm thiết lập các tham số dòng lệnh
def get_args():
    parser = argparse.ArgumentParser(description='Huấn luyện MVF-Net Supervised Pretraining')
    
    # Đường dẫn dữ liệu
    parser.add_argument('--data_root', type=str, required=True, help='Đường dẫn đến tập dữ liệu 300W-LP')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Thư mục lưu model')
    
    # Siêu tham số huấn luyện
    parser.add_argument('--batch_size', type=int, default=12, help='Kích thước batch (Mặc định: 12)')
    parser.add_argument('--lr', type=float, default=1e-5, help='Tỷ lệ học (Mặc định: 1e-5)')
    parser.add_argument('--epochs', type=int, default=10, help='Số lượng epoch (Mặc định: 10)')
    
    # Trọng số hàm Loss
    parser.add_argument('--l_3dmm', type=float, default=1.0, help='Trọng số cho loss 3DMM')
    parser.add_argument('--l_pose', type=float, default=10.0, help='Trọng số cho loss Pose')
    
    return parser.parse_args()

# 4. Tiến trình huấn luyện chính
def train(args):
    # Cấu hình thiết bị
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Bắt đầu huấn luyện trên thiết bị: {device}")

    # Khởi tạo model và optimizer
    model = VggEncoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Transform ảnh (resize 224x224 và chuyển sang tensor)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Khởi tạo Dataset và Dataloader
    dataset = ThreeHundredWLPDataset(data_root=args.data_root, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    model.train()
    for epoch in range(args.epochs):
        start_time = time.time()
        for i, (imgs, labels) in enumerate(dataloader):
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            preds = model(imgs)
            
            # Tính toán loss dựa trên các tham số từ args
            loss, l_3d, l_p = compute_loss(preds, labels, args.l_3dmm, args.l_pose)
            
            loss.backward()
            optimizer.step()
            
            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}], Step [{i}/{len(dataloader)}], "
                      f"Total Loss: {loss.item():.4f} (3DMM: {l_3d.item():.4f}, Pose: {l_p.item():.4f})")

        # Lưu checkpoint sau mỗi epoch
        save_path = os.path.join(args.save_dir, f'mvfnet_epoch_{epoch+1}.pth')
        torch.save(model.state_dict(), save_path)
        print(f"Đã lưu checkpoint tại: {save_path} - Thời gian epoch: {time.time() - start_time:.2f}s")

if __name__ == '__main__':
    # Parse arguments và chạy hàm train
    args = get_args()
    train(args)