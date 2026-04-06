# MVF-Net: Multi-View 3D Face Morphable Model Regression
Testing code for the paper.
> [MVF-Net: Multi-View 3D Face Morphable Model Regression](https://arxiv.org/abs/1904.04473).   
> Fanzi Wu*, Linchao Bao*, Yajing Chen, Yonggen Ling, Yibing Song, Songnan Li, King Ngi Ngan, Wei Liu. 
> CVPR 2019.

## Installation
1. Python 2.7 (Numpy, PIL, scipy)
2. Pytorch 0.4.0, torchvision
3. face-alignment package from [https://github.com/1adrianb/face-alignment](https://github.com/1adrianb/face-alignment). This code is used for face cropping and will be replaced by face detection algorithm in the future.

4. `Model_shape.mat` and `Model_Expression.mat` from [3DDFA](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm).
## Test
You can download the CNN model from [here](https://www.dropbox.com/s/7ds3aesjjmybjh9/net.pth?dl=0) and copy it into `data` folder.
Then you can test the model by:
```
python test_img.py --image_path ./data/imgs --save_dir ./result
```
If you are testing the code with your own images, please organize multiview images as:
```
folder
+--front.jpg
+--left.jpg
+--right.jpg
```
and change `line 15` in `test_img.py` as:
```
crop_opt = True
```
## Citation
If you find this work useful in your research, please cite:
```
@inproceedings{wu2019mvf,
  title={MVF-Net: Multi-View 3D Face Morphable Model Regression},
  author={Wu, Fanzi and Bao, Linchao and Chen, Yajing and Ling, Yonggen and Song, Yibing and Li, Songnan and Ngan, King Ngi and Liu, Wei},
  booktitle={CVPR},
  year={2019}
}
```

5. HƯỚNG DẪN HUẤN LUYỆN
Sử dụng lệnh sau để bắt đầu huấn luyện giai đoạn tiền huấn luyện (Pretraining):

Lệnh mẫu:
python train.py --data_root "đường/dẫn/đến/300W-LP" \
                --save_dir "./checkpoints" \
                --batch_size 12 \
                --lr 1e-5 \
                --epochs 10

Các tham số mặc định (theo bài báo):
- Optimizer: Adam[cite: 245].
- Learning Rate: 1e-5 cho supervised, 1e-6 cho self-supervised[cite: 245].
- Batch size: 12[cite: 245].
- Lambda 3DMM: 1.0[cite: 246].
- Lambda Pose: 10.0[cite: 246].

6. QUY TRÌNH HUẤN LUYỆN 2 GIAI ĐOẠN
- Bước 1 (Supervised): Huấn luyện trên 300W-LP để mô hình hội tụ cơ bản[cite: 183].
- Bước 2 (Self-supervised): Huấn luyện trên Multi-PIE sử dụng Photo Loss và 
  Align Loss để tinh chỉnh độ chính xác dựa trên ràng buộc đa góc nhìn
