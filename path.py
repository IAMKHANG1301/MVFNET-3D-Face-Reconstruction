import pathlib

def display_tree(directory, indent=""):
    path = pathlib.Path(directory)
    
    if not path.exists():
        print(f"Lỗi: Đường dẫn '{directory}' không tồn tại.")
        return

    # Lấy danh sách file/folder và sắp xếp (thư mục hiện trước)
    items = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
    
    for i, item in enumerate(items):
        # Kiểm tra xem đây có phải phần tử cuối cùng trong thư mục hiện tại không
        is_last = (i == len(items) - 1)
        prefix = "└── " if is_last else "├── "
        
        print(f"{indent}{prefix}{item.name}")
        
        # Nếu là thư mục, tiếp tục đệ quy vào bên trong
        if item.is_dir():
            extension = "    " if is_last else "│   "
            display_tree(item, indent + extension)

if __name__ == "__main__":
    # Thay đổi đường dẫn thư mục bạn muốn xem ở đây
    target_dir = "./data/NoWDataset"  # "." là thư mục hiện tại
    print(f"Cấu trúc thư mục của: {pathlib.Path(target_dir).absolute()}")
    display_tree(target_dir)