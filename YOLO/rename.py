import os
import glob


def rename_files_in_order(folder_path):
    # 获取文件夹中所有图片的路径
    img_files = glob.glob(os.path.join(folder_path, '*'))

    # 获取每个图片的大小并存储在字典中
    file_size_dict = {file: os.path.getsize(file) for file in img_files}

    # 按照图片大小排序
    sorted_files = sorted(file_size_dict.items(), key=lambda x: x[1])

    # 按照顺序重新命名图片
    for i, (file_path, _) in enumerate(sorted_files):
        # 获取文件扩展名（如 .jpg, .png）
        _, ext = os.path.splitext(file_path)
        # 创建新的文件名
        new_file_name = f"{i + 1}{ext}"
        new_file_path = os.path.join(folder_path, new_file_name)
        # 重命名文件
        os.rename(file_path, new_file_path)


def rename_images_in_subfolders(root_folder):
    # 遍历根文件夹下的所有子文件夹
    for folder_name in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder_name)
        # 如果这是一个文件夹，就对其中的图片进行重命名
        if os.path.isdir(folder_path):
            rename_files_in_order(folder_path)


# 使用方法：将下面的路径替换为你的根文件夹路径
root_folder = "./datasets"
# print(os.path.exists(root_folder))
rename_images_in_subfolders(root_folder)
