
import os
import argparse
from PIL import Image

def show_images(directory):
    """显示目录下所有图片"""
    image_exts = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')
    for file in os.listdir(directory):
        if file.lower().endswith(image_exts):
            try:
                img_path = os.path.join(directory, file)
                img = Image.open(img_path)
                img.show()
                print(f"已显示图片: {file}")
            except Exception as e:
                print(f"无法打开图片 {file}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='图片查看器')
    parser.add_argument('path', help='图片目录路径')
    args = parser.parse_args()
    
    if os.path.isdir(args.path):
        show_images(args.path)
    else:
        print("错误: 指定的路径不是目录")

if __name__ == '__main__':
    main()
