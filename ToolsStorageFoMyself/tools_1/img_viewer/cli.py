
import os
import argparse
from PIL import Image

def show_images(directory, max_count=None):
    """显示目录下图片（支持数量限制）"""
    image_exts = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')
    displayed = 0
    
    for file in os.listdir(directory):
        if max_count is not None and displayed >= max_count:
            break
            
        if file.lower().endswith(image_exts):
            try:
                img_path = os.path.join(directory, file)
                img = Image.open(img_path)
                img.show()
                displayed += 1
                print(f"已显示图片({displayed}): {file}")
            except Exception as e:
                print(f"无法打开图片 {file}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='图片查看器')
    parser.add_argument('path', help='图片目录路径')
    parser.add_argument('-n', '--max-count', type=int, 
                       help='最大显示图片数量')
    args = parser.parse_args()
    
    if os.path.isdir(args.path):
        show_images(args.path, args.max_count)
    else:
        print("错误: 指定的路径不是目录")
