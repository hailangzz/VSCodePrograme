
import os
import argparse
from PIL import Image
import cv2

def show_images(directory, max_count=None):
    """图片查看模式"""
    image_exts = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')
    displayed = 0
    
    for file in os.listdir(directory):
        if max_count and displayed >= max_count:
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

def show_videos(directory, max_count=None):
    """视频播放模式"""
    video_exts = ('.mp4', '.avi', '.mov', '.mkv')
    played = 0
    
    for file in os.listdir(directory):
        if max_count and played >= max_count:
            break
            
        if file.lower().endswith(video_exts):
            try:
                video_path = os.path.join(directory, file)
                cap = cv2.VideoCapture(video_path)
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    cv2.imshow(file, frame)
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        break
                
                cap.release()
                cv2.destroyAllWindows()
                played += 1
                print(f"已播放视频({played}): {file}")
            except Exception as e:
                print(f"无法播放视频 {file}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='多媒体查看器')
    parser.add_argument('path', help='文件目录路径')
    parser.add_argument('-n', '--max-count', type=int, 
                       help='最大显示数量')
    parser.add_argument('-m', '--mode', choices=['image', 'video'], 
                       required=True, help='查看模式: image/video')
    args = parser.parse_args()
    
    if os.path.isdir(args.path):
        if args.mode == 'image':
            show_images(args.path, args.max_count)
        else:
            show_videos(args.path, args.max_count)
    else:
        print("错误: 指定的路径不是目录")
