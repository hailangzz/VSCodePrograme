
import os
import argparse
from PIL import Image
import cv2

class MediaViewer:
    """多媒体查看器核心类"""
    IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')
    VIDEO_EXTS = ('.mp4', '.avi', '.mov', '.mkv')
    
    def __init__(self, directory):
        self.directory = directory
        self.displayed_count = 0
    
    def show_images(self, max_count=None):
        """图片查看方法"""
        for file in os.listdir(self.directory):
            if max_count and self.displayed_count >= max_count:
                break
                
            if file.lower().endswith(self.IMAGE_EXTS):
                self._display_image(file)
    
    def show_videos(self, max_count=None):
        """视频播放方法""" 
        for file in os.listdir(self.directory):
            if max_count and self.displayed_count >= max_count:
                break
                
            if file.lower().endswith(self.VIDEO_EXTS):
                self._play_video(file)
    
    def _display_image(self, filename):
        """内部图片显示方法"""
        try:
            img_path = os.path.join(self.directory, filename)
            img = Image.open(img_path)
            img.show()
            self.displayed_count += 1
            print(f"已显示图片({self.displayed_count}): {filename}")
        except Exception as e:
            print(f"无法打开图片 {filename}: {str(e)}")
    
    def _play_video(self, filename):
        """内部视频播放方法"""
        try:
            video_path = os.path.join(self.directory, filename)
            cap = cv2.VideoCapture(video_path)
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                cv2.imshow(filename, frame)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            
            cap.release()
            cv2.imshow(filename, None)
            cv2.destroyAllWindows()
            self.displayed_count += 1
            print(f"已播放视频({self.displayed_count}): {filename}")
        except Exception as e:
            print(f"无法播放视频 {filename}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='多媒体查看器')
    parser.add_argument('path', help='文件目录路径')
    parser.add_argument('-n', '--max-count', type=int, help='最大显示数量')
    parser.add_argument('-m', '--mode', choices=['image', 'video'], 
                       required=True, help='查看模式: image/video')
    args = parser.parse_args()
    
    if os.path.isdir(args.path):
        viewer = MediaViewer(args.path)
        if args.mode == 'image':
            viewer.show_images(args.max_count)
        else:
            viewer.show_videos(args.max_count)
    else:
        print("错误: 指定的路径不是目录")

if __name__ == '__main__':
    main()
