
import os
import subprocess
from you_get import common as you_get

def download_with_youget(url, save_path='./videos'):
    """使用you-get库下载"""
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    try:
        you_get.any_download(url, output_dir=save_path, merge=True)
        print(f"视频已保存到 {save_path}")
    except Exception as e:
        print(f"下载失败: {str(e)}")

def download_with_ffmpeg(url, cookie=None):
    """使用requests+ffmpeg下载(需自行安装ffmpeg)"""
    headers = {
        'User-Agent': 'Mozilla/5.0',
        'Referer': 'https://www.bilibili.com/'
    }
    if cookie:
        headers['Cookie'] = cookie
    
    # 实际使用需替换为真实API解析逻辑
    print("该方法需配合B站API解析实现")
    
if __name__ == '__main__':
    video_url = input("请输入B站视频链接: ")
    method = input("选择下载方式(1.you-get 2.requests+ffmpeg): ")
    
    if method == '1':
        download_with_youget(video_url)
    else:
        download_with_ffmpeg(video_url)
