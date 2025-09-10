import os
path = r"d:\image1.png"
print(f"文件存在: {os.path.exists(path)}")
print(f"文件大小: {os.path.getsize(path)/1024:.2f}KB" if os.path.exists(path) else "N/A")
print(f"可读性: {os.access(path, os.R_OK)}")
