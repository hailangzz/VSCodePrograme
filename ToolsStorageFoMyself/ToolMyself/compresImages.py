from PIL import Image
import os
import io

def compress_image(input_path, output_path, target_size_kb, step=5, min_quality=10):
    """
    压缩图片到目标大小（KB）
    :param input_path: 输入图片路径
    :param output_path: 输出图片路径
    :param target_size_kb: 目标大小（KB）
    :param step: 质量递减步长
    :param min_quality: 最低JPEG质量
    """
    target_size = target_size_kb * 1024  # 转换为字节
    img = Image.open(input_path)

    # 初始分辨率
    width, height = img.size

    # 尝试不同分辨率
    while True:
        # 在当前分辨率下，尝试不同的质量
        quality = 95
        while quality >= min_quality:
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=quality, optimize=True)
            size = buffer.tell()
            if size <= target_size:
                # 保存压缩后的图片
                with open(output_path, "wb") as f:
                    f.write(buffer.getvalue())
                print(f"压缩成功: {size/1024:.2f} KB, 分辨率: {img.size}, 质量: {quality}")
                return
            quality -= step

        # 如果质量已经到最低，仍然太大 -> 缩小分辨率
        width = int(width * 0.9)
        height = int(height * 0.9)
        img = img.resize((width, height), Image.LANCZOS)

        # 防止分辨率缩得太小
        if width < 100 or height < 100:
            raise ValueError("无法压缩到目标大小，图片太小或目标过低。")


if __name__ == "__main__":
    input_file = "d:\\3118AC6BDC29F954B8D888DDEB563CC7.png"      # 输入图片
    output_file = "d:\\output2.jpg"    # 输出图片
    target_kb = 1000               # 目标大小 KB

    compress_image(input_file, output_file, target_kb)
