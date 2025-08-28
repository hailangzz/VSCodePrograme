
from setuptools import setup, find_packages

setup(
    name="media-viewer",
    version="2.1.0",
    packages=find_packages(),
    install_requires=[
        'Pillow>=9.0.0',
        'opencv-python>=4.5.0',
    ],
    entry_points={
        'console_scripts': [
            'media-viewer=media_viewer:main',
        ],
    },
    author="Your Name",
    description="支持图片/视频查看的多媒体工具(类封装版)",
    license="MIT",
    keywords="media viewer image video class",
)
