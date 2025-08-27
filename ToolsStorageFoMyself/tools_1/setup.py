
from setuptools import setup, find_packages

setup(
    name="img-viewer",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        'Pillow>=9.0.0',
    ],
    entry_points={
        'console_scripts': [
            'img-viewer=img_viewer.cli:main',
        ],
    },
    author="Your Name",
    description="简单的图片目录查看工具",
    license="MIT",
    keywords="image viewer",
)
