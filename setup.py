import os
import shutil
from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import find_packages, setup


class DecoderBuild(build_ext):
    def build_extension(self, ext):
        # update.sh can reuse a decoder built for this interpreter.
        destination = Path(self.get_ext_fullpath(ext.name))
        cached = Path(__file__).resolve().parent.joinpath(*ext.name.split(".")[:-1], destination.name)
        if "ONLY_UPDATE_PYTHON" in os.environ and cached.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            if cached.resolve() != destination.resolve():
                shutil.copy2(cached, destination)
        else:
            super().build_extension(ext)


with open("./requirements.txt", "r", encoding="utf-8") as fh:
    install_requires = fh.read()


long_description = "See README: https://github.com/Tianmouc/tianmoucv/blob/main/README.md"


setup(
    name='tianmoucv',                     # 模块的名称
    version='0.4.3.2',                # 版本号 #带.1的是pub版本
    author='Yihan Lin,Taoyi Wang,Yapeng Meng,Yuguo Chen',        # 作者名称
    author_email='532109881@qq.com',      # 作者邮箱
    description='Algorithms library for Tianmouc sensor',   # 简要描述
    url='https://github.com/Tianmouc/tianmoucv_dev',  # 项目主页的URL
    ext_modules=[
        Pybind11Extension(f"tianmoucv.{decoder}.rod_decoder_py",
                          [f"tianmoucv/{decoder}/rod_decoder_py.cpp"], cxx_std=17)
        for decoder in ("rdp_usb", "rdp_pcie")
    ],
    cmdclass={"build_ext": DecoderBuild},
    exclude_package_data={"": ["rod_decoder_py*.so", "rod_decoder_py*.pyd",
                               "rdp_*/rod_decoder_py*.so", "rdp_*/rod_decoder_py*.pyd"]},
    packages=find_packages(),   # 告诉 setuptools 自动找到要安装的包
    package_data = {'':['rdp_pcie/*',
                        'rdp_usb/*',
                        'camera/*',
                        'data/blc/*',
                        'sim/*',
                        'camera/lib/*']},
    include_package_data=True,
    install_requires=install_requires,
    long_description=long_description,
    long_description_content_type="text/plain",
    # 可选的内容
    keywords='tianmoucv',           # 关键词
    classifiers=[
        "Programming Language :: Python :: 3 :: Only",
        'License :: OSI Approved :: MIT License',
        "Operating System :: OS Independent",
        'Intended Audience :: Developers',
    ],
    python_requires='>=3.8'
)
