from distutils.core import setup
from setuptools import find_packages

requirements = []
with open("requirements.txt", "r") as f:
    requirements = [line.rstrip('\n') for line in f]

setup(
    name="retinaface",
    version="0.2",
    packages=find_packages(),
    url="",
    license="MIT",
    author="Tsykunov Dmitriy",
    author_email="tsykunov.d@gmail.com",
    description="Inference script for RetinaFace with MobileNet encoder.",
    package_data={'retinaface': ["weights/*"]},
    include_package_data=True,
    install_requires=requirements
)