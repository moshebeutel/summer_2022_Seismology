import os
from pathlib import Path
from setuptools import setup, find_packages


with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


with open(Path(os.path.dirname(__file__)) / "requirements.txt") as f:
    required = f.readlines()

setup(
    name="seisynth",
    version="0.0.1",
    author="Moshe Beutel",
    author_email="moshebeutel@gmail.com",
    description="The creation and evaluation of AI models on synthetic test sets",
    license="GPLv3",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/moshebeutel/summer_2022_Seismology.git",
    packages=find_packages(exclude="tests"),
    python_requires=">=3.9",
    install_requires=required,
)
