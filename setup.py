# read the contents of your README file
from os import path

from setuptools import find_packages, setup

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    lines = f.readlines()

# remove images from README
lines = [x for x in lines if ".png" not in x]
long_description = "".join(lines)

setup(
    name="robosuite",
    packages=[package for package in find_packages() if package.startswith("robosuite")],
    install_requires=[
        "numpy==1.23.5",
        "numba>=0.52.0,<=0.53.1",
        "scipy>=1.2.3",
        "free-mujoco-py==2.1.6",
        "gym",
	"pybullet",
	"mujoco-py<2.2,>=2.1",
	"tensorboard==2.12.3",
	"metaworld",
	"git+https://github.com/Farama-Foundation/Metaworld.git@master#egg=metaworld",
	"tensorflow==2.12.0",
	"tensorflow-io-gcs-filesystem==0.32.0",
	"tensorflow-probability==0.20.1",
	"akro==0.0.8",
    ],
    eager_resources=["*"],
    include_package_data=True,
    python_requires=">=3",
    description="CRiSE 1-3-7 — Compilations of Real-World inspired Robotic Task Simulation Environments",
    author="Ivancsics Johannes",
    url="https://github.com/jivancsics/robosuite_CRiSE137",
    author_email="johannes.ivancsics93@gmail.com",
    version="1.0",
    long_description=long_description,
    long_description_content_type="text/markdown",
)
