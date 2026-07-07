"""Installation script for the 'unitree_rl_mjlab' python package."""

from setuptools import setup, find_packages

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    "mjlab==1.2.0",
    "pandas>=1.5",
    "tqdm>=4.65",
    "scipy>=1.10",
    "numpy>=1.23",
    "h5py>=3.9",
    "tabulate>=0.9",
    "pyyaml>=6.0",
    "opencv-python>=4.8",
    "mink>=0.3",
    "seaborn>=0.12",
    "pytest>=7.4"
]

# Installation operation
setup(
    name="softmimic-mjlab",
    version="0.1.0",
    packages=find_packages(
        include=[
            "src",
            "softmimic_deploy",
            "softmimic_deploy.*",
            "compliant_motion_augmentation",
            "compliant_motion_augmentation.*",
        ]
    ),
    include_package_data=True,
    install_requires=INSTALL_REQUIRES,
)
