from setuptools import find_packages, setup

setup(
    name="chiplet-gym",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "gymnasium>=0.29.0",
        "stable-baselines3>=2.2.0",
        "numpy>=1.24.0",
        "torch>=2.0.0",
    ],
    author="Jashaswimalya Acharjee",
    description="RL-based Optimization for Chiplet AI Accelerators",
    python_requires=">=3.8",
)
