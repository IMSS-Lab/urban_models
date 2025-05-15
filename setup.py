from setuptools import setup, find_packages

setup(
    name="urban_models",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "torch>=1.12.0",
        "torchvision>=0.13.0",
        "polars>=0.17.0",
        "matplotlib>=3.5.0",
        "netCDF4>=1.6.0",
        "Pillow>=9.0.0",
        "scikit-learn>=1.0.0",
        "seaborn>=0.12.0",
        "georasters>=0.5.24",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "isort",
            "flake8",
        ],
    },
    python_requires=">=3.8",
    author="Institute of Material Science and Sustainability",
    author_email="email@example.com",
    description="Urban modeling tools for crop yield prediction",
    keywords="urban, crop yield, machine learning, pytorch",
    entry_points={
        "console_scripts": [
            "urban-train=main:main",
            "urban-analyze=src.analysis:main",
        ],
    },
)