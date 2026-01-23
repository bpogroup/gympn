from setuptools import setup, find_packages

setup(
    name="gympn",
    version="0.0.1",
    author="Riccardo Lo Bianco",
    author_email="r.lo.bianco@tue.nl",
    description="A library for Action-Evolution Petri Net environments and agents, based on SimPN.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/bpogroup/gympn",  # Replace with your repository URL
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "gymnasium~=1.0.0",
        "matplotlib~=3.10.0",
        "simpn~=1.3.0",
        "torch~=2.6.0",
        "torch-geometric~=2.6.1",
        "wandb>=0.17.0",
        "dill~=0.4.0",
        "tensorboard>=2.13.0",
        "requests>=2.28.0",
        "Pillow>=10.0.0",
        "pytest>=7.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "networkx>=3.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires="==3.12.8",
)