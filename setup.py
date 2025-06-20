from setuptools import setup, find_packages

setup(
    name="gympn",
    version="0.0.1",
    author="Riccardo Lo Bianco",
    author_email="your.email@example.com",
    description="A library for Action-Evolution Petri Net environments and agents, based on SimPN.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/bpogroup/gympn",  # Replace with your repository URL
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "gymnasium==1.0.0",
        "numpy==1.26.4",
        "simpn>=1.2.9",
        "torch==2.5.1",
        "torch-geometric==2.6.1"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires="==3.12.8",
)