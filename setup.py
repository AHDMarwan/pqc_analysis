
from setuptools import setup, find_packages

setup(
    name="pqc_analysis",
    version="0.1.0",
    description="PQC analysis toolkit for quantum circuits: geometry and topology",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="AIT HADDOU Marwan",
    author_email="marwan.aithaddou@edu.uca.ac.ma",
    url="https://github.com/AHDMarwan/pqc_analysis",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",
        "pennylane",
        "tqdm",
        "tabulate",
        "ripser"
    ],
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
