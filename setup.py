from setuptools import find_packages, setup


setup(
    name="pqc_analysis",
    version="0.2.0",
    description="Trainability, geometry, and topology diagnostics for parameterized quantum circuits",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="AIT HADDOU Marwan",
    author_email="marwan.aithaddou@edu.uca.ac.ma",
    url="https://github.com/AHDMarwan/pqc_analysis",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pennylane",
    ],
    extras_require={
        "legacy": ["torch", "tqdm", "tabulate", "scipy", "matplotlib"],
        "tda": ["torch", "tqdm", "scipy", "matplotlib", "ripser", "persim"],
        "qiskit": ["qiskit", "qiskit-algorithms"],
        "dev": ["pytest"],
        "all": [
            "torch",
            "tqdm",
            "tabulate",
            "scipy",
            "matplotlib",
            "ripser",
            "persim",
            "qiskit",
            "qiskit-algorithms",
            "pytest",
        ],
    },
    python_requires=">=3.11",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
    ],
)
