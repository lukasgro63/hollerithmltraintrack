from setuptools import find_packages, setup

setup(
    name="HollerithMLTrainTrack",
    version="0.1.0",
    author="Lukas Grodmeier",
    author_email="lukasgro63@gmail.com",
    description="A tool for tracking and analyzing ML model training processes.",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/lukasgro63/hollerithmltraintrack",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
            "scikit-learn",
            "pandas",
            "numpy",
            "codecarbon"
        ],
    extras_require={
        "dev": [
            "pytest",
            "flake8",
        ],
    },
)
