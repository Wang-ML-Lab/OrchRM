from setuptools import setup, find_packages

setup(
    name="orchrm",
    version="0.1.0",
    packages=find_packages(exclude=["verl*", "outputs*", "*.egg-info"]),
    python_requires=">=3.10",
)
