from setuptools import setup, find_packages

setup(
    name="eyeoi",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.7",
)