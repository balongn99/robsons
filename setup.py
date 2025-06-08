from setuptools import setup, find_packages

setup(
    name="robsondb",
    version="0.1.0",
    author="balongn99",
    author_email="nbl130899@gmail.com",
    description="Packaged Robson ASE database (balongtest.db) for easy import",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/balongn99/robsondb",
    packages=find_packages(),
    include_package_data=True,
    package_data={"robsondb": ["*.db"]},
    install_requires=[
        "ase>=3.22.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: None",
    ],
    python_requires=">=3.7",
)

