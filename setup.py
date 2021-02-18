
from setuptools import setup, find_packages

setup(
    name="futures",
    scripts=["run.py"],
    package_dir={"futures": "futures"},
    packages=find_packages(),
)
