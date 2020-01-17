from setuptools import setup, find_packages


setup(
    name="ethpred",
    packages=find_packages(),
    scripts=["./bin/dummy"],
    install_requires=[
        "matplotlib",
        "tqdm",
        "pandas",
    ]
)
