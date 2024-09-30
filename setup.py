from setuptools import setup
import versioneer

# Your package requirements
requirements = [
    "pandas>=2.2.3",
    "tqdm>=4.62.3",
    "numpy>=2.0.0",
    "psutil",
    "duckdb",
    "numba",
    "pyarrow",
]

setup(
    setup_requires=[
        "setuptools>=39.1.0",
        "Cython>=0.29.24",
        "numpy>=1.21.2",
    ],
    name="x-filter",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="A simple tool to filter BLASTx m8 files using the FAMLI algorithm",
    license="GNUv3",
    author="Antonio Fernandez-Guerra",
    author_email="antonio@metagenomics.eu",
    url="https://github.com/genomewalker/x-filter",
    packages=["x_filter"],
    entry_points={"console_scripts": ["xFilter=x_filter.__main__:main"]},
    install_requires=requirements,
    keywords="x-filter",
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
