from setuptools import setup, find_packages
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

# Additional requirements for development and testing
dev_requirements = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "black>=23.0.0",
    "flake8>=6.0.0",
    "isort>=5.12.0",
    "mypy>=1.0.0",
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
    description="A BLASTx filtering tool using the FAMLI algorithm for ancient DNA studies",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="GNUv3",
    author="Antonio Fernandez-Guerra",
    author_email="antonio@metagenomics.eu",
    url="https://github.com/genomewalker/x-filter",
    packages=find_packages(exclude=["tests"]),
    entry_points={"console_scripts": ["xFilter=x_filter.__main__:main"]},
    install_requires=requirements,
    extras_require={
        "dev": dev_requirements,
        "test": ["pytest>=7.0.0", "pytest-cov>=4.0.0"],
    },
    python_requires=">=3.8",
    keywords=["x-filter", "blastx", "ancient dna", "bioinformatics", "famli"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    include_package_data=True,
)