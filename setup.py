from setuptools import setup
import versioneer

requirements = [
    "pandas>=1.3.4",
    "scipy>=1.7.3",
    "tqdm>=4.62.3",
    "numpy>=1.21.4",
    "pyrle>=0.0.33",
    "pyranges>=0.0.112",
    "datatable>=1.0.0",
    "psutil",
]

setup(
    setup_requires=[
        # Setuptools 18.0 properly handles Cython extensions.
        "setuptools>=39.1.0",
        "Cython>=0.29.24",
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
