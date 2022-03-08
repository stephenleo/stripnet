import setuptools
import pathlib

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# required_packages = (pathlib.Path(__file__).parent / "requirements.txt").read_text().splitlines()
required_packages = [
    'cython',
    'numpy',
    'bertopic',
    'networkx',                 
    'pandas',
    'plotly',
    'pyvis',
    'scikit_learn',
    'sentence_transformers',
    'ipywidgets']

setuptools.setup(
    name="stripnet",
    version="0.0.7",
    author="stephenleo",
    author_email="stephen.leo87@gmail.com",
    description="STriP Net: Semantic Similarity of Scientific Papers (S3P) Network",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/stephenleo/stripnet",
    install_requires=required_packages,
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6, <3.9',
)
