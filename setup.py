import setuptools
import pathlib

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# required_packages = (pathlib.Path(__file__).parent / "requirements.txt").read_text().splitlines()
required_packages = ['bertopic==0.9.4',
                     'networkx==2.6.3',
                     'numpy==1.22.0',
                     'pandas==1.3.5',
                     'plotly==5.5.0',
                     'pyvis==0.1.9',
                     'scikit_learn==1.0.2',
                     'sentence_transformers==2.1.0']

setuptools.setup(
    name="stripnet",
    version="0.0.4",
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
