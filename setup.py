import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="climtrends", # Replace with your own username
    version="0.0.1",
    author="Travis A. O'Brien",
    author_email="obrienta@iu.edu",
    description="A utility for Bayesian trend regression with a variety of statistical models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.iu.edu/obrienta/climtrends",
    packages=setuptools.find_packages(),
    classifiers=[
    "Programming Language :: Python :: 3",
    "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    )
