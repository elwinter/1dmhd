import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="1dmhd",
    version="0.0.1",
    author="Eric Winter",
    author_email="eric.winter62@gmail.com",
    description="1-D MHD modeling with neural networks for Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/elwinter/1dmhd",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    tests_require=['nose'],
    test_suite='nose.collector',
)
