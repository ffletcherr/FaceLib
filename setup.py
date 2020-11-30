import setuptools

with open("README.md", "r") as fh:
   long_description = fh.read()

setuptools.setup(
    name='facelib',
    version='1.5',
    author="Sajjad Ayobi",
    author_email="sajjjadayobi@gmail.com",
    description="Face Detection & Age Gender & Expression & Recognition",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sajjjadayobi/FaceLib",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)