import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="erexplain", # Replace with your own username
    version="0.1.2",
    author="Vincenzo Martello",
    author_email="v.martello@libero.it",
    description="library to explain models for Entity Resolution",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url= 'https://github.com/vincenzomartello/ERExplain.git',
    download_url = 'https://github.com/vincenzomartello/ERExplain/archive/0.1.2.tar.gz'
    packages=['erexplain'],
    install_requires=[
          'pandas',
          'numpy',
          'tqdm',
          'mlxtend>=0.17.2'
      ]
    classifiers=[
        "Programming Language :: Python :: 3",
        'License :: OSI Approved :: Apache Software License',
        "Operating System :: OS Independent",
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.6',
)
