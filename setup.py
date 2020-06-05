from distutils.core import setup
setup(
  name = 'erexplain',         
  packages = ['erxplain'],  
  version = '0.1.1',     
  license='apache-2.0',  
  description = 'library to explain models for Entity Resolution', 
  author = 'Vincenzo Martello',
  author_email = 'v.martello@libero.it',
  url = 'https://github.com/vincenzomartello/ERExplain.git',
  download_url = 'https://github.com/vincenzomartello/ERExplain/archive/0.1.1.tar.gz', 
  keywords = ['ML', 'Entity-Resolution', 'explainable-ai'],
  install_requires=[
          'pandas',
          'numpy',
          'tqdm',
          'mlxtend>=0.17.2'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers', 
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: Apache Software License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
  ],
)
