from distutils.core import setup
setup(
  name = 'explainer',         
  packages = ['explainer'],  
  version = '0.1',     
  license='apache-2.0',  
  description = 'library to explain models for Entity Resolution', 
  author = 'Vincenzo Martello',
  author_email = 'v.martello@libero.it',
  url = 'https://github.com/vincenzomartello/ExplainER.git',
  download_url = 'https://github.com/user/reponame/archive/v_01.tar.gz', 
  keywords = ['ML', 'Entity-Resolution', 'explainable-ai'],
  install_requires=[            # I get to this in a second
          'pandas',
          'numpy',
          'tqdm',
          'mlxtend>=0.17.2'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers', 
    'Topic :: Software Development :: Utilities',
    'License :: OSI Approved :: Apache-2.0 License',   # Again, pick a license
    'Programming Language :: Python :: 3', 
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
  ],
)
