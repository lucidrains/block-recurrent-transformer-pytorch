from setuptools import setup, find_packages

setup(
  name = 'block-recurrent-transformer-pytorch',
  packages = find_packages(exclude=[]),
  version = '0.1.0',
  license='MIT',
  description = 'Block Recurrent Transformer - Pytorch',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/lucidrains/block-recurrent-transformer-pytorch',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'transformers',
    'attention mechanism',
    'recurrence'
  ],
  install_requires=[
    'beartype',
    'einops>=0.4',
    'memorizing-transformers-pytorch>=0.4.0',
    'torch>=1.6',
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
