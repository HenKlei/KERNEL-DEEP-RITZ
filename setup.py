from setuptools import setup, find_packages


dependencies = [
    'numpy',
    'scipy',
    'matplotlib',
    '2l-VKOGA @ git+https://gitlab.mathematik.uni-stuttgart.de/pub/ians-anm/2l-vkoga@v0.1.2',
    'torch',
    'tikzplotlib',
]

setup(
    name='kernelDR',
    version='0.1.0',
    description='Python implementation of the deep Ritz method using kernel approximations',
    author='Tizian Wenzel, Hendrik Kleikamp',
    author_email='tizian.wenzel@uni-hamburg.de, hendrik.kleikamp@uni-muenster.de',
    maintainer='Tizian Wenzel',
    maintainer_email='tizian.wenzel@uni-hamburg.de',
    packages=find_packages(),
    install_requires=dependencies,
)
