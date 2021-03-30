from setuptools import setup, find_namespace_packages

setup(
    name='TulipProphet',
    version='0.1',
    description='',
    author='Lukas Humpe',
    author_email='l.humpe@hotmail.de',
    packages=find_namespace_packages(include=['tulipprophet']),
    install_requires=['pandas==1.2.3',
                      'tensorflow==2.4.1',
                      'stockstats==0.3.2',
                      'numpy==1.19.5',
                      'scikit-learn==0.24.1',
                      'keras-tuner==1.0.2',
                      'flask==1.1.2',
                      'requests==2.25.1',
                      'click==7.1.2'],
    entry_points={
        'console_scripts': ['crypto=tulipprophet.crypto.cli:cli']
    }
)
