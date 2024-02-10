from setuptools import setup, find_packages

setup(
    name='tft_dart_pip_package',
    version='0.1',
    packages=find_packages(),
    author='Labib',
    author_email='workmainulislam@email.com',
    description='Runs TFT model on "Electricity" Dataset',
    install_requires=[
    'darts',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
