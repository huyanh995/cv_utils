from setuptools import setup, find_packages

setup(
    name='cv_utils',
    version='0.1',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    description='Utility package for daily Computer Vision tasks',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Huy Anh Nguyen',
    author_email='ahunguyen@cs.stonybrook.edu',
    install_requires=[
        # Your dependencies
    ],
    # include any other necessary metadata
)
