from setuptools import setup, find_packages

setup(
    name='ChildProject',
    version='0.0.2',
    description='LAAC@LSCP',
    url='git@github.com:lucasgautheron/ChildRecordsData.git',
    author='Lucas',
    author_email='lucas.gautheron@gmail.com',
    license='unlicense',
    packages=find_packages(),
    install_requires=['pandas', 'xlrd', 'jinja2', 'numpy', 'pympi-ling', 'sox', 'datalad'],
    scripts=['child-project'],
    zip_safe=False
)
