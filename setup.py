from setuptools import setup

setup(
    name='ChildProject',
    version='0.0.2',
    description='LAAC@LSCP',
    url='git@github.com:lucasgautheron/ChildRecordsData.git',
    author='Lucas',
    author_email='lucas.gautheron@gmail.com',
    license='unlicense',
    packages=['ChildProject'],
    install_requires=['pandas', 'xlrd', 'jinja2', 'numpy', 'pympi-ling'],
    zip_safe=False
)
