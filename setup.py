from setuptools import setup, find_packages
import ChildProject

requires = {
    'core': ['pandas', 'xlrd', 'jinja2', 'numpy>=1.16.5', 'pympi-ling', 'lxml', 'sox', 'datalad'],
    'zooniverse': ['panoptes_client', 'pydub']
}

setup(
    name='ChildProject',
    version = ChildProject.__version__,
    description='LAAC@LSCP',
    url='git@github.com:lucasgautheron/ChildRecordsData.git',
    author='Lucas',
    author_email='lucas.gautheron@gmail.com',
    license='unlicense',
    packages=find_packages(),
    install_requires=requires['core'] + requires['zooniverse'],
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'child-project=ChildProject.cmdline:main',
        ],
    },
    zip_safe=False
)
