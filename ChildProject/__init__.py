# read version from installed package
try:
    from importlib.metadata import version
except ImportError:
    from importlib_metadata import version
__version__ = version("ChildProject")

__all__ = [
    'tables',
    'projects',
    'annotations',
    'converters',
    'metrics'
]
