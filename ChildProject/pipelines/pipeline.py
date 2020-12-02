from abc import ABC, abstractmethod

class Pipeline(ABC):
    def __init__(self):
        pass

    def check_setup(self):
        pass

    def setup(self):
        pass

    @abstractmethod
    def run(self, **kwargs):
        pass

    @staticmethod
    def setup_pipeline(parser):
        pass