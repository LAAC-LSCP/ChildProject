from abc import ABC, abstractmethod
from enum import Enum

class Message(ABC):
    FAM={2:'warning', 3:'error'}

    def __init__(self, family:int, level:int, category:str, message:str):
        self.fam=family
        self.level = level
        self.cat=category
        self.mes=message

    def __eq__(self,other):
        return self.fam == other.fam and self.level == other.level and self.cat == other.cat and self.mes == other.mes

    def __ne__(self, other):
        return self.fam != other.fam or self.level != other.level or self.cat != other.cat or self.mes != other.mes

    def __le__(self, other):
        return self.fam < other.fam or (self.fam == other.fam and (self.level < other.level or (self.level == other.level and self.cat <= other.cat)))

    def __ge__(self, other):
        return self.fam > other.fam or (self.fam == other.fam and (
                    self.level > other.level or (self.level == other.level and self.cat >= other.cat)))

    def __lt__(self, other):
        return self.fam < other.fam or (self.fam == other.fam and (self.level < other.level or (self.level == other.level and self.cat < other.cat)))

    def __gt__(self, other):
        return self.fam > other.fam or (self.fam == other.fam and (
                self.level > other.level or (self.level == other.level and self.cat > other.cat)))

    def __str__(self):
        return self.mes

    def __repr__(self):
        return f"{FAM[self.fam]}({self.mes})"

class Error(Message):
    def __init__(self, level:int, category:str, message:str):
        super().__init__(3,level, category, message)

class Warning(Message):
    def __init__(self, level:int, category:str, message:str):
        super().__init__(2,level, category, message)


class ERROR(Enum):
    UNQ_EXP = Error(1,'Experiment not unique', f"Column <experiment> must be unique across the dataset, in both children.csv and recordings.csv , {len(exp_values)} different values were found: {exp_values}")

class WARNING(Enum):
    FREQ_REC = Warning()