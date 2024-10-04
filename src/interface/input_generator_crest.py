from dataclasses import dataclass

@dataclass
class HeaderBlock:
    input: str
    runtype: str 
    threads: int    
    
    def __str__(self):
        outp = '#This is a CREST input file\n'
        for att in self.__dir__():
            if att.startswith('_'):
                continue
            value = getattr(self, att)
            if value is None:
                continue
            outp += format_line(att, value)
        return outp


@dataclass
class CalcBlock:
    id: int
    maxcycle: int
    def __str__(self):
        outp = '[calculation]\n'
        for att in self.__dir__():
            if att.startswith('_'):
                continue
            value = getattr(self, att)
            if value is None:
                pass
            outp += format_line(att, value)
        return outp


class CalculationlevelBlock:
    def __str__(self):
        outp = '[[calculation.level]]\n'
        for att in self.__dir__():
            if att.startswith('_'):
                continue
            value = getattr(self, att)
            if value is None:
                pass
            outp += format_line(att, value)
        return outp


@dataclass
class SsffBlock(CalculationlevelBlock):
    _name = "yeah"
    method: str 
    refgeo: str
    refhessian: str
    refwbo: str
    refff: str 
    useff: bool 


@dataclass
class TsffBlock(CalculationlevelBlock):
    method: str
    refff: str
    id1: int
    id2: int 


class InputMod:
    def __init__(self):
        pass


def format_line(att, value):
    if type(value) == bool:
        return f'{att} = {str(value).lower()}\n'
    elif type(value) == int:
        return f'{att} = {value}\n'
    else:
        return f'{att} = "{value}"\n'
    
def generate_input(header: HeaderBlock, calculation: CalcBlock, ssff1: SsffBlock, ssff2: SsffBlock = '', tsff: TsffBlock = '') :
    return f"{header} \n{calculation} \n{ssff1} \n{ssff2} \n{tsff} \n "

# block1 = HeaderBlock(input='struc2.xyz', runtype='ancopt', threads=4)
# block2 = CalcBlock(id=3, maxcycle=219)
# block3 = SsffBlock(method='ssff', refgeo='struc1.xyz', refhessian='hess1', refwbo='wbo1', refff='ff1', useff=False)
# block4 = SsffBlock(method='ssff', refgeo='struc2.xyz', refhessian='hess2', refwbo='wbo2', refff='ff2', useff=False)
# block5 = TsffBlock(method='tsff', refff='tsff.txt', id1=1, id2=2)

# print(generate_input(block1, block2, block3, block4, block5))