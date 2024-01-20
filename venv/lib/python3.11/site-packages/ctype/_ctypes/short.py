from .._cerr import (ctype_err, CTypeValue)

class short(int):
    '''
    short: This type is a 16-bit signed integer and can store values ranging from -32,768 to 32,767 (inclusive).
    '''
    _name_: str = 'short'
    _deprecated_: bool = False
    _version_: str = '0.0.0'
    _fields_: list = []
    _methods_: list = []
    _types_: list = []

    _bits_: int = 16
    _max_: int = 32_767
    _min_: int = -32_768

    def __new__(cls, value: int = 0) -> 'short':
        value_bits: int = value.bit_length()
        if value_bits > cls._bits_:
            cerr: CTypeValue = ctype_err(CTypeValue, f'{str(value)[:5]}...', f'bit_length of {value_bits} is greater than {cls._bits_}')
            raise cerr
        if value < cls._min_:
            value:  int = cls._min_
        if value > cls._max_:
            value:  int = cls._max_
        return super().__new__(cls, value)