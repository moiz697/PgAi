from .._cerr import (ctype_err, CTypeValue)

class longlong(int):
    '''
    long long: This type is a 64-bit signed integer and can store values ranging from -9,223,372,036,854,775,808 to 9,223,372,036,854,775,807 (inclusive).
    '''
    _name_: str = 'longlong'
    _deprecated_: bool = False
    _version_: str = '0.0.0'
    _fields_: list = []
    _methods_: list = []
    _types_: list = []

    _bits_: int = 64
    _max_: int = 9_223_372_036_854_775_807
    _min_: int = -9_223_372_036_854_775_808

    def __new__ (cls, value: int = 0) -> 'longlong':
        value_bits: int = value.bit_length()
        if value_bits > cls._bits_:
            cerr: CTypeValue = ctype_err(CTypeValue, f'{str(value)[:5]}...', f'bit_length of {value_bits} is greater than {cls._bits_}')
            raise cerr
        if value < cls._min_:
            value:  int = cls._min_
        if value > cls._max_:
            value:  int = cls._max_
        return super().__new__(cls, value)