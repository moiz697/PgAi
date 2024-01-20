from .._cerr import (ctype_err, CTypeValue)

class long(int):
    '''
    long: This type is a 32-bit signed integer and can store values ranging from -2,147,483,648 to 2,147,483,647 (inclusive).
    '''
    _name_: str = 'long'
    _deprecated_: bool = False
    _version_: str = '0.0.0'
    _fields_: list = []
    _methods_: list = []
    _types_: list = []

    _bits_: int = 32
    _max_: int = 2_147_483_647
    _min_: int = -2_147_483_648

    def __new__(cls, value: int = 0) -> 'long':
        value_bits: int = value.bit_length()
        if value_bits > cls._bits_:
            cerr: CTypeValue = ctype_err(CTypeValue, f'{str(value)[:5]}...', f'bit_length of {value_bits} is greater than {cls._bits_}')
            raise cerr
        if value < cls._min_:
            value:  int = cls._min_
        if value > cls._max_:
            value:  int = cls._max_
        return super().__new__(cls, value)