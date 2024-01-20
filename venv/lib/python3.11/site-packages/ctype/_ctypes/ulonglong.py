from .._cerr import (ctype_err, CTypeValue)

class ulonglong(int):
    '''
    unsigned long long: This type is typically represented using 64 bits (8 bytes) and can store values ranging from 0 to 18446744073709551615 (2^64 - 1). It provides an even larger range compared to unsigned long.
    '''
    _name_: str = 'ulonglong'
    _deprecated_: bool = False
    _version_: str = '0.0.0'
    _fields_: list = []
    _methods_: list = []
    _types_: list = []

    _bits_: int = 64
    _max_: int = 2 ** _bits_ - 1
    _min_: int = 0

    def __new__(cls, value: int = 0) -> 'ulonglong':
        value: int = abs(value)
        value_bits: int = value.bit_length()
        if value_bits > cls._bits_:
            cerr: CTypeValue = ctype_err(CTypeValue, f'{str(value)[:5]}...', f'bit_length of {value_bits} is greater than {cls._bits_}')
            raise cerr
        if value < cls._min_:
            value:  int = cls._min_
        if value > cls._max_:
            value:  int = cls._max_
        return super().__new__(cls, value)