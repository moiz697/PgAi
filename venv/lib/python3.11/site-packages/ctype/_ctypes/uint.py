from .._cerr import (ctype_err, CTypeValue)

class uint(int):
    '''
    unsigned int: This type is typically represented using 32 bits (4 bytes) and can store values ranging from 0 to 4294967295 (2^32 - 1).
    '''
    _name_: str = 'uint'
    _deprecated_: bool = False
    _version_: str = '0.0.0'
    _fields_: list = []
    _methods_: list = []
    _types_: list = []

    _bits_: int = 32
    _max_: int = 2 ** _bits_ - 1
    _min_: int = 0
    
    def __new__(cls, value: int = 0) -> 'uint':
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

class uint4(uint):
    '''
    uint4_t: This type is a 4-bit unsigned integer and can store values ranging from 0 to 15 (inclusive).
    '''
    _name_: str = 'uint4'
    _deprecated_: bool = False
    _version_: str = '0.0.0'
    _fields_: list = []
    _methods_: list = []
    _types_: list = []

    _bits_: int = 4
    _max_: int = 15
    _min_: int = 0

class uint8(uint):
    '''
    uint8_t: This type is an 8-bit unsigned integer and can store values ranging from 0 to 255 (2^8 - 1).
    '''
    _name_: str = 'uint8'
    _deprecated_: bool = False
    _version_: str = '0.0.0'
    _fields_: list = []
    _methods_: list = []
    _types_: list = []

    _bits_: int = 8
    _max_: int = 2 ** _bits_ - 1
    _min_: int = 0

class uint16(uint):
    '''
    uint16_t: This type is a 16-bit unsigned integer and can store values ranging from 0 to 65535 (2^16 - 1).
    '''
    _name_: str = 'uint16'
    _deprecated_: bool = False
    _version_: str = '0.0.0'
    _fields_: list = []
    _methods_: list = []
    _types_: list = []

    _bits_: int = 16
    _max_: int = 2 ** _bits_ - 1
    _min_: int = 0
    

class uint32(uint):
    '''
    uint32_t: This type is a 32-bit unsigned integer and can store values ranging from 0 to 4294967295 (2^32 - 1).
    '''
    _name_: str = 'uint32'
    _deprecated_: bool = False
    _version_: str = '0.0.0'
    _fields_: list = []
    _methods_: list = []
    _types_: list = []

    _bits_: int = 32
    _max_: int = 2 ** _bits_ - 1
    _min_: int = 0

class uint64(uint):
    '''
    uint64_t: This type is a 64-bit unsigned integer and can store values ranging from 0 to 18446744073709551615 (2^64 - 1).
    '''
    _name_: str = 'uint64'
    _deprecated_: bool = False
    _version_: str = '0.0.0'
    _fields_: list = []
    _methods_: list = []
    _types_: list = []

    _bits_: int = 64
    _max_: int = 2 ** _bits_ - 1
    _min_: int = 0