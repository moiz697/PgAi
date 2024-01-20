from .._cerr import (ctype_err, CTypeValue)

class ints(int):
    '''
    int: This type is a 32-bit signed integer and can store values ranging from -2,147,483,648 to 2,147,483,647 (inclusive)
    '''
    _name_: str = 'int'
    _deprecated_: bool = False
    _version_: str = '0.0.0'
    _fields_: list = []
    _methods_: list = []
    _types_: list = []

    _bits_: int = 32
    _max_: int = 2_147_483_647
    _min_: int = -2_147_483_648

    def __new__(cls, value: int = 0) -> 'int':
        value_bits: int = value.bit_length()
        if value_bits > cls._bits_:
            cerr: CTypeValue = ctype_err(CTypeValue, f'{str(value)[:5]}...', f'bit_length of {value_bits} is greater than {cls._bits_}')
            raise cerr
        if value < cls._min_:
            value:  int = cls._min_
        if value > cls._max_:
            value:  int = cls._max_
        return super().__new__(cls, value)

class int4(ints):
    '''
    int4_t: This type is a 4-bit signed integer and can store values ranging from -8 to 7 (inclusive).
    '''
    _name_: str = 'int4'
    _deprecated_: bool = False
    _version_: str = '0.0.0'
    _fields_: list = []
    _methods_: list = []
    _types_: list = []

    _bits_: int = 4
    _max_: int = 7
    _min_: int = -8

class int8(ints):
    '''
    int8_t: This type is an 8-bit signed integer and can store values ranging from -128 to 127 (inclusive).
    '''
    _name_: str = 'int8'
    _deprecated_: bool = False
    _version_: str = '0.0.0'
    _fields_: list = []
    _methods_: list = []
    _types_: list = []

    _bits_: int = 8
    _max_: int = 127
    _min_: int = -128

class int16(ints):
    '''
    int16_t: This type is a 16-bit signed integer and can store values ranging from -32,768 to 32,767 (inclusive).
    '''
    _name_: str = 'int16'
    _deprecated_: bool = False
    _version_: str = '0.0.0'
    _fields_: list = []
    _methods_: list = []
    _types_: list = []

    _bits_: int = 16
    _max_: int = 32_767
    _min_: int = -32_768

class int32(ints):
    '''
    int32_t: This type is a 32-bit signed integer and can store values ranging from -2,147,483,648 to 2,147,483,647 (inclusive).
    '''
    _name_: str = 'int32'
    _deprecated_: bool = False
    _version_: str = '0.0.0'
    _fields_: list = []
    _methods_: list = []
    _types_: list = []

    _bits_: int = 32
    _max_: int = 2_147_483_647
    _min_: int = -2_147_483_648

class int64(ints):
    '''
    int64_t: This type is a 64-bit signed integer and can store values ranging from -9,223,372,036,854,775,808 to 9,223,372,036,854,775,807 (inclusive).
    '''
    _name_: str = 'int64'
    _deprecated_: bool = False
    _version_: str = '0.0.0'
    _fields_: list = []
    _methods_: list = []
    _types_: list = []

    _bits_: int = 64
    _max_: int = 9_223_372_036_854_775_807
    _min_: int = -9_223_372_036_854_775_808