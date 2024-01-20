from .._cerr import (ctype_err, CTypeValue)

class double(float):
    '''
    double: This type is a 64-bit floating-point number and can store values ranging from 2.2250738585072014e-308 to 1.7976931348623157e+308 (inclusive).
    '''
    _name_: str = 'double'
    _deprecated_: bool = False
    _version_: str = '0.0.0'
    _fields_: list = []
    _methods_: list = []
    _types_: list = []

    _bits_: int = 64
    _max_: float = 1.7976931348623157e+308
    _min_: float = 2.2250738585072014e-308

    def __new__(cls, value: float = 0.0) -> 'double':
        value_bits: int = value.bit_length()
        if value_bits > cls._bits_:
            cerr: CTypeValue = ctype_err(CTypeValue, f'{str(value)[:5]}...', f'bit_length of {value_bits} is greater than {cls._bits_}')
            raise cerr
        if value < cls._min_:
            value:  float = cls._min_
        if value > cls._max_:
            value:  float = cls._max_
        return super().__new__(cls, value)

class double4(double):
    '''
    double4_t: This type is a 4-bit floating-point number and can store values ranging from 2.2250738585072014e-308 to 1.7976931348623157e+308 (inclusive).
    '''
    _name_: str = 'double4'
    _deprecated_: bool = False
    _version_: str = '0.0.0'
    _fields_: list = []
    _methods_: list = []
    _types_: list = []

    _bits_: int = 4

class double8(double):
    '''
    double8_t: This type is a 8-bit floating-point number and can store values ranging from 2.2250738585072014e-308 to 1.7976931348623157e+308 (inclusive).
    '''
    _name_: str = 'double8'
    _deprecated_: bool = False
    _version_: str = '0.0.0'
    _fields_: list = []
    _methods_: list = []
    _types_: list = []

    _bits_: int = 8

class double16(double):
    '''
    double16_t: This type is a 16-bit floating-point number and can store values ranging from 2.2250738585072014e-308 to 1.7976931348623157e+308 (inclusive).
    '''
    _name_: str = 'double16'
    _deprecated_: bool = False
    _version_: str = '0.0.0'
    _fields_: list = []
    _methods_: list = []
    _types_: list = []

    _bits_: int = 16

class double32(double):
    '''
    double32_t: This type is a 32-bit floating-point number and can store values ranging from 2.2250738585072014e-308 to 1.7976931348623157e+308 (inclusive).
    '''
    _name_: str = 'double32'
    _deprecated_: bool = False
    _version_: str = '0.0.0'
    _fields_: list = []
    _methods_: list = []
    _types_: list = []

    _bits_: int = 32

class double64(double):
    '''
    double64_t: This type is a 64-bit floating-point number and can store values ranging from 2.2250738585072014e-308 to 1.7976931348623157e+308 (inclusive).
    '''
    _name_: str = 'double64'
    _deprecated_: bool = False
    _version_: str = '0.0.0'
    _fields_: list = []
    _methods_: list = []
    _types_: list = []

    _bits_: int = 64