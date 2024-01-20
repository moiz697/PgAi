from .._cerr import (ctype_err, CTypeValue)

class floats(float):
    '''
    float: This type is a 32-bit floating-point number and can store values ranging from 1.175494351e-38 to 3.402823466e+38 (inclusive).
    '''
    _name_: str = 'float'
    _deprecated_: bool = False
    _version_: str = '0.0.0'
    _fields_: list = []
    _methods_: list = []
    _types_: list = []

    _bits_: int = 32
    _max_: float = 3.402823466e+38
    _min_: float = 1.175494351e-38

    def __new__(cls, value: float = 0.0) -> 'float':
        value_bits: int = value.bit_length()
        if value_bits > cls._bits_:
            cerr: CTypeValue = ctype_err(CTypeValue, f'{str(value)[:5]}...', f'bit_length of {value_bits} is greater than {cls._bits_}')
            raise cerr
        if value < cls._min_:
            value:  float = cls._min_
        if value > cls._max_:
            value:  float = cls._max_
        return super().__new__(cls, value)

class float4(floats):
    '''
    float4_t: This type is a 4-bit floating-point number and can store values ranging from 1.175494351e-38 to 3.402823466e+38 (inclusive).
    '''
    _name_: str = 'float4'
    _deprecated_: bool = False
    _version_: str = '0.0.0'
    _fields_: list = []
    _methods_: list = []
    _types_: list = []

    _bits_: int = 4

class float8(floats):
    '''
    float8_t: This type is an 8-bit floating-point number and can store values ranging from 1.175494351e-38 to 3.402823466e+38 (inclusive).
    '''
    _name_: str = 'float8'
    _deprecated_: bool = False
    _version_: str = '0.0.0'
    _fields_: list = []
    _methods_: list = []
    _types_: list = []

    _bits_: int = 8

class float16(floats):
    '''
    float16_t: This type is a 16-bit floating-point number and can store values ranging from 1.175494351e-38 to 3.402823466e+38 (inclusive).
    '''
    _name_: str = 'float16'
    _deprecated_: bool = False
    _version_: str = '0.0.0'
    _fields_: list = []
    _methods_: list = []
    _types_: list = []

    _bits_: int = 16

class float32(floats):
    '''
    float32_t: This type is a 32-bit floating-point number and can store values ranging from 1.175494351e-38 to 3.402823466e+38 (inclusive).
    '''
    _name_: str = 'float32'
    _deprecated_: bool = False
    _version_: str = '0.0.0'
    _fields_: list = []
    _methods_: list = []
    _types_: list = []

    _bits_: int = 32

class float64(floats):
    '''
    float16_t: This type is a 6-bit floating-point number and can store values ranging from 1.175494351e-38 to 3.402823466e+38 (inclusive).
    '''
    _name_: str = 'float6'
    _deprecated_: bool = False
    _version_: str = '0.0.0'
    _fields_: list = []
    _methods_: list = []
    _types_: list = []

    _bits_: int = 64