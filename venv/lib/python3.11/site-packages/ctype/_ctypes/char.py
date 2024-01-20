from .._cerr import (ctype_err, CTypeValue)

class char(str):
    '''
    char: This type is a single-byte character and can store values ranging from 0 to 255 (2^8 - 1).
    '''
    _name_: str = 'char'
    _deprecated_: bool = False
    _version_: str = '0.0.0'
    _fields_: list = []
    _methods_: list = []
    _types_: list = []

    _bits_: int = 8
    _max_: int = 2 ** _bits_ - 1
    _min_: int = 0

    def __new__(cls, value: str = '') -> 'char':
        if len(value) > 1:
            cerr: CTypeValue = ctype_err(CTypeValue, f'{str(value)[:5]}...{str(value)[-5:]}', f'value is too long')
            raise cerr

        return super().__new__(cls, value)

class wchar(str):
    '''
    wchar_t: This type is a wide character and can store values ranging from 0 to 65535 (2^16 - 1).
    '''
    _name_: str = 'wchar'
    _deprecated_: bool = False
    _version_: str = '0.0.0'
    _fields_: list = []
    _methods_: list = []
    _types_: list = []

    _bits_: int = 16
    _max_: int = 2 ** _bits_ - 1
    _min_: int = 0

    def __new__(cls, value: str = '') -> 'wchar':
        if len(value) > 1:
            cerr: CTypeValue = ctype_err(CTypeValue, f'{str(value)[:5]}...{str(value)[-5:]}', f'value is too long')
            raise cerr
        return super().__new__(cls, value)

class chararray(list):
    '''
    chararray: This type is an array of characters and can store values ranging from 0 to 255 (2^8 - 1).
    '''
    _name_: str = 'chararray'
    _deprecated_: bool = False
    _version_: str = '0.0.0'
    _fields_: list = []
    _methods_: list = []
    _types_: list = []

    _bits_: int = 8
    _max_: int = 2 ** _bits_ - 1
    _min_: int = 0

    def __new__(cls, value: list = []) -> 'chararray':
        if not isinstance(value, (int, float, complex, bool)):
            value = list(value)
        else:
            value = [value]
        return super().__new__(cls, value)

class wchararray(list):
    '''
    wchararray: This type is an array of wide characters and can store values ranging from 0 to 65535 (2^16 - 1).
    '''
    _name_: str = 'wchararray'
    _deprecated_: bool = False
    _version_: str = '0.0.0'
    _fields_: list = []
    _methods_: list = []
    _types_: list = []

    _bits_: int = 16
    _max_: int = 2 ** _bits_ - 1
    _min_: int = 0

    def __new__(cls, value: list = []) -> 'wchararray':
        if not isinstance(value, (int, float, complex, bool)):
            value = list(value)
        else:
            value = [value]
        return super().__new__(cls, value)

class string(str):
    '''
    string: Normal string object.
    '''
    _name_: str = 'string'
    _deprecated_: bool = False
    _version_: str = '0.0.0'
    _fields_: list = []
    _methods_: list = []
    _types_: list = []

    _bits_: int = 16
    _max_: int = 2 ** _bits_ - 1
    _min_: int = 0

    def __new__(cls, value: str = '') -> 'string':
        return super().__new__(cls, value)