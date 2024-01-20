class ptr(object):
    '''
    ptr: This type is a pointer and can store any pointer value.
    '''
    _name_: str = 'ptr'
    _deprecated_: bool = False
    _version_: str = '0.0.0'
    _fields_: list = []
    _methods_: list = []
    _types_: list = []

    _bits_: int = 0
    _max_: object = None
    _min_: object = None

    def __new__(cls, value: object = None) -> 'ptr':
        return super().__new__(cls, value)