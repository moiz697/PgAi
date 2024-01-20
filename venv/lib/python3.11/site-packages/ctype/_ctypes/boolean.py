class boolean(object):
    '''
    boolean: This type is a boolean value and can store either true or false.
    '''
    _name_: str = 'boolean'
    _deprecated_: bool = False
    _version_: str = '0.0.0'
    _fields_: list = []
    _methods_: list = []
    _types_: list = []

    _bits_: int = 1
    _max_: bool = True
    _min_: bool = False

    def __new__(cls, value: bool = False) -> 'boolean':
        return super().__new__(cls, bool(value))