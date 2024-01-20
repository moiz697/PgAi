class typedef(object):
    '''
    typedef: This type is used to create a new type with the same name as the original type.
    '''
    _name_: str = 'typedef'
    _deprecated_: bool = False
    _version_: str = '0.0.0'
    _fields_: list = []
    _methods_: list = []
    _types_: list = []

    def __new__(cls, value: object = None) -> 'typedef':
        if value is None:
            return super().__new__(cls)
        return super().__new__(cls, value)

class volatile(object):
    '''
    volatile: This type is used to indicate that a variable or function parameter is not modified by the function.
    '''
    _name_: str = 'volatile'
    _deprecated_: bool = False
    _version_: str = '0.0.0'
    _fields_: list = []
    _methods_: list = []
    _types_: list = []

    def __new__(cls, value: object = None) -> 'volatile':
        if value is None:
            return super().__new__(cls)
        return super().__new__(cls, value)