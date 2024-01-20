class struct(dict):
    '''
    struct: This type is used to define a structure.
    '''
    _name_: str = 'struct'
    _deprecated_: bool = False
    _version_: str = '0.0.0'
    _fields_: list = []
    _methods_: list = []
    _types_: list = []

    def __new__(cls, value: dict = None) -> 'struct':
        if value is None:
            return super().__new__(cls)
        return super().__new__(cls, value)

class union(dict):
    '''
    union: This type is used to define a union.
    '''
    _name_: str = 'union'
    _deprecated_: bool = False
    _version_: str = '0.0.0'
    _fields_: list = []
    _methods_: list = []
    _types_: list = []

    def __new__(cls, value: dict = None) -> 'union':
        if value is None:
            return super().__new__(cls)
        return super().__new__(cls, value)

class enum(dict):
    '''
    enum: This type is used to define an enumerated type.
    '''
    _name_: str = 'enum'
    _deprecated_: bool = False
    _version_: str = '0.0.0'
    _fields_: list = []
    _methods_: list = []
    _types_: list = []

    def __new__(cls, value: dict = None) -> 'enum':
        if value is None:
            return super().__new__(cls)
        return super().__new__(cls, value)