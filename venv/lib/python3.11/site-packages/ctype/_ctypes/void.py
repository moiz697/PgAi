class void(object):
    '''
    void: This type is a void pointer and can store any pointer value.
    '''
    _name_: str = 'void'
    _deprecated_: bool = False
    _version_: str = '0.0.0'
    _fields_: list = []
    _methods_: list = []
    _types_: list = []

    _bits_: int = 0
    _max_: object = None
    _min_: object = None

    def __new__(cls, value: object = None) -> 'void':
        return super().__new__(cls, value)