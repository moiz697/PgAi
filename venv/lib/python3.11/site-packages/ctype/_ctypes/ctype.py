from collections.abc import Callable
from typing import Any

class CType(object):
    _name_:       str  = 'CType'
    _deprecated_: bool = False
    _version_:    str  = '0.0.1'
    _fields_:     list = [
        ('cfunction', Callable),
        ('args',      tuple),
        ('kwargs',    dict)
    ]
    _methods_:    list = [
        ('__init__',    None),
        ('__call__',    None),
        ('__getitem__', int ),
        ('__setitem__', Any ),
        ('__delitem__', int ),
        ('__len__',     int ),
        ('__int__',     int ),
        ('__str__',     str ),
        ('__repr__',    str ),
        ('describe',    dict),
        ('name',        str ),
        ('version',     str ),
        ('deprecated',  bool),
        ('fields',      list),
        ('methods',     list),
        ('types',       list)
    ]
    _types_:      list = []
    def __init__(self, cfunction: Callable, *args, **kwargs) -> None:
        self.cfunction:    Callable = cfunction
        self.args:         tuple    = args
        self.kwargs:       dict     = kwargs
        self.call_stack:   list     = []
        self.result_stack: list     = []

        self.call_stack.append(self)
        self.result_stack.append(None)
        self._types_.append(self)

    def __call__(self, *args, **kwargs) -> 'CType.cfunction':
        def cfunction_wrapper(*args, **kwargs) -> 'CType.cfunction':
            self.call_stack.append((self.cfunction, args, kwargs))
            self.args: tuple = (*self.args, args)
            self.kwargs: dict = {**self.kwargs, **kwargs}
            cfunction_result: Any = None
            try:
                cfunction_result: Any = self.cfunction(*args, **kwargs)
            except Exception as e:
                cfunction_result: Any = e

            finally:
                self.result_stack.append(cfunction_result)
            return cfunction_result

        cfunction_wrapper.cfunction:  Callable = self.cfunction
        cfunction_wrapper.describe:   dict     = self.describe
        cfunction_wrapper.name:       str      = self.name
        cfunction_wrapper.version:    str      = self.version
        cfunction_wrapper.deprecated: bool     = self.deprecated
        cfunction_wrapper.fields:     list     = self.fields
        cfunction_wrapper.methods:    list     = self.methods
        return cfunction_wrapper(*args, **kwargs)

    def __getitem__(self, index: int) -> Any:
        return self.result_stack[index]

    def __setitem__(self, index: int, value: Any) -> None:
        self.result_stack[index] = value

    def __delitem__(self, index: int) -> None:
        del self.result_stack[index]

    def __len__(self) -> int:
        return len(self.result_stack)

    def __int__(self) -> int:
        return len(self.call_stack)

    def __str__(self) -> str:
        return f'{self._name_}(cfunction: {self.cfunction}, lencallstack: {len(self.call_stack)}, lenresultstack: {len(self.result_stack)})'

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def describe(self) -> dict:
        return {'name': self._name_, 'deprecated': self._deprecated_, 'version': self._version_, 'fields': self._fields_, 'methods': self._methods_}

    @property
    def name(self) -> str:
        return self._name_

    @property
    def deprecated(self) -> bool:
        return self._deprecated_

    @property
    def version(self) -> str:
        return self._version_

    @property
    def fields(self) -> list:
        return self._fields_

    @property
    def methods(self) -> list:
        return self._methods_

    @property
    def types(self) -> list:
        return self._types_

class Structure(object):
    _name_:       str  = 'Structure'
    _deprecated_: bool = False
    _version_:    str  = '0.0.1'
    _fields_:     list = [
        ('args',      tuple),
        ('kwargs',    dict)
    ]
    def __init__(self, *args, **kwargs) -> None:
        self.name: str = self.__class__.__name__
        self.base: str = self._name_
        self.args:  tuple = args
        self.kwargs: dict = kwargs
        self.deprecated: bool = self._deprecated_
        self.version: str = self._version_
        self.fields: list = self._fields_
        CType._types_.append(self)

    def __call__(self, *args, **kwargs) -> 'Structure':
        return self

    def __str__(self) -> str:
        return f'ctype.{self.name}(args: {self.args}, kwargs: {self.kwargs})'

    def __repr__(self) -> str:
        return self.__str__()