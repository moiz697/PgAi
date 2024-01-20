'''
```
ctype: (V7.4.2)

Interact with cython, python, ctypes, and use other c/c++ tools
```

## Installing
```shell
# Linux/macOS
python3 pip install -U ctype

# Windows
py -3 -m pip install -U ctype
```

```python
import ctype
```
```python
# Base Classes
ctype.CType         # Decorator Class
ctype.Structure
```
```python
ctype.uint
ctype.uint4
ctype.uint8
ctype.uint16
ctype.uint32
ctype.uint64
```
```python
ctype.ints
ctype.int4
ctype.int8
ctype.int16
ctype.int32
ctype.int64
```
```python
ctype.floats
ctype.float4
ctype.float8
ctype.float16
ctype.float32
ctype.float64
```
```python
ctype.double
ctype.double4
ctype.double8
ctype.double16
ctype.double32
ctype.double64
```
```python
ctype.long
ctype.longlong
ctype.short
```
```python
ctype.ulong
ctype.ulonglong
ctype.ushort
```
```python
ctype.char
ctype.wchar
ctype.chararray
ctype.wchararray
```
```python
ctype.struct
ctype.union
ctype.enum
```
```python
ctype.typedef
ctype.volatile
```
```python
ctype.boolean
```
```python
ctype.void
```
```python
ctype.ptr
```
```python
ctype.cout, ctype.stdout, ctype.printf
ctype.cin, ctype.stdin
ctype.cerr, ctype.stderr
ctype.clog, ctype.log
ctype.cexit, ctype.exit
ctype.cabort, ctype.abort
ctype.endl
```
```python
# ctype.Exceptions
ctype.ctype_err <cerr_type> <cobject> <cmessage>

ctype.CTypeException
ctype.CTypeRuntime
ctype.CTypeIndex
ctype.CTypeName
ctype.CTypeImport
ctype.CTypeAttr
ctype.CTypeKey
ctype.CTypeValue
ctype.CTypeIndent
ctype.CTypeError
ctype.CTypeKeyboard
ctype.CTypeUnfinished
```
# Use ctypes in ctype
```python
# Comes inside of the ctype lib

from ctype import cdll, c_int, c_char_p, c_void_p, c_double, c_bool, ...
```
```python
# View all defined CType's
ctype.types

# CType
ctype.ctype
ctype.CType

# Structure
ctype.Structure
ctype.Struct
ctype.structure
```
```python
ctype.require('a_module', 'numpy', 'some_other_module')
```

# Links
- [Github](https://github.com/notepads-ai)
- [Project](https://pypi.org/project/ctype/)
'''

from ctypes import *

from ._cerr import *
from ._ctypes import *
from ._cbuiltin import *

def require(*modules, **kwargs) -> tuple:
    for module in modules:
        try:
            __import__(module, globals(), locals(), [], 0)
        except ImportError as e:
            raise ctype_err(CTypeImport, module, str(e))
    return tuple(globals().get(module) for module in modules)

structure: CType = Structure
Struct: CType = Structure

types: CType = CType._types_
ctype: CType = CType

__version__: str = '7.4.2'
__author__:  str = 'notepads'
__github__:  str = 'https://github.com/notepads'
__repo__ :   str = 'https://github.com/notepads/ctype'
__license__: str = 'MIT'
__copyright__: str = 'Copyright (c) 2024 notepads'