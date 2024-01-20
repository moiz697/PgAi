import sys
from .._ctypes.ctype import CType

endl: str = '\n'

@CType
def cout(*args, **kwargs) -> None:
    print(*args, **kwargs)

@CType
def cin(*args, **kwargs) -> str:
    return input(*args, **kwargs)

@CType
def cerr(*args, **kwargs) -> None:
    print(*args, **kwargs, file=sys.stderr)

@CType
def clog(*args, **kwargs) -> None:
    print(*args, **kwargs, file=sys.stdout)

@CType
def cexit(*args, **kwargs) -> None:
    sys.exit(*args, **kwargs)

@CType
def cabort(*args, **kwargs) -> None:
    sys.exit(*args, **kwargs)

@CType
def printf(*args, **kwargs) -> None:
    print(*args, **kwargs)

@CType
def stdout(*args, **kwargs) -> None:
    print(*args, **kwargs, file=sys.stdout)

@CType
def stdin(*args, **kwargs) -> None:
    return input(*args, **kwargs)

@CType
def stderr(*args, **kwargs) -> None:
    print(*args, **kwargs, file=sys.stderr)

@CType
def log(*args, **kwargs) -> None:
    print(*args, **kwargs, file=sys.stdout)

@CType
def exit(*args, **kwargs) -> None:
    sys.exit(*args, **kwargs)

@CType
def abort(*args, **kwargs) -> None:
    sys.exit(*args, **kwargs)