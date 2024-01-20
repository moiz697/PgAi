from typing import Any

class CTypeException(Exception):
    '''ctype.CTypeException | builtins.Exception'''
    cbase_message: str = '\n\nctype.CTypeException(`<!__object__!>`, `<!__message__!>`)'

class CTypeRuntime(RuntimeError):
    '''ctype.CTypeRuntime | builtins.RuntimeError'''
    cbase_message: str = '\n\nctype.CTypeRuntime(`<!__object__!>`, `<!__message__!>`)'

class CTypeIndex(IndexError):
    '''ctype.CTypeIndex | builtins.IndexError'''
    cbase_message: str = '\n\nctype.CTypeIndex(`<!__object__!>`, `<!__message__!>`)'

class CTypeName(NameError):
    '''ctype.CTypeName | builtins.NameError'''
    cbase_message: str = '\n\nctype.CTypeName(`<!__object__!>`, `<!__message__!>`)'

class CTypeImport(ImportError):
    '''ctype.CTypeImport | builtins.ImportError'''
    cbase_message: str = '\n\nctype.CTypeImport(`<!__object__!>`, `<!__message__!>`)'

class CTypeAttr(AttributeError):
    '''ctype.CTypeAttr | builtins.AttributeError'''
    cbase_message: str = '\n\nctype.CTypeAttr(`<!__object__!>`, `<!__message__!>`)'

class CTypeKey(KeyError):
    '''ctype.CTypeKey | builtins.KeyError'''
    cbase_message: str = '\n\nctype.CTypeKey(`<!__object__!>`, `<!__message__!>`)'

class CTypeIndent(IndentationError):
    '''ctype.CTypeIndent | builtins.IndentationError'''
    cbase_message: str = '\n\nctype.CTypeIndent(`<!__object__!>`, `<!__message__!>`)'

class CTypeError(TypeError):
    '''ctype.CTypeError | builtins.TypeError'''
    cbase_message: str = '\n\nctype.CTypeError(`<!__object__!>`, `<!__message__!>`)'

class CTypeKeyboard(KeyboardInterrupt):
    '''ctype.CTypeKeyboard | builtins.KeyboardInterrupt'''
    cbase_message: str = '\n\nctype.CTypeKeyboard(`<!__object__!>`, `<!__message__!>`)'

class CTypeUnfinished(NotImplementedError):
    '''ctype.CTypeUnfinished | builtins.NotImplementedError'''
    cbase_message: str = '\n\nctype.CTypeUnfinished(`<!__object__!>`, `<!__message__!>`)'

class CTypeOverflow(OverflowError):
    '''ctype.CTypeOverflow | builtins.OverflowError'''
    cbase_message: str = '\n\nctype.CTypeOverflow(`<!__object__!>`, `<!__message__!>`)'

class CTypeValue(ValueError):
    '''ctype.CTypeValue | builtins.ValueError'''
    cbase_message: str = '\n\nctype.CTypeValue(`<!__object__!>`, `<!__message__!>`)'

def ctype_err(cerr_type: Exception, cobject: Any, cmessage: str) -> Exception:
    cerr_message: str = str(cerr_type.cbase_message).replace('<!__object__!>', f'{cobject}').replace('<!__message__!>', f'{cmessage}')
    return cerr_type(cerr_message)