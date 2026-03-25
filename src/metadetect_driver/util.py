import importlib

# https://packaging.python.org/en/latest/specifications/entry-points/#data-model
def from_entrypoint(object_ref):
    modname, qualname_separator, qualname = object_ref.partition(':')
    obj = importlib.import_module(modname)
    if qualname_separator:
        for attr in qualname.split('.'):
            obj = getattr(obj, attr)

    return obj
