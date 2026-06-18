import importlib


# https://packaging.python.org/en/latest/specifications/entry-points/#data-model
def from_entrypoint(object_ref):
    """
    Get the running method for metadetect from a string. With the format
    "module.submodule:object.attr" where the module is importable and the object is an attribute of the
    module.

    Parameters
    ----------
    object_ref : str
        A string of the form "module.submodule:object.attr" where the module is
        importable and the object is an attribute of the module.

    Returns
    -------
    object
        The object referenced by the string.
    """
    modname, qualname_separator, qualname = object_ref.partition(":")
    obj = importlib.import_module(modname)
    if qualname_separator:
        for attr in qualname.split("."):
            obj = getattr(obj, attr)

    return obj
