import inspect
import functools
import warnings
warnings.filterwarnings("ignore")


def configurable(init=None, *, from_config=None):
    if init is not None:
        assert (
                inspect.isfunction(init)
                and from_config is None
                and init.__name__ == "__init__"
        ), "Incorrect use of @configurable"

        @functools.wraps(init)
        def wrapped(self, *args, **kwargs):
            try:
                from_config_func = type(self).from_config
            except AttributeError as e:
                raise AttributeError(
                    "Class with @configurable must have a `from_config` classmethod."
                ) from e
            if not inspect.ismethod(from_config_func):
                raise TypeError("Class with @configurable must have a `from_config` classmethod.")

            if _called_with_cfg(*args, **kwargs):
                explicit_args = _get_args_from_config(from_config_func, *args, **kwargs)
                init(self, **explicit_args)
            else:
                init(self, *args, **kwargs)

        return wrapped
    else:
        if from_config is None:
            return configurable

        assert inspect.isfunction(
            from_config
        ), "from_config argument of configurable must be a function!"

        def wrapper(orig_func):
            @functools.wraps(orig_func)
            def wrapped(*args, **kwargs):
                if _called_with_cfg(*args, **kwargs):
                    explicit_args = _get_args_from_config(from_config, *args, **kwargs)
                    return orig_func(**explicit_args)
                else:
                    return orig_func(*args, **kwargs)
            wrapped.from_config = from_config
            return wrapped
        return wrapper


def _called_with_cfg(*args, **kwargs):
    from gnetmdk.config import BaseConfig
    if len(args) and isinstance(args[0], BaseConfig):
        return True
    if isinstance(kwargs.pop("cfg", None), BaseConfig):
        return True
    return False


def _get_args_from_config(from_config_func, *args, **kwargs):
    signature = inspect.signature(from_config_func)
    support_var_arg = any(
        param.kind in [param.VAR_POSITIONAL, param.VAR_KEYWORD]
        for param in signature.parameters.values()
    )
    if support_var_arg:
        ret = from_config_func(*args, **kwargs)
    else:
        support_arg_names = set(signature.parameters.keys())
        extra_kwargs = {}
        for name in list(kwargs.keys()):
            if name not in support_arg_names:
                extra_kwargs[name] = kwargs.pop(name)
        ret = from_config_func(*args, **kwargs)
        ret.update(extra_kwargs)
    return ret


class PostInit(type):

    def __init__(cls, name, base, attr_dict):
        super().__init__(name, base, attr_dict)

        __init__ = attr_dict["__init__"]
        __post_init__ = getattr(cls, "__post_init__", None)

        if __post_init__ is not None and not hasattr(cls, "_init_"):
            setattr(cls, "_init_", __init__)

            def __init_with_post__(self, *args, **kwargs):
                __init__(self, *args, **kwargs)
                __post_init__(self)

            cls.__init__ = __init_with_post__
