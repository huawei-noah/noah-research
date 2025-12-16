# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
import inspect
from typing import Any, Callable, Dict, get_args, get_origin, Type

from pydantic import BaseModel
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined
from typing_extensions import Annotated, TypedDict

from ..typing import MISSING
from ...logger import get_logger

logger = get_logger()

_TYPE_TO_DEFAULT: Dict[Type[Any], Callable[[], Any]] = {
    list: list,
    dict: dict,
    set: set,
    int: lambda: 0,
    float: lambda: 0,
    str: lambda: "",
    bool: lambda: False,
}


def is_typeddict(tp) -> bool:
    try:
        from typing_extensions import is_typeddict as is_typeddict_ext
    except ImportError:
        is_typeddict_ext = lambda _: False

    return is_typeddict_ext(tp) or (
            inspect.isclass(tp) and issubclass(tp, dict) and hasattr(tp, "__annotations__")
    )


def is_basemodel(typ) -> bool:
    return isinstance(typ, type) and issubclass(typ, BaseModel)


def is_dataclass(typ: type) -> bool:
    return hasattr(typ, "__pydantic_config__")


def strip_annotated(tp):
    return get_args(tp)[0] if get_origin(tp) is Annotated else tp


def deep_dump(obj: Any) -> Any:
    """Recursively convert all values into dict"""
    if isinstance(obj, BaseModel):
        return {k: deep_dump(v) for k, v in obj.model_dump().items()}
    if isinstance(obj, dict):
        return {k: deep_dump(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [deep_dump(i) for i in obj]
    return obj


def _default_by_type(tp: Type[Any]) -> Any:
    if tp in _TYPE_TO_DEFAULT:
        return _TYPE_TO_DEFAULT[tp]()

    origin = get_origin(tp)
    if origin in _TYPE_TO_DEFAULT:
        return _TYPE_TO_DEFAULT[origin]()

    try:
        return tp() if callable(tp) else None
    except Exception:
        logger.warning(f"Unrecognized type {tp}, fallback to None")
        return None


def _smart_default_for_pydantic_field(field: FieldInfo) -> Any:
    """BaseModel.field -> default value"""
    if field.default_factory:
        return field.default_factory()
    if field.default is not PydanticUndefined:
        return field.default
    return _default_by_type(field.annotation)


def _smart_default_for_typed_dict_field(tp: Type[Any]) -> Any:
    """TypedDict annotation -> default value"""
    return _default_by_type(strip_annotated(tp))


def _fill_for_typed_dict(
        cls: type[TypedDict],
        extra: Dict[str, Any],
) -> Dict[str, Any]:
    annotations = cls.__annotations__
    return {
        name: extra.get(name) or _smart_default_for_typed_dict_field(tp)
        for name, tp in annotations.items()
    }


def _fill_for_base_model(
        cls: type[BaseModel],
        extra: Dict[str, Any],
) -> Dict[str, Any]:
    fields = cls.model_fields
    data: Dict[str, Any] = {}

    for name, field in fields.items():
        if name in extra:
            data[name] = extra[name]
        elif field.default is not PydanticUndefined or field.default_factory:
            data[name] = _smart_default_for_pydantic_field(field)
        else:
            data[name] = _default_by_type(field.annotation)

    return data


def fill_defaults(
        model_or_cls: type[BaseModel] | type[TypedDict],
        *,
        extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Fill default values for basemodel or typed dict.
    """
    extra = extra or {}

    if is_typeddict(model_or_cls):
        return _fill_for_typed_dict(model_or_cls, extra)

    if is_basemodel(model_or_cls):
        return _fill_for_base_model(model_or_cls, extra)

    raise TypeError(f"Only BaseModel or TypedDict supported, got {model_or_cls}")


def safe_get_attr(data, attr, default=MISSING):
    if isinstance(data, dict):
        return data.get(attr, default)
    return getattr(data, attr, default)


def safe_set_attr(data, attr, value):
    if isinstance(data, dict):
        data[attr] = value
    else:
        setattr(data, attr, value)


def safe_convert_to_schema(data, schema):
    if isinstance(data, BaseModel):
        data = data.model_dump()
    elif isinstance(data, dict):
        data = data
    else:
        data = {k: getattr(data, k) for k in dir(data) if not k.startswith("_") and not callable(getattr(data, k))}

    if isinstance(schema, type) and issubclass(schema, BaseModel):
        return schema.model_construct(**data)
    return data
