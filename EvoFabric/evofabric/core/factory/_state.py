# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
import importlib
import json
import typing
from typing import Annotated, Any, Dict, get_args, get_origin, get_type_hints, Type, Union

from pydantic import BaseModel, create_model, field_serializer, field_validator
from pydantic_core import PydanticUndefined
from typing_extensions import Annotated, TypedDict

from ._utils import is_basemodel, is_typeddict
from ..typing import StateSchema


def dump_schema_annotated_info(schema: Type[Union[BaseModel, Dict]]) -> Dict:
    """Convert a base model type to json
    Args:
        schema (type of basemodel or typeddict): Schema definition

    Returns:
        Json of schem define
    """
    is_type_base_model = is_basemodel(schema)
    is_type_typed_dict = is_typeddict(schema)
    if not isinstance(schema, type) or not (is_type_base_model or is_type_typed_dict):
        raise ValueError(f"Invalid type of schema, got {schema}, must be BaseModel or TypedDict")

    schema_info = {
        'name': schema.__name__,
        'type': 'BaseModel' if is_type_base_model else 'TypedDict',
        'fields': {}
    }

    try:
        annotations = get_type_hints(schema, include_extras=True)
    except (AttributeError, TypeError, NameError):
        annotations = getattr(schema, '__annotations__', {})

    for name, type_hint in annotations.items():
        field_info = {}
        origin = get_origin(type_hint)
        args = get_args(type_hint)

        default_value = None
        if is_type_base_model:
            schema: type[BaseModel]
            default_value = schema.model_fields[name].default
            if default_value is PydanticUndefined:
                default_value = None

        if origin is Annotated:
            base_type = args[0]
            metadata = args[1:]

            base_type_origin = get_origin(base_type) or base_type

            field_info['type_module'] = base_type_origin.__module__
            field_info['type_name'] = base_type_origin.__name__
            field_info['metadata'] = metadata
        else:
            type_origin = get_origin(type_hint) or type_hint
            field_info['type_module'] = type_origin.__module__
            field_info['type_name'] = type_origin.__name__

        if default_value:
            field_info['default'] = default_value

        schema_info['fields'][name] = field_info

    return schema_info


def load_schema_annotated_info(schema_info: Dict) -> Type[Union[BaseModel, Dict]]:
    """
    Load json and convert to a basemodel type

    Args:
        schema_info (dict): Schema info

    Returns:
        BaseModel or TypedDict
    """
    class_name = schema_info['name']
    class_type = schema_info['type']
    fields_for_creation = {}
    for name, info in schema_info['fields'].items():
        try:
            module = importlib.import_module(info['type_module'])
            base_type = getattr(module, info['type_name'])
        except (ImportError, AttributeError):
            if info['type_module'] == 'builtins':
                base_type = getattr(typing, info['type_name'].capitalize(), getattr(__builtins__, info['type_name']))
            else:
                raise
        metadata = tuple(info.get('metadata', []))

        if metadata:
            field_type = Annotated[(base_type,) + tuple(metadata)]
        else:
            field_type = base_type

        if 'default' in info:
            fields_for_creation[name] = (field_type, info['default'])
        else:
            fields_for_creation[name] = (field_type, ...)

    if class_type == 'BaseModel':
        return create_model(class_name, **fields_for_creation)
    elif class_type == 'TypedDict':
        type_dict = {k: v[0] for k, v in fields_for_creation.items()}
        return TypedDict(class_name, type_dict)

    raise TypeError(f"Unsupported schema type: {class_type}")


class StateSchemaSerializable:
    @field_validator("state_schema", mode="before")
    @classmethod
    def _deserialize_state_schema(cls, v: Any) -> type:
        if isinstance(v, str):
            data = json.loads(v)
            return load_schema_annotated_info(data)
        elif isinstance(v, dict):
            return load_schema_annotated_info(v)
        elif isinstance(v, type):
            return v
        raise TypeError(
            "state_schema must be either a JSON-string or a type (BaseModel / TypedDict)"
        )

    @field_serializer("state_schema", when_used="json")
    def _serialize_state_schema(self, schema_cls: type[StateSchema]) -> str:
        return json.dumps(dump_schema_annotated_info(schema_cls))
