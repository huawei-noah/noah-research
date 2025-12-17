# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from __future__ import annotations

from typing import (
    Any, ClassVar, Dict, Type, TypeVar
)

from pydantic import BaseModel, ConfigDict
from pydantic_core import core_schema
from typing_extensions import Unpack

from ...logger import get_logger

logger = get_logger()

T = TypeVar("T")


class ComponentFactory:
    _registry: ClassVar[Dict[str, Type[BaseComponent]]] = {}

    @classmethod
    def create(cls, name: str, /, **kwargs) -> BaseComponent:
        """Create a class instance using the given name and kwargs."""
        try:
            component_cls = cls._registry[name]
        except KeyError as e:
            raise ValueError(f"Unknown component '{name}'") from e
        if issubclass(component_cls, BaseModel):
            return component_cls.model_validate(kwargs)
        return component_cls(**kwargs)

    @classmethod
    def register(cls, name: str, component_cls: Type[BaseComponent]) -> None:
        """Register a class into the factory"""
        if name in cls._registry:
            raise ValueError(f"Component name '{name}' already registered")
        cls._registry[name] = component_cls

    @classmethod
    def is_registered(cls, name: str) -> bool:
        return name in cls._registry


class BaseComponent(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    def __init_subclass__(cls, **kwargs: Unpack[ConfigDict]):
        """For any module that inherit BaseComponent will automatically register the class into the factory"""
        super().__init_subclass__(**kwargs)
        name = getattr(cls, "__component_name__", None) or cls.__name__
        ComponentFactory.register(name, cls)


class FactoryTypeAdapter:
    @classmethod
    def __get_pydantic_core_schema__(
            cls,
            source_type,
            handler
    ) -> core_schema.CoreSchema:
        def validate_from_dict(data: Dict[str, Any]) -> BaseComponent:
            if not isinstance(data, dict):
                raise TypeError("Input must be a dict")

            class_name = data.pop("__class_name__", None)
            if not class_name:
                raise ValueError(f"Input must contain key '__class_name__', got {data.keys()}")

            return ComponentFactory.create(class_name, **data)  # type: ignore

        def serialize_to_dict(instance: BaseComponent) -> Dict[str, Any]:
            class_name = instance.__class__.__name__
            instance_data = instance.model_dump()
            return {"__class_name__": class_name, **instance_data}

        from_dict_schema = core_schema.chain_schema([
            core_schema.dict_schema(),
            core_schema.no_info_plain_validator_function(validate_from_dict),
        ])

        return core_schema.json_or_python_schema(
            json_schema=from_dict_schema,
            python_schema=core_schema.union_schema([
                core_schema.is_instance_schema(BaseComponent),
                from_dict_schema,
            ]),
            serialization=core_schema.plain_serializer_function_ser_schema(
                serialize_to_dict,
                when_used='unless-none'
            ),
        )
