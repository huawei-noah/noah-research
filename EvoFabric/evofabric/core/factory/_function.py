# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
import base64
import importlib
import threading
from typing import Any, Callable, List, Optional, Protocol
from ...logger import get_logger


logger = get_logger()


DESERIALIZER_MODULES = [
    "evofabric.logger",
    "evofabric.core.agent",
    "evofabric.core.clients",
    "evofabric.core.factory",
    "evofabric.core.graph",
    "evofabric.core.mem",
    "evofabric.core.multi_agent",
    "evofabric.core.tool",
    "evofabric.core.trace",
    "evofabric.core.typing",
    "evofabric.core.vectorstore"
]


def register_deserialize_modules(modules: List[str]) -> None:
    """Register some modules that deserialization may need"""
    global DESERIALIZER_MODULES
    for module in modules:
        DESERIALIZER_MODULES.append(module)


class FunctionSerializerProto(Protocol):
    """Defines serialization and deserialization methods for Python function handles in the module,
     used for storing and reloading DSL files.
    """

    def serialize(self, obj: Any) -> str: ...

    def deserialize(self, s: str) -> Any: ...


class _FunctionSerializerProxy:
    __slots__ = ("_impl", "_lock")

    def __init__(self, impl: FunctionSerializerProto):
        self._impl = impl
        self._lock = threading.RLock()

    def serialize(self, obj: Any) -> str:
        with self._lock:
            return self._impl.serialize(obj)

    def deserialize(self, s: str) -> Any:
        with self._lock:
            return self._impl.deserialize(s)

    def _swap(self, new_impl: FunctionSerializerProto) -> None:
        with self._lock:
            self._impl = new_impl


class FunctionSerializerCloudPickle(FunctionSerializerProto):
    def __init__(self):
        try:
            from cloudpickle import dumps, loads
        except ImportError as e:
            raise ImportError(
                "Cannot find package named cloudpickle, "
                "use `pip install cloudpickle` to install it.") from e

        self.dumps = dumps
        self.loads = loads

    def serialize(self, function: Callable) -> str:
        pickled_bytes = self.dumps(function)
        base64_bytes = base64.b64encode(pickled_bytes)
        base64_string = base64_bytes.decode('ascii')
        return base64_string

    def deserialize(self, string: str, required_modules: Optional[List[str]] = None) -> Callable:
        """
        Deserializes a string into a function.

        Args:
            string: The serialized function string.
            required_modules: A list of module names to import before deserialization.
                              e.g., ['my_project.my_classes', 'another.module']
        """
        required_modules = required_modules or DESERIALIZER_MODULES
        if required_modules:
            for module_name in required_modules:
                try:
                    importlib.import_module(module_name)
                except ImportError as e:
                    logger.warning(f"Warning: Could not import required module '{module_name}'. {e}")

        restored_base64_bytes = string.encode('utf-8')
        restored_string = base64.b64decode(restored_base64_bytes)
        return self.loads(restored_string)


FUNC_SERIALIZER: _FunctionSerializerProxy = _FunctionSerializerProxy(FunctionSerializerCloudPickle())


def set_func_serializer(impl: Optional[FunctionSerializerProto]) -> None:
    if impl is None:
        impl = FunctionSerializerCloudPickle()
    FUNC_SERIALIZER._swap(impl)


def get_func_serializer() -> FunctionSerializerProto:
    return FUNC_SERIALIZER
