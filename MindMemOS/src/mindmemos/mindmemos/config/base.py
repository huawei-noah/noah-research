from dataclasses import field, fields, is_dataclass
from typing import Any, TypeVar, cast

from omegaconf import DictConfig, OmegaConf

T = TypeVar("T")
FROZEN_KEY = "frozen"
FROZEN_VALUE = True
SECRET_VALUE = True
SECRET_KEY = "secret"
MASK = "*****"


def frozen_field(default: Any = ..., *, secret: bool = False, **kwargs) -> Any:
    return _make_field(default, frozen=FROZEN_VALUE, secret=secret, **kwargs)


def secret_field(default: Any = ..., *, frozen: bool = False, **kwargs) -> Any:
    return _make_field(default, frozen=frozen, secret=SECRET_VALUE, **kwargs)


def _make_field(default, *, frozen=False, secret=False, **kwargs):
    meta = dict(kwargs.pop("metadata", {}))
    if frozen:
        meta[FROZEN_KEY] = FROZEN_VALUE
    if secret:
        meta[SECRET_KEY] = SECRET_VALUE
    if default is ...:
        return field(metadata=meta, **kwargs)
    if callable(default) and not isinstance(default, type):
        return field(default_factory=default, metadata=meta, **kwargs)
    return field(default=default, metadata=meta, **kwargs)


def build(schema: type[T], overrides: Any = None) -> T:
    cfg = OmegaConf.structured(schema)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))
    _apply_frozen_flags(cfg, schema)
    return cast("T", cfg)


def safe_dict(cfg: Any, schema: type | None = None) -> dict:
    if schema is None:
        schema = OmegaConf.get_type(cfg)
    raw = OmegaConf.to_container(cfg, resolve=True)
    _mask_in_place(raw, schema)
    return raw


def _apply_frozen_flags(cfg: DictConfig, schema: type) -> None:
    if not is_dataclass(schema):
        return
    for f in fields(schema):
        if f.metadata.get(FROZEN_KEY):
            cfg._get_node(f.name)._set_flag("readonly", FROZEN_VALUE)
        if is_dataclass(f.type):
            _apply_frozen_flags(getattr(cfg, f.name), f.type)


def _mask_in_place(data: Any, schema: type | None) -> None:
    if not (is_dataclass(schema) and isinstance(data, dict)):
        return
    for f in fields(schema):
        if f.name not in data:
            continue
        if f.metadata.get(SECRET_KEY) and data[f.name] is not None:
            data[f.name] = MASK
        elif is_dataclass(f.type):
            _mask_in_place(data[f.name], f.type)
