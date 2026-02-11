import argparse
import dataclasses
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Tuple, Union, cast, List

from omegaconf import OmegaConf

DataClass = Any
DataClassType = Any


@dataclass
class ConfigBase:
    """Base class that should handle parsing from command line,
    json, dicts.
    """

    @classmethod
    def parse_from_command_line(cls):
        return omegaconf_parse(cls)

    @classmethod
    def parse_from_file(cls, path: str):
        oc = OmegaConf.load(path)
        return cls.parse_from_dict(OmegaConf.to_container(oc))

    @classmethod
    def parse_from_command_line_deprecated(cls):
        result = DataclassArgParser(
            cls, fromfile_prefix_chars="@"
        ).parse_args_into_dataclasses()
        if len(result) > 1:
            raise RuntimeError(
                f"The following arguments were not recognized: {result[1:]}"
            )
        return result[0]

    @classmethod
    def parse_from_dict(cls, inputs):
        return DataclassArgParser._populate_dataclass_from_dict(cls, inputs.copy())

    @classmethod
    def parse_from_flat_dict(cls, inputs):
        return DataclassArgParser._populate_dataclass_from_flat_dict(cls, inputs.copy())

    def save(self, path: str):
        with open(path, "w") as f:
            OmegaConf.save(config=self, f=f)


class DataclassArgParser(argparse.ArgumentParser):
    """A class for handling dataclasses and argument parsing.
    Closely based on Hugging Face's HfArgumentParser class,
    extended to support recursive dataclasses.
    """

    def __init__(
        self,
        dataclass_types: Union[DataClassType, Iterable[DataClassType]],
        **kwargs,
    ):
        """
        Args:
            dataclass_types:
                Dataclass type, or list of dataclass types for which we will
                "fill" instances with the parsed args.
            kwargs:
                (Optional) Passed to `argparse.ArgumentParser()` in the regular
                way.
        """
        super().__init__(**kwargs)
        if dataclasses.is_dataclass(dataclass_types):
            dataclass_types = cast(DataClassType, dataclass_types)
            dataclass_types = [dataclass_types]
        self.dataclass_types = cast(Iterable[DataClassType], dataclass_types)
        for dtype in self.dataclass_types:
            self._add_dataclass_arguments(dtype)

    def _add_dataclass_arguments(self, dtype: DataClassType):
        for f in dataclasses.fields(dtype):
            field_name = f"--{f.name}"
            kwargs = dict(f.metadata).copy()
            typestring = str(f.type)
            for x in (int, float, str):
                if typestring == f"typing.Union[{x.__name__}, NoneType]":
                    f.type = x
            if isinstance(f.type, type) and issubclass(f.type, Enum):
                kwargs["choices"] = list(f.type)
                kwargs["type"] = f.type
                if f.default is not dataclasses.MISSING:
                    kwargs["default"] = f.default
            elif f.type is bool:
                kwargs["action"] = "store_false" if f.default is True else "store_true"
                if f.default is True:
                    field_name = f"--no-{f.name}"
                    kwargs["dest"] = f.name
            elif dataclasses.is_dataclass(f.type):
                self._add_dataclass_arguments(f.type)
            else:
                kwargs["type"] = f.type
                if f.default is not dataclasses.MISSING:
                    kwargs["default"] = f.default
                else:
                    kwargs["required"] = True
            self.add_argument(field_name, **kwargs)

    def parse_args_into_dataclasses(
        self,
        args=None,
    ) -> Tuple[DataClass, ...]:
        """
        Parse command-line args into instances of the specified dataclass
        types.  This relies on argparse's `ArgumentParser.parse_known_args`.
        See the doc at:
        docs.python.org/3.7/library/argparse.html#argparse.ArgumentParser.parse_args
        Args:
            args:
                List of strings to parse. The default is taken from sys.argv.
                (same as argparse.ArgumentParser)
        Returns:
            Tuple consisting of:
                - the dataclass instances in the same order as they
                  were passed to the initializer.abspath
                - if applicable, an additional namespace for more
                  (non-dataclass backed) arguments added to the parser
                  after initialization.
                - The potential list of remaining argument strings.
                  (same as argparse.ArgumentParser.parse_known_args)
        """
        namespace, unknown = self.parse_known_args(args=args)
        outputs = []

        for dtype in self.dataclass_types:
            outputs.append(self._populate_dataclass(dtype, namespace))
        if len(namespace.__dict__) > 0:
            # additional namespace.
            outputs.append(namespace)
        if len(unknown) > 0:
            outputs.append(unknown)
        return tuple(outputs)

    @staticmethod
    def _populate_dataclass(dtype: DataClassType, namespace: argparse.Namespace):
        keys = {f.name for f in dataclasses.fields(dtype)}
        inputs = {k: v for k, v in vars(namespace).items() if k in keys}
        for k in keys:
            delattr(namespace, k)
        sub_dataclasses = {
            f.name: f.type
            for f in dataclasses.fields(dtype)
            if dataclasses.is_dataclass(f.type)
        }
        for k, s in sub_dataclasses.items():
            inputs[k] = DataclassArgParser._populate_dataclass(s, namespace)
        obj = dtype(**inputs)
        return obj

    @staticmethod
    def _populate_dataclass_from_dict(dtype: DataClassType, d: dict):
        d = DataclassArgParser.legacy_transform_dict(d.copy())
        keys = {f.name for f in dataclasses.fields(dtype)}
        inputs = {k: v for k, v in d.items() if k in keys}
        for k in keys:
            if k in d:
                del d[k]
        sub_dataclasses = {
            f.name: f.type
            for f in dataclasses.fields(dtype)
            if dataclasses.is_dataclass(f.type)
        }
        for k, s in sub_dataclasses.items():
            if k not in inputs:
                v = {}
            else:
                v = inputs[k]
            inputs[k] = DataclassArgParser._populate_dataclass_from_dict(s, v)
        obj = dtype(**inputs)
        return obj

    @staticmethod
    def _populate_dataclass_from_flat_dict(dtype: DataClassType, d: dict):
        d = DataclassArgParser.legacy_transform_dict(d.copy())
        keys = {f.name for f in dataclasses.fields(dtype)}
        inputs = {k: v for k, v in d.items() if k in keys}
        for k in keys:
            if k in d:
                del d[k]
        sub_dataclasses = {
            f.name: f.type
            for f in dataclasses.fields(dtype)
            if dataclasses.is_dataclass(f.type)
        }
        for k, s in sub_dataclasses.items():
            inputs[k] = DataclassArgParser._populate_dataclass_from_dict(s, d)
        obj = dtype(**inputs)
        return obj

    @staticmethod
    def legacy_transform_dict(d: dict):
        """Transforms the dictionary to an older version of the dataclasses"""
        key_mapping = {
            "training_config": "training",
            "model_config": "model",
            "cost_config": "cost",
        }
        nd = {}
        for k in d:
            if k in key_mapping:
                nd[key_mapping[k]] = d[k]
            else:
                nd[k] = d[k]
        return nd


def omegaconf_parse(cls):
    parser = argparse.ArgumentParser(fromfile_prefix_chars="@")
    parser.add_argument(
        "--configs",
        nargs="*",
        default=[],
        help="Configs to load",
    )
    parser.add_argument(
        "--values",
        nargs="*",
        default=[],
        help="Dot values to change configs",
    )
    args, _unknown = parser.parse_known_args()

    return omegaconf_parse_files_vals(cls, args.configs, args.values)


def omegaconf_parse_files_vals(cls, files_paths: List[str], dotlist: List[str]):
    configs = [OmegaConf.structured(cls)]
    for path in files_paths:
        configs.append(OmegaConf.load(path))
    configs.append(OmegaConf.from_dotlist(dotlist))
    omega_config = OmegaConf.merge(*configs)
    res = cls.parse_from_dict(OmegaConf.to_container(omega_config))
    return res


def combine_cli_dict(cls, c_dict):
    """A function to load cli configs and merge them with a dictionary"""
    config_base = cls.parse_from_command_line()
    return combine_dataclass_dict(config_base, c_dict)


def combine_dataclass_dict(dcls, c_dict):
    """Combines the parameters in an instantiated dataclass with the dictionary."""
    config = OmegaConf.create(dataclasses.asdict(dcls))
    for k, v in c_dict.items():
        OmegaConf.update(config, k, v)
    return dcls.parse_from_dict(OmegaConf.to_container(config))
