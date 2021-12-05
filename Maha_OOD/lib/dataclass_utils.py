# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

from argparse import ArgumentParser
from dataclasses import MISSING, fields


def make_parser(cls):
    """Create an argument parser from a dataclass."""
    parser = ArgumentParser()

    defaults = {}
    for field in fields(cls):
        required = (field.default is MISSING
                    and field.default_factory is MISSING)
        arg_type = field.metadata.get("arg_type", field.type)
        if arg_type == bool:
            parser.add_argument("--" + field.name, dest=field.name,
                                action='store_true')
            parser.add_argument('--no-' + field.name, dest=field.name,
                                action='store_false')
            defaults[field.name] = field.default if field.default is not MISSING else True
        else:
            parser.add_argument("--" + field.name,
                                type=arg_type, required=required)
        parser.set_defaults(**defaults)
    return parser


def datacli(cls, argv=None):
    """Parse command line arguments into a 'cls' object."""
    parser = make_parser(cls)
    data = {key: val for key, val in vars(parser.parse_args(argv)).items()
            if val is not None}
    return cls(**data)


def get_dataclass_params(datacls):
    """Extract fields that interest us from dataclass"""
    param_fields = fields(datacls)
    return {f.name: getattr(datacls, f.name) for f in param_fields}
