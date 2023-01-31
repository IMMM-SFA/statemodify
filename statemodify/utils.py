import pkg_resources
from typing import Union

import yaml


def yaml_to_dict(yaml_file: str) -> dict:
    """Read in a YAML file and convert to a typed dictionary.
    NOTE:  code can be executed from the YAML file due to the use of UnsafeLoader.

    :param yaml_file:               Full path with file name and extension to the input YAML file.
    :type yaml_file:                str

    :return:                        Dictionary of typed elements of the YAML file.
    :rtype:                         dict

    """

    with open(yaml_file, "r") as yml:
        return yaml.load(yml, Loader=yaml.UnsafeLoader)


def select_template_file(template_file: Union[None, str],
                         extension: Union[None, str] = None) -> str:
    """Select either the default template file or a user provided one.

    :param template_file:       If a full path to a template file is provided it will be used.  Otherwise the
                                default template in this package will be used.
    :type template_file:        Union[None, str]

    :param extension:           Extension of the target template file with no dot.
    :type extension:            Union[None, str]

    :return:                    Template file path
    :rtype:                     str

    """

    if template_file is None:
        return pkg_resources.resource_filename("statemodify", f"data/template.{extension}")
    else:
        return template_file


def select_data_specification_file(yaml_file: Union[None, str],
                                   extension: Union[None, str] = None) -> str:
    """Select either the default template file or a user provided one.

    :param yaml_file:           If a full path to a YAML file is provided it will be used.  Otherwise the
                                default file in this package will be used.
    :type yaml_file:            Union[None, str]

    :param extension:           Extension of the target template file with no dot.
    :type extension:            Union[None, str]

    :return:                    Template file path
    :rtype:                     str

    """

    if yaml_file is None:
        return pkg_resources.resource_filename("statemodify", f"data/{extension}_data_specification.yml")
    else:
        return yaml_file