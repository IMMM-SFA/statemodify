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
