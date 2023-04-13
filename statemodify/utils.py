import glob
import os
import pkg_resources
from typing import Union

import yaml

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
from pygame.locals import QUIT, KEYDOWN


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


def select_template_file(basin_name: str,
                         template_file: Union[None, str],
                         extension: Union[None, str] = None) -> str:
    """Select either the default template file or a user provided one.

    :param basin_name:                      Name of basin for either:
                                                Upper_Colorado
                                                Yampa
                                                San_Juan
                                                Gunnison
                                                White
    :type basin_name:                       str

    :param template_file:       If a full path to a template file is provided it will be used.  Otherwise the
                                default template in this package will be used.
    :type template_file:        Union[None, str]

    :param extension:           Extension of the target template file with no dot.
    :type extension:            Union[None, str]

    :return:                    Template file path
    :rtype:                     str

    """

    if template_file is None:

        # get basin abbreviation from basin name
        spec_file = pkg_resources.resource_filename("statemodify", "data/basin_specification.yml")
        basin_spec = yaml_to_dict(spec_file)
        basin_abbrev = basin_spec[basin_name]["abbrev"]

        data_dir = pkg_resources.resource_filename("statemodify", "data")
        return glob.glob(os.path.join(data_dir, f"{basin_abbrev}*.{extension}"))[0]

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


def credits():
    """Run credit reel."""

    credit_list = ["statemodify",
                   " ",
                   "A gift to you",
                   "and yours from:",
                   " ",
                   "Rohini S. Gupta",
                   "and",
                   "Chris R. Vernon"]

    pygame.init()
    pygame.display.set_caption('End credits')
    screen = pygame.display.set_mode((800, 600))
    screen_r = screen.get_rect()
    font = pygame.font.Font(pkg_resources.resource_filename("statemodify", "data/EightBit-Atari-Regular.ttf"), 26)
    music_file = pkg_resources.resource_filename("statemodify", "data/corrina.midi")

    clock = pygame.time.Clock()

    pygame.mixer.music.load(music_file)
    pygame.mixer.music.play()

    background_image = pygame.image.load(pkg_resources.resource_filename("statemodify", "data/background.png"))

    content = []
    for index, line in enumerate(credit_list):
        ft = font.render(line, 1, (255, 255, 255))  # white font

        placement = ft.get_rect(centerx=screen_r.centerx, y=screen_r.bottom + index * 45)
        content.append((placement, ft))

    while True:
        for e in pygame.event.get():
            if e.type == QUIT or e.type == KEYDOWN and e.key == pygame.K_ESCAPE:
                break

        screen.blit(background_image, (0, 0))

        for r, s in content:
            r.move_ip(0, -1)
            screen.blit(s, r)

        if not screen_r.collidelistall([r for (r, _) in content]):
            break

        pygame.display.flip()

        # fps designation
        clock.tick(25)

    pygame.mixer.music.stop()
    pygame.mixer.quit()
    pygame.quit()

    return None
