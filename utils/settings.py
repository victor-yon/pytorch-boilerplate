import argparse
from dataclasses import dataclass, asdict
from typing import Union

import configargparse

from utils.logger import logger


@dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False)
class Settings:
    """
    Storing all settings for this program with default values.
    Setting are loaded from (last override first):
        - default values (in this file)
        - local file (default: ./settings.yaml)
        - environment variables
        - arguments of the command line (with "--" in front)

    TODO create a parent class to wrap all the logic
    """

    run_name: str = ''

    logger_output_level: Union[str, int] = 'INFO'
    show_images: bool = True

    nb_classes: int = 4
    train_point_per_class: int = 200
    test_point_per_class: int = 50

    batch_size: int = 4
    nb_epoch: int = 4

    def validate(self):
        """
        Validate settings.
        """
        # TODO automatically check type based on type hint
        # TODO check if run_name have valid character for a file
        assert self.logger_output_level in ['CRITICAL', 'FATAL', 'ERROR', 'WARN', 'WARNING', 'INFO', 'DEBUG',
                                            'NOTSET'] or isinstance(self.logger_output_level, int), 'Invalid log level'

        assert self.nb_classes > 0, 'At least one class is required'
        assert self.train_point_per_class > 0, 'At least one training point is required'
        assert self.test_point_per_class > 0, 'At least one testing point is required'

        assert self.batch_size > 0, 'Batch size should be a positive integer'
        assert self.nb_epoch > 0, 'Number of epoch should be at least 1'

    def _load_file_and_cmd(self) -> None:
        """
        Load settings from local file and arguments of the command line.
        """

        def str_to_bool(arg_value: str) -> bool:
            """
            Used to handle boolean settings.
            If not the 'bool' type convert all not empty string as true.

            :param arg_value: The boolean value as a string.
            :return: The value parsed as a string.
            """
            if isinstance(arg_value, bool):
                return arg_value
            if arg_value.lower() in {'false', 'f', '0', 'no', 'n'}:
                return False
            elif arg_value.lower() in {'true', 't', '1', 'yes', 'y'}:
                return True
            raise argparse.ArgumentTypeError(f'{arg_value} is not a valid boolean value')

        p = configargparse.get_argument_parser(default_config_files=['./settings.yaml'])

        # Spacial argument
        p.add_argument('-s', '--settings', required=False, is_config_file=True,
                       help='path to custom configuration file')

        # Create argument for each attribute of this class
        for name, value in asdict(self).items():
            p.add_argument(f'--{name.replace("_", "-")}',
                           f'--{name}',
                           dest=name,
                           required=False,
                           type=str_to_bool if type(value) == bool else type(value))

        # TODO deal with unknown arguments with a warning
        # Load arguments form file, environment, command line to override the defaults
        for name, value in vars(p.parse_args()).items():
            if name == 'settings':
                continue
            if value is not None:
                # Directly set the value to bypass the "__setattr__" function
                self.__dict__[name] = value

        self.validate()

    def __post_init__(self):
        """
        Called after the init of this object.
        """
        self._load_file_and_cmd()

    def __setattr__(self, name, value) -> None:
        """
        Set an attribute and valide the new value.

        :param name: The name of the attribut
        :param value: The value of the attribut
        """
        logger.info(f'Setting "{name}" changed from "{getattr(self, name)}" to "{value}".')
        self.__dict__[name] = value
        self.validate()

    def __delattr__(self, name):
        raise AttributeError('Removing a setting is forbidden for the sake of consistency.')

    def __str__(self) -> str:
        """
        :return: Human readable description of the settings.
        """
        return 'Settings:\n\t' + \
               '\n\t'.join([f'{name}: {str(value)}' for name, value in asdict(self).items()])


# Singleton setting object
settings = Settings()
