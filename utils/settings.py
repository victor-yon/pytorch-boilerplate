import argparse
import re
from dataclasses import asdict, dataclass
from typing import Sequence, Union

import configargparse
from numpy.distutils.misc_util import is_sequence

from utils.logger import logger


# TODO create a parent class to wrap all the logic
@dataclass(init=False, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False)
class Settings:
    """
    Storing all settings for this program with default values.
    Setting are loaded from (last override first):
        - default values (in this file)
        - local file (default path: ./settings.yaml)
        - environment variables
        - arguments of the command line (with "--" in front)
    """

    # ==================================================================================================================
    # ==================================================== General =====================================================
    # ==================================================================================================================

    # Name of the run to save the result ('tmp' for temporary files).
    # If empty or None thing is saved.
    run_name: str = ''

    # The seed to use for all random number generator during this run.
    seed: int = 42  # FIXME allow to set is as None from args or settings file

    # ==================================================================================================================
    # ============================================== Logging and Outputs ===============================================
    # ==================================================================================================================

    # The minimal logging level to show in the console (see https://docs.python.org/3/library/logging.html#levels).
    logger_console_level: Union[str, int] = 'INFO'

    # The minimal logging level to write in the log file (see https://docs.python.org/3/library/logging.html#levels).
    logger_file_level: Union[str, int] = 'DEBUG'

    # If True a log file is created for each run with a valid run_name.
    # The console logger could be enable at the same time.
    # If False the logging will only be in console.
    logger_file_enable: bool = True

    # If True use a visual progress bar in the console during training and loading.
    # Should be use with a logger_console_level as INFO or more for better output.
    visual_progress_bar: bool = True

    # The console logging refresh time during training and testing (used only if visual_progress_bar is False).
    # Value in second.
    # TODO use logger_progress_frequency
    logger_progress_frequency: int = 10

    # If True show matplotlib images when they are ready.
    show_images: bool = True

    # If True and the run have a valid name, save matplotlib images in the run directory
    save_images: bool = True

    # If True and the run have a valid name, save the neural network parameters in the run directory at the end of the
    # training. Saved before applying early stopping if enable.
    # The file will be at the root of run directory, under then name: "final_network.pt"
    save_network: bool = True

    # ==================================================================================================================
    # ==================================================== Dataset =====================================================
    # ==================================================================================================================

    # The number of classes generated in the mock dataset
    nb_classes: int = 4

    # The number of train data generated in the mock dataset
    train_point_per_class: int = 2000

    # The number of test data generated in the mock dataset
    test_point_per_class: int = 500

    # The number of validation data generated in the mock dataset
    validation_point_per_class: int = 500

    # ==================================================================================================================
    # ==================================================== Networks ====================================================
    # ==================================================================================================================

    # The number hidden layer and their respective number of neurons
    hidden_layers_size: Sequence = (50, 50)

    # ==================================================================================================================
    # ==================================================== Training ====================================================
    # ==================================================================================================================

    # If a valid path to a file containing neural network parameters is set, they will be loaded in the current neural
    # network and the training step will be skipped.
    trained_network_cache_path: str = ''

    # The pytorch device to use for training and testing. Can be 'cpu', 'cuda' or 'auto'.
    # The automatic setting will use CUDA is a compatible hardware is detected.
    device: str = 'auto'

    # The learning rate value used by the SGD for parameters update.
    learning_rate: float = 0.001

    # The momentum value used by the SGD for parameters update.
    momentum: float = 0.9

    # The size of the mini-batch for the training and testing.
    batch_size: int = 128

    # The number of training epoch.
    nb_epoch: int = 20

    # Save the best network state during the training based on the test accuracy.
    # Then load it when the training is complet.
    # The file will be at the root of run directory, under then name: "best_network.pt"
    # Required checkpoint_train_size > 0 and checkpoint_test_size > 0
    early_stopping: bool = True

    # ==================================================================================================================
    # ================================================== Checkpoints ===================================================
    # ==================================================================================================================

    # The number of checkpoints per training epoch, if 0 no checkpoint is processed
    checkpoints_per_epoch: int = 1

    # The number of data in the checkpoint training subset.
    # Set to 0 to don't compute the train accuracy during checkpoints.
    checkpoint_train_size: int = 1280

    # If the inference accuracy of the validation dataset should be computed, or not, during checkpoint.
    checkpoint_validation: bool = True

    # If True and the run have a valid name, save the neural network parameters in the run directory at each checkpoint.
    checkpoint_save_network: bool = False

    def is_named_run(self) -> bool:
        """ Return True only if the name of the run is set (could be a temporary name). """
        return len(self.run_name) > 0

    def is_unnamed_run(self) -> bool:
        """ Return True only if the name of the run is NOT set. """
        return len(self.run_name) == 0

    def is_temporary_run(self) -> bool:
        """ Return True only if the name of the run is set and is temporary name. """
        return self.run_name == 'tmp'

    def is_saved_run(self) -> bool:
        """ Return True only if the name of the run is set and is NOT temporary name. """
        return self.is_named_run() and not self.is_temporary_run()

    def validate(self):
        """
        Validate settings values, to assure integrity and prevent issues during the run.
        """
        # TODO automatically check type based on type hint

        # General
        assert self.run_name.strip() == self.run_name, 'The run name can\' start or end by a white character'
        assert self.run_name is None or not re.search('[/:"*?<>|\\\\]+', self.run_name), \
            'Invalid character in run name (should be a valid directory name)'

        # Logging and Outputs
        possible_log_levels = ('CRITICAL', 'FATAL', 'ERROR', 'WARN', 'WARNING', 'INFO', 'DEBUG', 'NOTSET')
        assert self.logger_console_level.upper() in possible_log_levels or isinstance(self.logger_console_level, int), \
            f"Invalid console log level '{self.logger_console_level}'"
        assert self.logger_file_level.upper() in possible_log_levels or isinstance(self.logger_file_level, int), \
            f"Invalid file log level '{self.logger_file_level}'"

        # Dataset
        assert self.nb_classes > 0, 'At least one class is required'
        assert self.train_point_per_class > 0, 'At least one training point is required'
        assert self.test_point_per_class > 0, 'At least one testing point is required'

        # Networks
        assert all((a > 0 for a in self.hidden_layers_size)), 'Hidden layer size should be more than 0'

        # Training
        # TODO should also accept "cuda:1" format
        assert self.device in ('auto', 'cpu', 'cuda'), f'Not valid torch device name: {self.device}'
        assert self.batch_size > 0, 'Batch size should be a positive integer'
        assert self.nb_epoch > 0, 'Number of epoch should be at least 1'

        # Checkpoints
        assert self.checkpoints_per_epoch >= 0, 'The number of checkpoints should be >= 0'

    def __init__(self):
        """
        Create the setting object.
        """
        self._load_file_and_cmd()

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

        def type_mapping(arg_value):
            if type(arg_value) == bool:
                return str_to_bool
            if is_sequence(arg_value):
                if len(arg_value) == 0:
                    return str
                else:
                    return type_mapping(arg_value[0])

            # Default same as current value
            return type(arg_value)

        p = configargparse.get_argument_parser(default_config_files=['./settings.yaml'])

        # Spacial argument
        p.add_argument('-s', '--settings', required=False, is_config_file=True,
                       help='path to custom configuration file')

        # Create argument for each attribute of this class
        # TODO create automatic helper with doc string or annotation
        for name, value in asdict(self).items():
            p.add_argument(f'--{name.replace("_", "-")}',
                           f'--{name}',
                           dest=name,
                           required=False,
                           action='append' if is_sequence(value) else 'store',
                           type=type_mapping(value))

        # TODO deal with unknown arguments with a warning
        # Load arguments form file, environment and command line to override the defaults
        for name, value in vars(p.parse_args()).items():
            if name == 'settings':
                continue
            if value is not None:
                # Directly set the value to bypass the "__setattr__" function
                self.__dict__[name] = value

        self.validate()

    def __setattr__(self, name, value) -> None:
        """
        Set an attribute and valide the new value.

        :param name: The name of the attribut
        :param value: The value of the attribut
        """
        logger.debug(f'Setting "{name}" changed from "{getattr(self, name)}" to "{value}".')
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
