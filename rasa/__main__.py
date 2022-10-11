import argparse
import logging
import os
import platform
import sys

from rasa_sdk import __version__ as rasa_sdk_version
from rasa.constants import MINIMUM_COMPATIBLE_VERSION

import rasa.telemetry
import rasa.utils.io
import rasa.utils.tensorflow.environment as tf_env
from rasa import version
from rasa.cli import (
    data,
    export,
    interactive,
    run,
    scaffold,
    shell,
    telemetry,
    test,
    train,
    visualize,
    x,
    evaluate,
)
from rasa.cli.arguments.default_arguments import add_logging_options
from rasa.cli.utils import parse_last_positional_argument_as_model_path
from rasa.shared.exceptions import RasaException
from rasa.shared.utils.cli import print_error
from rasa.utils.common import configure_logging_and_warnings

logger = logging.getLogger(__name__)

#yd。创建命令参数解析器，为了可以执行rasa init、 rasa run、rasa shell等命令
def create_argument_parser() -> argparse.ArgumentParser:
    """Parse all the command line arguments for the training script."""

    parser = argparse.ArgumentParser(
        prog="rasa",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Rasa command line interface. Rasa allows you to build "
        "your own conversational assistants 🤖. The 'rasa' command "
        "allows you to easily run most common commands like "
        "creating a new bot, training or evaluating models.",
    )

    parser.add_argument(
        "--version",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Print installed Rasa version",
    ) #yd。创建一条子命令 rasa --version

    parent_parser = argparse.ArgumentParser(add_help=False)
    add_logging_options(parent_parser)
    parent_parsers = [parent_parser]

    subparsers = parser.add_subparsers(help="Rasa commands")

    #yd。为了子命令(如 rasa init、 rasa run、rasa shell等)而创建的子解析器
    scaffold.add_subparser(subparsers, parents=parent_parsers) #调用rasa/cli/scaffold.py中的add_subparser()方法，对应命令"rasa init"
    run.add_subparser(subparsers, parents=parent_parsers) #yd。调用rasa/cli/run.py中的add_subparser()方法，对应命令"rasa run"
    shell.add_subparser(subparsers, parents=parent_parsers) #yd。调用rasa/cli/shell.py中的add_subparser()方法，对应命令"rasa shell"
    train.add_subparser(subparsers, parents=parent_parsers)#yd。调用rasa/cli/train.py中的add_subparser()方法，对应命令"rasa train"
    interactive.add_subparser(subparsers, parents=parent_parsers)#yd。调用rasa/cli/interactive.py中的add_subparser()方法，对应命令"rasa interactive"
    telemetry.add_subparser(subparsers, parents=parent_parsers)#yd。调用rasa/cli/telemetry.py中的add_subparser()方法，对应命令"rasa telemetry"
    test.add_subparser(subparsers, parents=parent_parsers) #yd。调用rasa/cli/test.py中的add_subparser()方法，对应命令"rasa test"
    visualize.add_subparser(subparsers, parents=parent_parsers)#yd。调用rasa/cli/visualize.py中的add_subparser()方法，对应命令"rasa visualize"
    data.add_subparser(subparsers, parents=parent_parsers)
    export.add_subparser(subparsers, parents=parent_parsers)
    x.add_subparser(subparsers, parents=parent_parsers)
    evaluate.add_subparser(subparsers, parents=parent_parsers)

    return parser


def print_version() -> None:
    """Prints version information of rasa tooling and python."""
    print(f"Rasa Version      :         {version.__version__}")
    print(f"Minimum Compatible Version: {MINIMUM_COMPATIBLE_VERSION}")
    print(f"Rasa SDK Version  :         {rasa_sdk_version}")
    print(f"Python Version    :         {platform.python_version()}")
    print(f"Operating System  :         {platform.platform()}")
    print(f"Python Path       :         {sys.executable}")


def main() -> None:
    """Run as standalone python application."""
    parse_last_positional_argument_as_model_path()
    arg_parser = create_argument_parser()
    cmdline_arguments = arg_parser.parse_args()

    log_level = getattr(cmdline_arguments, "loglevel", None)
    configure_logging_and_warnings(
        log_level, warn_only_once=True, filter_repeated_logs=True
    )

    tf_env.setup_tf_environment()
    tf_env.check_deterministic_ops()

    # insert current path in syspath so custom modules are found
    sys.path.insert(1, os.getcwd())

    try:
        if hasattr(cmdline_arguments, "func"):
            rasa.utils.io.configure_colored_logging(log_level)
            rasa.telemetry.initialize_telemetry()
            rasa.telemetry.initialize_error_reporting()

            print(f"yd。开始执行具体的rasa命令")
            cmdline_arguments.func(cmdline_arguments)
        elif hasattr(cmdline_arguments, "version"):
            print_version()
        else:
            # user has not provided a subcommand, let's print the help
            logger.error("No command specified.")
            arg_parser.print_help()
            sys.exit(1)
    except RasaException as e:
        # these are exceptions we expect to happen (e.g. invalid training data format)
        # it doesn't make sense to print a stacktrace for these if we are not in
        # debug mode
        logger.debug("Failed to run CLI command due to an exception.", exc_info=e)
        print_error(f"{e.__class__.__name__}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
