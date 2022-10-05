import argparse
import os
import sys
from typing import List, Text

from rasa import telemetry
from rasa.cli import SubParsersAction
from rasa.cli.shell import shell
from rasa.shared.utils.cli import print_success, print_error_and_exit
from rasa.shared.constants import (
    DOCS_BASE_URL,
    DEFAULT_CONFIG_PATH,
    DEFAULT_DOMAIN_PATH,
    DEFAULT_DATA_PATH,
    DEFAULT_MODELS_PATH,
)


def add_subparser(
    subparsers: SubParsersAction, parents: List[argparse.ArgumentParser]
) -> None:
    """Add all init parsers.

    Args:
        subparsers: subparser we are going to attach to
        parents: Parent parsers, needed to ensure tree structure in argparse
    """
    scaffold_parser = subparsers.add_parser(
        "init",
        parents=parents,
        help="Creates a new project, with example training data, "
        "actions, and config files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    scaffold_parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="Automatically choose default options for prompts and suppress warnings.",
    )
    scaffold_parser.add_argument(
        "--init-dir",
        default=None,
        help="Directory where your project should be initialized.",
    )

    scaffold_parser.set_defaults(func=run)


def print_train_or_instructions(args: argparse.Namespace) -> None:
    """Train a model if the user wants to."""
    import questionary
    import rasa

    print_success("Finished creating project structure.")

    should_train = (
        questionary.confirm("Do you want to train an initial model? 💪🏽")
        .skip_if(args.no_prompt, default=True)
        .ask()
    )

    if should_train:
        print_success("Training an initial model...")
        training_result = rasa.train(
            DEFAULT_DOMAIN_PATH,
            DEFAULT_CONFIG_PATH,
            DEFAULT_DATA_PATH,
            DEFAULT_MODELS_PATH,
        )
        args.model = training_result.model

        print_run_or_instructions(args)

    else:
        print_success(
            "No problem 👍🏼. You can also train a model later by going "
            "to the project directory and running 'rasa train'."
        )


def print_run_or_instructions(args: argparse.Namespace) -> None:
    from rasa.core import constants
    import questionary

    should_run = (
        questionary.confirm(
            "Do you want to speak to the trained assistant on the command line? 🤖"
        )
        .skip_if(args.no_prompt, default=False)
        .ask()
    )

    if should_run:
        # provide defaults for command line arguments
        attributes = [
            "endpoints",
            "credentials",
            "cors",
            "auth_token",
            "jwt_secret",
            "jwt_method",
            "enable_api",
            "remote_storage",
        ]
        for a in attributes:
            setattr(args, a, None)

        args.port = constants.DEFAULT_SERVER_PORT

        shell(args)
    else:
        if args.no_prompt:
            print(
                "If you want to speak to the assistant, "
                "run 'rasa shell' at any time inside "
                "the project directory."
            )
        else:
            print_success(
                "Ok 👍🏼. "
                "If you want to speak to the assistant, "
                "run 'rasa shell' at any time inside "
                "the project directory."
            )


def init_project(args: argparse.Namespace, path: Text) -> None:
    """Inits project."""
    os.chdir(path) #yd。修改当前的工作目录至path所对应的路径，执行了这句后，在执行os.getcwd()，返回的路径即为path对应的值
    print_success(f"当前工作目录已经被切换，新的os.getcwd() = {os.getcwd()}") #yd。提示当前工作目录
    create_initial_project(".") #yd。将"rasa\\cli\\scaffold\\initial_project"下的文件模板拷贝到path所对应的文件夹下，用于初始化rasa project
    print(f"Created project directory at '{os.getcwd()}'.")
    print_train_or_instructions(args)


def create_initial_project(path: Text) -> None:
    """
    yd。功能：将"rasa\\cli\\scaffold\\initial_project"下的文件模板拷贝到path所对应的文件夹下，用于初始化rasa project
    :param path:
    :return:
    """
    """Creates directory structure and templates for initial project."""
    from distutils.dir_util import copy_tree

    copy_tree(scaffold_path(), path)


def scaffold_path() -> Text:
    """
    yd。功能：获取"rasa\\cli\\scaffold\\initial_project"这个路径，因为这个文件夹下保存着初始化一个rasa project所需的文件模板
    :return:
    """
    import pkg_resources

    extracted_scaffold_path = pkg_resources.resource_filename(__name__, "initial_project") #yd。__name__变量的值为当前.py文件的名称，此时它的值为"rasa.cli.scaffold"
    return extracted_scaffold_path

def print_cancel() -> None:
    print_success("Ok. You can continue setting up by running 'rasa init' 🙋🏽‍♀️")
    sys.exit(0)


def _ask_create_path(path: Text) -> None:
    import questionary

    should_create = questionary.confirm(
        f"Path '{path}' does not exist 🧐. Create path?"
    ).ask() #yd。确认是否要创建path对应的路径，默认为True，即需要创建

    if should_create:
        try:
            os.makedirs(path)
        except (PermissionError, OSError, FileExistsError) as e:
            print_error_and_exit(
                f"Failed to create project path at '{path}'. " f"Error: {e}"
            )
    else:
        print_success(
            "Ok, will exit for now. You can continue setting up by "
            "running 'rasa init' again 🙋🏽‍♀️"
        )
        sys.exit(0)


def _ask_overwrite(path: Text) -> None:
    import questionary

    overwrite = questionary.confirm(
        "Directory '{}' is not empty. Continue?".format(os.path.abspath(path))
    ).ask()
    if not overwrite:
        print_cancel()


def run(args: argparse.Namespace) -> None:
    import questionary

    print_success("Welcome to Rasa! 🤖\n") #yd。打印绿色的提示信息
    if args.no_prompt:
        print(
            f"To get started quickly, an "
            f"initial project will be created.\n"
            f"If you need some help, check out "
            f"the documentation at {DOCS_BASE_URL}.\n"
        )
    else:
        print(
            f"To get started quickly, an "
            f"initial project will be created.\n"
            f"If you need some help, check out "
            f"the documentation at {DOCS_BASE_URL}.\n"
            f"Now let's start! 👇🏽\n"
        )

    if args.init_dir is not None:
        path = args.init_dir
    else:
        if 0:
            path = (
                questionary.text(
                    "Please enter a path where the project will be "
                    "created [default: current directory]"
                )
                .skip_if(args.no_prompt, default="")
                .ask()
            ) #yd。获取输入的project_path
            # set the default directory. we can't use the `default` property
            # in questionary as we want to avoid showing the "." in the prompt as the
            # initial value. users tend to overlook it and it leads to invalid
            # paths like: ".C:\mydir".
            # Can't use `if not path` either, as `None` will be handled differently (abort)
            if path == "": #yd。如果没有指定project path，则默认使用当前
                path = "."
        else:
            for_debug_path = "rasa_demo_yd"
            path = (
                questionary.text(
                    "Please enter a path where the project will be "
                    "created [default: {}]".format(for_debug_path)                )
                    .skip_if(args.no_prompt, default="")
                    .ask()
            )  # yd。获取输入的project_path
            # set the default directory. we can't use the `default` property
            # in questionary as we want to avoid showing the "." in the prompt as the
            # initial value. users tend to overlook it and it leads to invalid
            # paths like: ".C:\mydir".
            # Can't use `if not path` either, as `None` will be handled differently (abort)
            if path == "":  # yd。如果没有指定project path，则默认使用当前
                #path = "."
                path = for_debug_path # yd。用于调试

    if args.no_prompt and not os.path.isdir(path):
        print_error_and_exit(f"Project init path '{path}' not found.")

    if path and not os.path.isdir(path):
        _ask_create_path(path)

    if path is None or not os.path.isdir(path):
        print_cancel()

    if not args.no_prompt and len(os.listdir(path)) > 0:
        _ask_overwrite(path)

    telemetry.track_project_init(path)

    init_project(args, path)
