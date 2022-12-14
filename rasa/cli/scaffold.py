import argparse
import os
import sys
from typing import List, Text
from rasa.constants import RASA_DEMO_FILES_DIR
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
        questionary.confirm("Do you want to train an initial model? ðªð½")
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
        ) #ydãå®ææ¨¡åçè®­ç»ï¼å°æ¨¡åä¿å­å¨'models\yyyymmdd-200313-plastic-application.tar.gz'ä¸­
        args.model = training_result.model

        print_run_or_instructions(args) #ydãè¿éæ¯è¾å¥å¥å­ï¼è·åæºå¨äººçåç­çå¥å£å½æ°

    else:
        print_success(
            "No problem ðð¼. You can also train a model later by going "
            "to the project directory and running 'rasa train'."
        )


def print_run_or_instructions(args: argparse.Namespace) -> None:
    from rasa.core import constants
    import questionary

    should_run = (
        questionary.confirm(
            "Do you want to speak to the trained assistant on the command line? ð¤"
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

        shell(args)#ydãè°ç¨rasa/cli/shell.pyçshell()æ¹æ³
    else:
        if args.no_prompt:
            print(
                "If you want to speak to the assistant, "
                "run 'rasa shell' at any time inside "
                "the project directory."
            )
        else:
            print_success(
                "Ok ðð¼. "
                "If you want to speak to the assistant, "
                "run 'rasa shell' at any time inside "
                "the project directory."
            )


def init_project(args: argparse.Namespace, path: Text) -> None:
    """

    :param args:
    :param path:æ°åå»ºçæä»¶å¤¹åç§°ï¼ä¾å¦"rasa_demo_files_dir"
    :return:
    """
    """Inits project."""
    os.chdir(path) #ydãä¿®æ¹å½åçå·¥ä½ç®å½è³pathæå¯¹åºçè·¯å¾ï¼æ§è¡äºè¿å¥åï¼å¨æ§è¡os.getcwd()ï¼è¿åçè·¯å¾å³ä¸ºpathå¯¹åºçå¼
    print_success(f"å½åå·¥ä½ç®å½å·²ç»è¢«åæ¢ï¼æ°ços.getcwd() = {os.getcwd()}") #ydãæç¤ºå½åå·¥ä½ç®å½
    create_initial_project(".") #ydãå°"rasa\\cli\\scaffold\\initial_project"ä¸çæä»¶æ¨¡æ¿æ·è´å°pathæå¯¹åºçæä»¶å¤¹ä¸ï¼ç¨äºåå§årasa project
    print(f"Created project directory at '{os.getcwd()}'.")
    print_train_or_instructions(args)


def create_initial_project(path: Text) -> None:
    """
    ydãåè½ï¼å°"rasa\\cli\\scaffold\\initial_project"ä¸çæä»¶æ¨¡æ¿æ·è´å°pathæå¯¹åºçæä»¶å¤¹ä¸ï¼ç¨äºåå§årasa project
    :param path:
    :return:
    """
    """Creates directory structure and templates for initial project."""
    from distutils.dir_util import copy_tree

    copy_tree(scaffold_path(), path)


def scaffold_path() -> Text:
    """
    ydãåè½ï¼è·å"rasa\\cli\\scaffold\\initial_project"è¿ä¸ªè·¯å¾ï¼å ä¸ºè¿ä¸ªæä»¶å¤¹ä¸ä¿å­çåå§åä¸ä¸ªrasa projectæéçæä»¶æ¨¡æ¿
    :return:
    """
    import pkg_resources

    extracted_scaffold_path = pkg_resources.resource_filename(__name__, "initial_project") #ydã__name__åéçå¼ä¸ºå½å.pyæä»¶çåç§°ï¼æ­¤æ¶å®çå¼ä¸º"rasa.cli.scaffold"
    return extracted_scaffold_path

def print_cancel() -> None:
    print_success("Ok. You can continue setting up by running 'rasa init' ðð½ââï¸")
    sys.exit(0)


def _ask_create_path(path: Text) -> None:
    import questionary

    should_create = questionary.confirm(
        f"Path '{path}' does not exist ð§. Create path?"
    ).ask() #ydãç¡®è®¤æ¯å¦è¦åå»ºpathå¯¹åºçè·¯å¾ï¼é»è®¤ä¸ºTrueï¼å³éè¦åå»º

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
            "running 'rasa init' again ðð½ââï¸"
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

    print_success("Welcome to Rasa! ð¤\n") #ydãæå°ç»¿è²çæç¤ºä¿¡æ¯
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
            f"Now let's start! ðð½\n"
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
            ) #ydãè·åè¾å¥çproject_path
            # set the default directory. we can't use the `default` property
            # in questionary as we want to avoid showing the "." in the prompt as the
            # initial value. users tend to overlook it and it leads to invalid
            # paths like: ".C:\mydir".
            # Can't use `if not path` either, as `None` will be handled differently (abort)
            if path == "": #ydãå¦ææ²¡ææå®project pathï¼åé»è®¤ä½¿ç¨å½å
                path = "."
        else:
            for_debug_path = RASA_DEMO_FILES_DIR
            path = (
                questionary.text(
                    "Please enter a path where the project will be "
                    "created [default: {}]".format(for_debug_path)                )
                    .skip_if(args.no_prompt, default="")
                    .ask()
            )  # ydãè·åè¾å¥çproject_path
            # set the default directory. we can't use the `default` property
            # in questionary as we want to avoid showing the "." in the prompt as the
            # initial value. users tend to overlook it and it leads to invalid
            # paths like: ".C:\mydir".
            # Can't use `if not path` either, as `None` will be handled differently (abort)
            if path == "":  # ydãå¦ææ²¡ææå®project pathï¼åé»è®¤ä½¿ç¨å½å
                #path = "."
                path = for_debug_path # ydãç¨äºè°è¯

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
    print("ydãå®ææ§è¡run()")
