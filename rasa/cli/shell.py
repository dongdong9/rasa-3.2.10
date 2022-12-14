import argparse
import logging
import uuid

from typing import List

from rasa import telemetry
from rasa.cli import SubParsersAction
from rasa.cli.arguments import shell as arguments
from rasa.engine.storage.local_model_storage import LocalModelStorage
from rasa.model import get_local_model
from rasa.shared.data import TrainingType
from rasa.shared.utils.cli import print_error
from rasa.exceptions import ModelNotFound
from rasa.utils.common import change_cur_work_dir

logger = logging.getLogger(__name__)


def add_subparser(
    subparsers: SubParsersAction, parents: List[argparse.ArgumentParser]
) -> None:
    """Add all shell parsers.

    Args:
        subparsers: subparser we are going to attach to
        parents: Parent parsers, needed to ensure tree structure in argparse
    """
    shell_parser = subparsers.add_parser(
        "shell",
        parents=parents,
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help=(
            "Loads your trained model and lets you talk to your "
            "assistant on the command line."
        ),
    )
    shell_parser.set_defaults(func=shell)

    shell_parser.add_argument(
        "--conversation-id",
        default=uuid.uuid4().hex,
        required=False,
        help="Set the conversation ID.",
    )

    run_subparsers = shell_parser.add_subparsers()

    shell_nlu_subparser = run_subparsers.add_parser(
        "nlu",
        parents=parents,
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Interprets messages on the command line using your NLU model.",
    )

    shell_nlu_subparser.set_defaults(func=shell_nlu)

    arguments.set_shell_arguments(shell_parser)
    arguments.set_shell_nlu_arguments(shell_nlu_subparser)


def shell_nlu(args: argparse.Namespace) -> None:
    """Talk with an NLU only bot though the command line."""
    from rasa.cli.utils import get_validated_path
    from rasa.shared.constants import DEFAULT_MODELS_PATH
    import rasa.nlu.run

    args.connector = "cmdline"

    model_path = get_validated_path(args.model, "model", DEFAULT_MODELS_PATH)

    try:
        model = get_local_model(model_path)
    except ModelNotFound:
        print_error(
            "No model found. Train a model before running the "
            "server using `rasa train nlu`."
        )
        return

    metadata = LocalModelStorage.metadata_from_archive(model)
    if metadata.training_type == TrainingType.CORE:
        print_error(
            "No NLU model found. Train a model before running the "
            "server using `rasa train nlu`."
        )
        return

    telemetry.track_shell_started("nlu")
    rasa.nlu.run.run_cmdline(model)

#yd。执行命令"rasa shell"后调用本方法，"rasa shell"等同于命令"rasa run"
def shell(args: argparse.Namespace) -> None:
    """
    yd。功能：运行一个rasa模型，启动一个基于命令行界面的对话系统。
    :param args:
    :return:
    """
    """Talk with a bot though the command line."""
    from rasa.cli.utils import get_validated_path
    from rasa.shared.constants import DEFAULT_MODELS_PATH

    #yd。下面是切换当前工作目录
    change_cur_work_dir()

    args.connector = "cmdline"

    model = get_validated_path(args.model, "model", DEFAULT_MODELS_PATH) #yd。获取训练好的模型的路径，例如"models"

    try:
        model = get_local_model(model) #yd。判断当前model对应的路径是否有效，如果有效，则返回最新模型的路径，保存在model中，例如'models\\nlu-20221020-152636-religious-deque.tar.gz'
    except ModelNotFound:
        print_error(
            "No model found. Train a model before running the "
            "server using `rasa train`."
        )
        return

    metadata = LocalModelStorage.metadata_from_archive(model)#yd。功能：从已经保存的模型中，获取模型的元数据（即与模型有关的各项参数，如模型的训练时间，对应的rasa源码版本等）

    if metadata.training_type == TrainingType.NLU:
        import rasa.nlu.run

        telemetry.track_shell_started("nlu")

        rasa.nlu.run.run_cmdline(model)
    else:
        import rasa.cli.run

        telemetry.track_shell_started("rasa")

        rasa.cli.run.run(args)#yd。调用rasa/cli/run.py的run()方法，启动一个rasa模型
