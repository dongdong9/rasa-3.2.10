import argparse
import os
from typing import List

from rasa.cli import SubParsersAction
from rasa.cli.arguments import visualize as arguments
from rasa.shared.constants import DEFAULT_DATA_PATH
from rasa.utils.common import change_cur_work_dir

def add_subparser(
    subparsers: SubParsersAction, parents: List[argparse.ArgumentParser]
) -> None:
    """Add all visualization parsers.

    Args:
        subparsers: subparser we are going to attach to
        parents: Parent parsers, needed to ensure tree structure in argparse
    """
    visualize_parser = subparsers.add_parser(
        "visualize",
        parents=parents,
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Visualize stories.",
    )
    visualize_parser.set_defaults(func=visualize_stories)

    arguments.set_visualize_stories_arguments(visualize_parser)

#yd。执行命令"rasa visualize"调用本方法
def visualize_stories(args: argparse.Namespace) -> None:
    """
    yd。功能：在config.yml所在的文件夹下生成graph.html文件，看一下rasa整体基于story和rules相关的流程图
    :param args:
    :return:
    """
    import rasa.core.visualize

    #yd。下面是切换当前工作目录
    change_cur_work_dir()

    args.stories = rasa.shared.data.get_core_directory(args.stories)
    if args.nlu is None and os.path.exists(DEFAULT_DATA_PATH):
        args.nlu = rasa.shared.data.get_nlu_directory(DEFAULT_DATA_PATH)

    rasa.core.visualize.visualize(
        args.domain, args.stories, args.nlu, args.out, args.max_history
    )
