from typing import Iterable, Text, Optional, List

from rasa.shared.core.domain import Domain
from rasa.shared.core.training_data.structures import StoryGraph
from rasa.shared.nlu.training_data.training_data import TrainingData


def training_data_from_paths(paths: Iterable[Text], language: Text) -> TrainingData:
    """
    yd。功能：读取paths所对应的文件，将读取的结果保存在training_data_sets中，并将training_data_sets合并到一起
    :param paths:
    :param language:
    :return:
    """
    from rasa.shared.nlu.training_data import loading

    training_data_sets = [loading.load_data(nlu_file, language) for nlu_file in paths]
    return TrainingData().merge(*training_data_sets)


def story_graph_from_paths(
    files: List[Text], domain: Domain, exclusion_percentage: Optional[int] = None
) -> StoryGraph:
    """Returns the `StoryGraph` from paths."""
    from rasa.shared.core.training_data import loading

    story_steps = loading.load_data_from_files(files, domain, exclusion_percentage)
    return StoryGraph(story_steps)
