from typing import Iterable, Text, Optional, List

from rasa.shared.core.domain import Domain
from rasa.shared.core.training_data.structures import StoryGraph
from rasa.shared.nlu.training_data.training_data import TrainingData


def training_data_from_paths(paths: Iterable[Text], language: Text) -> TrainingData:
    """
    yd。功能：读取paths（例如['data\\nlu.yml']）所对应的文件，解析文件中每个句子的意图和实体，
            用解析结果构建TrainingData类对象保存在training_data_sets中，最后将training_data_sets中的TrainingData类对象合并成一个。
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
