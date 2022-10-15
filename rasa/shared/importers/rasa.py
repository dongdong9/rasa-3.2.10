import logging
import os
from typing import Dict, List, Optional, Text, Union

import rasa.shared.data
import rasa.shared.utils.common
import rasa.shared.utils.io
from rasa.shared.core.training_data.structures import StoryGraph
from rasa.shared.importers import utils
from rasa.shared.importers.importer import TrainingDataImporter
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.core.domain import InvalidDomain, Domain
from rasa.shared.core.training_data.story_reader.yaml_story_reader import (
    YAMLStoryReader,
)
from rasa.shared.utils.cli import print_success
logger = logging.getLogger(__name__)


class RasaFileImporter(TrainingDataImporter):
    """Default `TrainingFileImporter` implementation."""

    def __init__(
        self,
        config_file: Optional[Text] = None,
        domain_path: Optional[Text] = None,
        training_data_paths: Optional[Union[List[Text], Text]] = None,
    ):

        self._domain_path = domain_path

        print(f"\n-----开始为self._nlu_files赋值")
        self._nlu_files = rasa.shared.data.get_data_files(
            training_data_paths, rasa.shared.data.is_nlu_file
        ) #yd。从training_data_paths对应的路径（例如"data"）下找出NLU文件所在的路径，即找到".\\data\\nlu.yml"

        print(f"\n-----开始为self._story_files赋值")
        self._story_files = rasa.shared.data.get_data_files(
            training_data_paths, YAMLStoryReader.is_stories_file
        )#yd。从training_data_paths对应的路径（例如"data"）下找出存在以"stories"或"rules"开头的行的yml文件，即找到".\\data\\rules.yml"、".\\data\\stories.yml"和'.\\tests\\test_stories.yml'

        print(f"\n-----开始为self._conversation_test_files赋值")
        self._conversation_test_files = rasa.shared.data.get_data_files(
            training_data_paths, YAMLStoryReader.is_test_stories_file
        )#yd。从training_data_paths对应的路径（例如"data"）下找出文件名前缀为"test_"且存在以"stories"或"rules"开头的行的文件，返回结果为['.\\tests\\test_stories.yml']

        self.config_file = config_file

    def get_config(self) -> Dict:
        """
        yd。功能：读取self.config_file（默认为config.yml）的内容，以dict的形式保存结果，例如返回的config内容为{'recipe': 'default.v1', 'language': 'zh', 'pipeline': [{'name': 'JiebaTokenizer'}, {'name': 'LanguageModelFeaturizer', 'model_name': 'bert', 'model_weights': 'bert-base-chinese'}, {'name': 'DIETClassifier', 'epochs': 100}], 'policies': None}
        :return:
        """
        """Retrieves model config (see parent class for full docstring)."""
        if not self.config_file or not os.path.exists(self.config_file):
            logger.debug("No configuration file was provided to the RasaFileImporter.")
            return {}

        config = rasa.shared.utils.io.read_model_configuration(self.config_file)
        return config

    @rasa.shared.utils.common.cached_method
    def get_config_file_for_auto_config(self) -> Optional[Text]:
        """Returns config file path for auto-config only if there is a single one."""
        return self.config_file

    def get_stories(self, exclusion_percentage: Optional[int] = None) -> StoryGraph:
        """Retrieves training stories / rules (see parent class for full docstring)."""
        return utils.story_graph_from_paths(
            self._story_files, self.get_domain(), exclusion_percentage
        )

    def get_conversation_tests(self) -> StoryGraph:
        """Retrieves conversation test stories (see parent class for full docstring)."""
        #yd。功能：读取self._conversation_test_files对应的yml文件，即'.\\tests\\test_stories.yml'
        return utils.story_graph_from_paths(
            self._conversation_test_files, self.get_domain()
        )

    def get_nlu_data(self, language: Optional[Text] = "en") -> TrainingData:
        """
        yd。功能：读取self._nlu_files（例如['data\\nlu.yml']）所对应的文件，解析文件中每个句子的意图和实体，
            用解析结果构建TrainingData类对象保存在training_data_sets中，最后将training_data_sets中的TrainingData类对象合并成一个。
        :param language:
        :return:
        """
        """Retrieves NLU training data (see parent class for full docstring)."""
        return utils.training_data_from_paths(self._nlu_files, language)

    def get_domain(self) -> Domain:
        """
        yd。功能：待补充
        :return:
        """
        """Retrieves model domain (see parent class for full docstring)."""
        domain = Domain.empty()

        # If domain path is None, return an empty domain
        if not self._domain_path:
            return domain
        try:
            domain = Domain.load(self._domain_path)
        except InvalidDomain as e:
            rasa.shared.utils.io.raise_warning(
                f"Loading domain from '{self._domain_path}' failed. Using "
                f"empty domain. Error: '{e}'"
            )
        else:
            print_success(f"yd。成功加载self._domain_path = {self._domain_path}对应的文件")

        return domain
