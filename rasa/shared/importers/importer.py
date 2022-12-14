from abc import ABC, abstractmethod
from functools import reduce
from typing import Text, Optional, List, Dict, Set, Any, Tuple, Type, Union, cast
import logging

import rasa.shared.constants
import rasa.shared.utils.common
import rasa.shared.core.constants
import rasa.shared.utils.io
from rasa.shared.core.domain import (
    Domain,
    KEY_E2E_ACTIONS,
    KEY_INTENTS,
    KEY_RESPONSES,
    KEY_ACTIONS,
)
from rasa.shared.core.events import ActionExecuted, UserUttered
from rasa.shared.core.training_data.structures import StoryGraph
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.constants import ENTITIES, ACTION_NAME
from rasa.shared.core.domain import IS_RETRIEVAL_INTENT_KEY

logger = logging.getLogger(__name__)


class TrainingDataImporter(ABC):
    """Common interface for different mechanisms to load training data."""

    @abstractmethod
    def __init__(
        self,
        config_file: Optional[Text] = None,
        domain_path: Optional[Text] = None,
        training_data_paths: Optional[Union[List[Text], Text]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialise the importer."""
        ...

    @abstractmethod
    def get_domain(self) -> Domain:
        """Retrieves the domain of the bot.

        Returns:
            Loaded `Domain`.
        """
        ...

    @abstractmethod
    def get_stories(self, exclusion_percentage: Optional[int] = None) -> StoryGraph:
        """Retrieves the stories that should be used for training.

        Args:
            exclusion_percentage: Amount of training data that should be excluded.

        Returns:
            `StoryGraph` containing all loaded stories.
        """
        ...

    def get_conversation_tests(self) -> StoryGraph:
        """Retrieves end-to-end conversation stories for testing.

        Returns:
            `StoryGraph` containing all loaded stories.
        """
        return self.get_stories()

    @abstractmethod
    def get_config(self) -> Dict:
        """Retrieves the configuration that should be used for the training.

        Returns:
            The configuration as dictionary.
        """
        ...

    @abstractmethod
    def get_config_file_for_auto_config(self) -> Optional[Text]:
        """Returns config file path for auto-config only if there is a single one."""
        ...

    @abstractmethod
    def get_nlu_data(self, language: Optional[Text] = "en") -> TrainingData:
        """Retrieves the NLU training data that should be used for training.

        Args:
            language: Can be used to only load training data for a certain language.

        Returns:
            Loaded NLU `TrainingData`.
        """
        ...

    @staticmethod
    def load_from_config(
        config_path: Text,
        domain_path: Optional[Text] = None,
        training_data_paths: Optional[List[Text]] = None,
    ) -> "TrainingDataImporter":
        """
        yd。功能：创建一个E2EImporter对象，该对象的成员变量保存了nlu文件所在的路径，story文件所在的路径和测试会话文件所在的路径
        :param config_path:
        :param domain_path:
        :param training_data_paths:
        :return:
        """
        """Loads a `TrainingDataImporter` instance from a configuration file."""
        # yd。读取config_path对应对配置文件（例如config.yml），过滤被注释的内容，以dict的形式返回有效的字段内容。例如config = {'recipe': 'default.v1', 'language': 'en', 'pipeline': None, 'policies': None}
        config = rasa.shared.utils.io.read_config_file(config_path)

        return TrainingDataImporter.load_from_dict(
            config, config_path, domain_path, training_data_paths
        )

    @staticmethod
    def load_core_importer_from_config(
        config_path: Text,
        domain_path: Optional[Text] = None,
        training_data_paths: Optional[List[Text]] = None,
    ) -> "TrainingDataImporter":
        """Loads core `TrainingDataImporter` instance.

        Instance loaded from configuration file will only read Core training data.
        """
        importer = TrainingDataImporter.load_from_config(
            config_path, domain_path, training_data_paths
        )
        return importer

    @staticmethod
    def load_nlu_importer_from_config(
        config_path: Text, #yd。config文件的路径，比如"config.yml"
        domain_path: Optional[Text] = None,
        training_data_paths: Optional[List[Text]] = None, #yd。nlu data 所在的文件夹路径
    ) -> "TrainingDataImporter":
        """
        #yd。功能：创建一个NluDataImporter对象，该对象的成员变量保存了nlu文件所在的路径（例如".\\data\\nlu.yml"），story文件所在的路径和测试会话文件所在的路径
        :param config_path:
        :param domain_path:
        :param training_data_paths:
        :return:
        """
        """Loads nlu `TrainingDataImporter` instance.

        Instance loaded from configuration file will only read NLU training data.
        """
        importer = TrainingDataImporter.load_from_config(
            config_path, domain_path, training_data_paths
        )#yd。功能：创建一个E2EImporter对象，该对象的成员变量保存了nlu文件所在的路径，story文件所在的路径和测试会话文件所在的路径

        if isinstance(importer, E2EImporter):
            # When we only train NLU then there is no need to enrich the data with
            # E2E data from Core training data.
            importer = importer.importer #yd。返回的importer为ResponsesSyncImporter对象

        return NluDataImporter(importer)

    @staticmethod
    def load_from_dict(
        config: Optional[Dict] = None,
        config_path: Optional[Text] = None,
        domain_path: Optional[Text] = None,
        training_data_paths: Optional[List[Text]] = None,
    ) -> "TrainingDataImporter":
        """
        yd。功能：创建一个E2EImporter对象，该对象的成员变量保存了nlu文件所在的路径，story文件所在的路径和测试会话文件所在的路径
        :param config: 以dict的形式保存config.yml中每个字段的内容，例如config = {'recipe': 'default.v1', 'language': 'en', 'pipeline': None, 'policies': None}
        :param config_path: config文件的路径，例如"config.yml"
        :param domain_path:
        :param training_data_paths: 训练数据所在的文件夹，例如"data"
        :return:
        """
        """Loads a `TrainingDataImporter` instance from a dictionary."""
        from rasa.shared.importers.rasa import RasaFileImporter

        config = config or {}
        importers = config.get("importers", [])
        importers = [
            TrainingDataImporter._importer_from_dict(
                importer, config_path, domain_path, training_data_paths
            )
            for importer in importers
        ]
        importers = [importer for importer in importers if importer]
        if not importers: #yd。如果importers为空
            importers = [
                # yd。根据config_path和training_data_paths，创建一个importer对象，该对象中包括nlu文件的路径（例如".\\data\\nlu.yml"）
                # story文件的路径（例如".\\data\\rules.yml"、".\\data\\stories.yml"和'.\\tests\\test_stories.yml'）
                # 测试会话的文件路径（例如'.\\tests\\test_stories.yml'）
                RasaFileImporter(config_path, domain_path, training_data_paths)
            ]

        return E2EImporter(ResponsesSyncImporter(CombinedDataImporter(importers))) #yd。一层层继承，得到E2EImporter对象

    @staticmethod
    def _importer_from_dict(
        importer_config: Dict,
        config_path: Text,
        domain_path: Optional[Text] = None,
        training_data_paths: Optional[List[Text]] = None,
    ) -> Optional["TrainingDataImporter"]:
        from rasa.shared.importers.multi_project import MultiProjectImporter
        from rasa.shared.importers.rasa import RasaFileImporter

        module_path = importer_config.pop("name", None)
        if module_path == RasaFileImporter.__name__:
            importer_class: Type[TrainingDataImporter] = RasaFileImporter
        elif module_path == MultiProjectImporter.__name__:
            importer_class = MultiProjectImporter
        else:
            try:
                importer_class = rasa.shared.utils.common.class_from_module_path(
                    module_path
                )
            except (AttributeError, ImportError):
                logging.warning(f"Importer '{module_path}' not found.")
                return None

        constructor_arguments = rasa.shared.utils.common.minimal_kwargs(
            importer_config, importer_class
        )

        return importer_class(
            config_path, domain_path, training_data_paths, **constructor_arguments
        )

    def fingerprint(self) -> Text:
        """Returns a random fingerprint as data shouldn't be cached."""
        return rasa.shared.utils.io.random_string(25)

    def __repr__(self) -> Text:
        """Returns text representation of object."""
        return self.__class__.__name__


class NluDataImporter(TrainingDataImporter):
    """Importer that skips any Core-related file reading."""

    def __init__(self, actual_importer: TrainingDataImporter):
        """Initializes the NLUDataImporter."""
        self._importer = actual_importer

    def get_domain(self) -> Domain:
        """Retrieves model domain (see parent class for full docstring)."""
        return Domain.empty()

    def get_stories(self, exclusion_percentage: Optional[int] = None) -> StoryGraph:
        """Retrieves training stories / rules (see parent class for full docstring)."""
        return StoryGraph([])

    def get_conversation_tests(self) -> StoryGraph:
        """Retrieves conversation test stories (see parent class for full docstring)."""
        return StoryGraph([])

    def get_config(self) -> Dict:
        """
        yd。功能：获取self._importers中每个importer.config_file（默认值为“config.yml”）的内容，以key-value对的形式保存在dict中。然后将这些dict合并后返回
        :return:
        """
        """Retrieves model config (see parent class for full docstring)."""
        return self._importer.get_config()

    def get_nlu_data(self, language: Optional[Text] = "en") -> TrainingData:
        """
        yd。功能：读取所有关于nlu data对应的文件（包括路径self._nlu_files，默认为['data\\nlu.yml']；包括路径self._domain_path，默认值为None），
                并将读取的内容合并到一起后返回
        :param language:
        :return:
        """
        """Retrieves NLU training data (see parent class for full docstring)."""
        return self._importer.get_nlu_data(language)

    @rasa.shared.utils.common.cached_method
    def get_config_file_for_auto_config(self) -> Optional[Text]:
        """
        yd。功能：返回config文件的路径，默认为"config.yml"
        :return:
        """
        """Returns config file path for auto-config only if there is a single one."""
        return self._importer.get_config_file_for_auto_config()


class CombinedDataImporter(TrainingDataImporter):
    """A `TrainingDataImporter` that combines multiple importers.

    Uses multiple `TrainingDataImporter` instances
    to load the data as if they were a single instance.
    """

    def __init__(self, importers: List[TrainingDataImporter]):
        self._importers = importers

    @rasa.shared.utils.common.cached_method
    def get_config(self) -> Dict:
        """
        yd。功能：获取self._importers中每个importer.config_file（默认值为“config.yml”）的内容，以key-value对的形式保存在dict中。然后将这些dict合并后返回
        :return:
        """
        """Retrieves model config (see parent class for full docstring)."""
        configs = [importer.get_config() for importer in self._importers]

        return reduce(lambda merged, other: {**merged, **(other or {})}, configs, {})

    @rasa.shared.utils.common.cached_method
    def get_domain(self) -> Domain:
        """
        yd。功能：将self._importers中每个importer的importer._domain_path（默认为domain.yml）的内容读取出来并合并，返回合并后的结果
        :return:
        """
        """Retrieves model domain (see parent class for full docstring)."""
        domains = [importer.get_domain() for importer in self._importers]

        return reduce(
            lambda merged, other: merged.merge(other),
            domains,
            Domain.empty(),
        )

    @rasa.shared.utils.common.cached_method
    def get_stories(self, exclusion_percentage: Optional[int] = None) -> StoryGraph:
        """
        yd。功能：默认读取"data/stories.yml"中stories字段，用每个story创建一个StoryStep类对象，将这些对象保存在StoryGraph类对象的成员变量story_steps中。
        :param exclusion_percentage:
        :return:
        """
        """Retrieves training stories / rules (see parent class for full docstring)."""
        stories = [
            importer.get_stories(exclusion_percentage) for importer in self._importers
        ]

        return reduce(
            lambda merged, other: merged.merge(other), stories, StoryGraph([])
        )

    @rasa.shared.utils.common.cached_method
    def get_conversation_tests(self) -> StoryGraph:
        """Retrieves conversation test stories (see parent class for full docstring)."""
        #yd。功能：读取self._conversation_test_files对应的yml文件（默认为'.\\tests\\test_stories.yml'）
        stories = [importer.get_conversation_tests() for importer in self._importers]

        return reduce(
            lambda merged, other: merged.merge(other), stories, StoryGraph([])
        )

    @rasa.shared.utils.common.cached_method
    def get_nlu_data(self, language: Optional[Text] = "en") -> TrainingData:
        """
        yd。功能：读取self._importers中每个importer的importer._nlu_files（例如['data\\nlu.yml']）所对应的文件，
            解析文件中每个句子的意图和实体，用解析结果构建TrainingData类对象保存在training_data_sets中，
            最后将training_data_sets中的TrainingData类对象合并成一个。
        :param language:
        :return:
        """
        """Retrieves NLU training data (see parent class for full docstring)."""

        #yd。读取importer._nlu_files（默认为".\\data\\nlu.yml"）所对应的内容
        nlu_data = [importer.get_nlu_data(language) for importer in self._importers]

        return reduce(
            lambda merged, other: merged.merge(other), nlu_data, TrainingData()
        )

    @rasa.shared.utils.common.cached_method
    def get_config_file_for_auto_config(self) -> Optional[Text]:
        """
        yd。功能：返回config文件的路径，默认为"config.yml"
        :return:
        """
        """Returns config file path for auto-config only if there is a single one."""
        if len(self._importers) != 1:
            rasa.shared.utils.io.raise_warning(
                "Auto-config for multiple importers is not supported; "
                "using config as is."
            )
            return None
        return self._importers[0].get_config_file_for_auto_config()


class ResponsesSyncImporter(TrainingDataImporter):
    """Importer that syncs `responses` between Domain and NLU training data.

    Synchronizes responses between Domain and NLU and
    adds retrieval intent properties from the NLU training data
    back to the Domain.
    """

    def __init__(self, importer: TrainingDataImporter):
        """Initializes the ResponsesSyncImporter."""
        self._importer = importer

    def get_config(self) -> Dict:
        """Retrieves model config (see parent class for full docstring)."""
        return self._importer.get_config()

    @rasa.shared.utils.common.cached_method
    def get_config_file_for_auto_config(self) -> Optional[Text]:
        """Returns config file path for auto-config only if there is a single one."""
        return self._importer.get_config_file_for_auto_config()

    @rasa.shared.utils.common.cached_method
    def get_domain(self) -> Domain:
        """Merge existing domain with properties of retrieval intents in NLU data."""
        existing_domain = self._importer.get_domain()
        existing_nlu_data = self._importer.get_nlu_data()

        # Merge responses from NLU data with responses in the domain.
        # If NLU data has any retrieval intents, then add corresponding
        # retrieval actions with `utter_` prefix automatically to the
        # final domain, update the properties of existing retrieval intents.
        domain_with_retrieval_intents = self._get_domain_with_retrieval_intents(
            existing_nlu_data.retrieval_intents,
            existing_nlu_data.responses,
            existing_domain,
        )

        existing_domain = existing_domain.merge(
            domain_with_retrieval_intents, override=True
        )
        existing_domain.check_missing_responses()

        return existing_domain

    @staticmethod
    def _construct_retrieval_action_names(retrieval_intents: Set[Text]) -> List[Text]:
        """Lists names of all retrieval actions related to passed retrieval intents.

        Args:
            retrieval_intents: List of retrieval intents defined in the NLU training
                data.

        Returns: Names of corresponding retrieval actions
        """
        return [
            f"{rasa.shared.constants.UTTER_PREFIX}{intent}"
            for intent in retrieval_intents
        ]

    @staticmethod
    def _get_domain_with_retrieval_intents(
        retrieval_intents: Set[Text],
        responses: Dict[Text, List[Dict[Text, Any]]],
        existing_domain: Domain,
    ) -> Domain:
        """Construct a domain consisting of retrieval intents.

         The result domain will have retrieval intents that are listed
         in the NLU training data.

        Args:
            retrieval_intents: Set of retrieval intents defined in NLU training data.
            responses: Responses defined in NLU training data.
            existing_domain: Domain which is already loaded from the domain file.

        Returns: Domain with retrieval actions added to action names and properties
          for retrieval intents updated.
        """
        # Get all the properties already defined
        # for each retrieval intent in other domains
        # and add the retrieval intent property to them
        retrieval_intent_properties = []
        for intent in retrieval_intents:
            intent_properties = (
                existing_domain.intent_properties[intent]
                if intent in existing_domain.intent_properties
                else {}
            )
            intent_properties[IS_RETRIEVAL_INTENT_KEY] = True
            retrieval_intent_properties.append({intent: intent_properties})

        action_names = ResponsesSyncImporter._construct_retrieval_action_names(
            retrieval_intents
        )

        return Domain.from_dict(
            {
                KEY_INTENTS: retrieval_intent_properties,
                KEY_RESPONSES: responses,
                KEY_ACTIONS: action_names,
            }
        )

    def get_stories(self, exclusion_percentage: Optional[int] = None) -> StoryGraph:
        """Retrieves training stories / rules (see parent class for full docstring)."""
        return self._importer.get_stories(exclusion_percentage)

    def get_conversation_tests(self) -> StoryGraph:
        """Retrieves conversation test stories (see parent class for full docstring)."""
        return self._importer.get_conversation_tests()

    @rasa.shared.utils.common.cached_method
    def get_nlu_data(self, language: Optional[Text] = "en") -> TrainingData:
        """
        yd。功能：读取所有关于nlu data对应的文件（包括路径self._nlu_files，默认为['data\\nlu.yml']；包括路径self._domain_path，默认值为None），
                并将读取的内容合并到一起后然后返回。data\\nlu.yml中每个句子构建一个Message类对象，用这些类对象列表填充TrainingData类对象的
                entity_examples（存在实体的句子）、intent_examples（与意图有关的句子）、nlu_examples(所有与nlu有关的句子)、training_examples(所有训练的句子)
        :param language:
        :return:
        """
        """Updates NLU data with responses for retrieval intents from domain."""

        # yd。读取self._importer的成员self._importers中每个importer的importer._nlu_files （例如['data\\nlu.yml']）所对应的文件，
        # 解析文件中每个句子的意图和实体，用解析结果构建TrainingData类对象保存在training_data_sets中，
        # 最后将training_data_sets中的TrainingData类对象合并成一个。
        existing_nlu_data = self._importer.get_nlu_data(language)

        #yd。读取self._importer的成员self._importers中每个importer的importer._nlu_files（默认值为domain.yml）的内容，并合并
        existing_domain = self._importer.get_domain()

        return existing_nlu_data.merge(
            self._get_nlu_data_with_responses(
                existing_domain.retrieval_intent_responses
            )
        )

    @staticmethod
    def _get_nlu_data_with_responses(
        responses: Dict[Text, List[Dict[Text, Any]]]
    ) -> TrainingData:
        """Construct training data object with only the responses supplied.

        Args:
            responses: Responses the NLU data should
            be initialized with.

        Returns: TrainingData object with responses.

        """
        return TrainingData(responses=responses)


class E2EImporter(TrainingDataImporter):
    """Importer with the following functionality.

    - enhances the NLU training data with actions / user messages from the stories.
    - adds potential end-to-end bot messages from stories as actions to the domain
    """

    def __init__(self, importer: TrainingDataImporter) -> None:
        """Initializes the E2EImporter."""
        self.importer = importer

    @rasa.shared.utils.common.cached_method
    def get_domain(self) -> Domain:
        """Retrieves model domain (see parent class for full docstring)."""
        original = self.importer.get_domain()
        e2e_domain = self._get_domain_with_e2e_actions()

        return original.merge(e2e_domain)

    def _get_domain_with_e2e_actions(self) -> Domain:

        stories = self.get_stories()

        additional_e2e_action_names = set()
        for story_step in stories.story_steps:
            additional_e2e_action_names.update(
                {
                    event.action_text
                    for event in story_step.events
                    if isinstance(event, ActionExecuted) and event.action_text
                }
            )

        return Domain.from_dict({KEY_E2E_ACTIONS: list(additional_e2e_action_names)})

    def get_stories(self, exclusion_percentage: Optional[int] = None) -> StoryGraph:
        """Retrieves the stories that should be used for training.

        See parent class for details.
        """
        return self.importer.get_stories(exclusion_percentage)

    def get_conversation_tests(self) -> StoryGraph:
        """Retrieves conversation test stories (see parent class for full docstring)."""
        return self.importer.get_conversation_tests()

    def get_config(self) -> Dict:
        """Retrieves model config (see parent class for full docstring)."""
        return self.importer.get_config()

    @rasa.shared.utils.common.cached_method
    def get_config_file_for_auto_config(self) -> Optional[Text]:
        """Returns config file path for auto-config only if there is a single one."""
        return self.importer.get_config_file_for_auto_config()

    @rasa.shared.utils.common.cached_method
    def get_nlu_data(self, language: Optional[Text] = "en") -> TrainingData:
        """
        yd。功能：Rasa提供的默认动作、data\\nlu.yml中的句子、data/stories.yml中每个故事的intent和action，用他们构建Message类对象，
                 用这些类对象来初始化TrainingData类对象，将得到的TrainingData类对象保存在training_datasets这个list中，然后将这个list
                 中的TrainingData类对象合并成一个并返回。
        :param language:
        :return:
        """
        """Retrieves NLU training data (see parent class for full docstring)."""
        training_datasets = [
            _additional_training_data_from_default_actions(),#yd。功能用Rasa提供的默认动作来构建Message类对象列表，将这些Message类对象保存在TrainingData类对象的成员变量training_examples中
            self.importer.get_nlu_data(language), #yd。功能：默认读取data\\nlu.yml，用其中每个句子构建一个Message类对象，用这些类对象列表填充TrainingData类对象的entity_examples、intent_examples、nlu_examples、training_examples
            self._additional_training_data_from_stories(),#yd。功能：读取"data/stories.yml"中stories字段，用每个story创建intent和action分别创建两个Message类对象。用这些Message类对象填充TrainingData的成员变量training_examples
        ]

        return reduce(
            lambda merged, other: merged.merge(other), training_datasets, TrainingData()
        )

    def _additional_training_data_from_stories(self) -> TrainingData:
        """
        yd。功能：读取"data/stories.yml"中stories字段，用每个story创建intent和action分别创建两个Message类对象。用这些Message类对象填充TrainingData的成员变量training_examples
        :return:
        """
        stories = self.get_stories() #yd。功能：默认读取"data/stories.yml"中stories字段，用每个story创建一个StoryStep类对象，将这些对象保存在StoryGraph类对象的成员变量story_steps中。

        #yd。功能：将StoryGraph类对象中的StoryStep类对象（stories.yml中stories字段下，一个故事对应一个StoryStep类对象）按ActionExecuted类与UserUttered类分开
        utterances, actions = _unique_events_from_stories(stories)

        # Sort events to guarantee deterministic behavior and to avoid that the NLU
        # model has to be retrained due to changes in the event order within
        # the stories.
        sorted_utterances = sorted(
            utterances, key=lambda user: user.intent_name or user.text or ""
        )
        sorted_actions = sorted(
            actions, key=lambda action: action.action_name or action.action_text or ""
        )

        additional_messages_from_stories = [
            _messages_from_action(action) for action in sorted_actions
        ] + [_messages_from_user_utterance(user) for user in sorted_utterances]

        logger.debug(
            f"Added {len(additional_messages_from_stories)} training data examples "
            f"from the story training data."
        )
        return TrainingData(additional_messages_from_stories)


def _unique_events_from_stories(
    stories: StoryGraph,
) -> Tuple[Set[UserUttered], Set[ActionExecuted]]:
    """
    yd。功能：将StoryGraph类对象中的StoryStep类对象（stories.yml中stories字段下，一个故事对应一个StoryStep类对象）按ActionExecuted类与UserUttered类分开
    :param stories:
    :return:
    """
    action_events = set()
    user_events = set()

    for story_step in stories.story_steps:
        for event in story_step.events:
            if isinstance(event, ActionExecuted):
                action_events.add(event)
            elif isinstance(event, UserUttered):
                user_events.add(event)

    return user_events, action_events


def _messages_from_user_utterance(event: UserUttered) -> Message:
    # sub state correctly encodes intent vs text
    data = cast(Dict[Text, Any], event.as_sub_state())
    # sub state stores entities differently
    if data.get(ENTITIES) and event.entities:
        data[ENTITIES] = event.entities

    return Message(data=data)


def _messages_from_action(event: ActionExecuted) -> Message:
    # sub state correctly encodes action_name vs action_text
    return Message(data=event.as_sub_state())


def _additional_training_data_from_default_actions() -> TrainingData:
    """
    yd。功能：用Rasa提供的默认动作来构建Message类对象列表，将这些Message类对象保存在TrainingData类对象的成员变量training_examples中
    :return:
    """
    additional_messages_from_default_actions = [
        Message(data={ACTION_NAME: action_name})
        for action_name in rasa.shared.core.constants.DEFAULT_ACTION_NAMES
    ]

    return TrainingData(additional_messages_from_default_actions)
