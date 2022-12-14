import json
import logging
import os
import typing
from typing import Optional, Text, Callable, Dict, Any, List

import rasa.shared.utils.io
from rasa.shared.nlu.training_data.formats.dialogflow import (
    DIALOGFLOW_AGENT,
    DIALOGFLOW_ENTITIES,
    DIALOGFLOW_ENTITY_ENTRIES,
    DIALOGFLOW_INTENT,
    DIALOGFLOW_INTENT_EXAMPLES,
    DIALOGFLOW_PACKAGE,
)
from rasa.shared.nlu.training_data.training_data import TrainingData

if typing.TYPE_CHECKING:
    from rasa.shared.nlu.training_data.formats.readerwriter import TrainingDataReader

logger = logging.getLogger(__name__)

# Different supported file formats and their identifier
WIT = "wit"
LUIS = "luis"
RASA = "rasa_nlu"
RASA_YAML = "rasa_yml"
UNK = "unk"
DIALOGFLOW_RELEVANT = {DIALOGFLOW_ENTITIES, DIALOGFLOW_INTENT}

_json_format_heuristics: Dict[Text, Callable[[Any, Text], bool]] = {
    WIT: lambda js, fn: "utterances" in js and "luis_schema_version" not in js,
    LUIS: lambda js, fn: "luis_schema_version" in js,
    RASA: lambda js, fn: "rasa_nlu_data" in js,
    DIALOGFLOW_AGENT: lambda js, fn: "supportedLanguages" in js,
    DIALOGFLOW_PACKAGE: lambda js, fn: "version" in js and len(js) == 1,
    DIALOGFLOW_INTENT: lambda js, fn: "responses" in js,
    DIALOGFLOW_ENTITIES: lambda js, fn: "isEnum" in js,
    DIALOGFLOW_INTENT_EXAMPLES: lambda js, fn: "_usersays_" in fn,
    DIALOGFLOW_ENTITY_ENTRIES: lambda js, fn: "_entries_" in fn,
}


def load_data(resource_name: Text, language: Optional[Text] = "en") -> "TrainingData":
    """
    yd。功能：读取resource_name（例如'data\\nlu.yml'）所对应的文件，解析文件中每个句子的意图和实体，用解析结果构建TrainingData类对象。
            如果有多个TrainingData对象，则将其他的TrainingData类对象合并到首个TrainingData类对象中，最终只返回一个TrainingData类对象。
    :param resource_name:
    :param language:
    :return: training_data，由'data\\nlu.yml'中每个句子的意图和实体组成的TrainingData类对象
    """
    """Load training data from disk.

    Merges them if loaded from disk and multiple files are found."""
    if not os.path.exists(resource_name):
        raise ValueError(f"File '{resource_name}' does not exist.")

    if os.path.isfile(resource_name):
        files = [resource_name]
    else:
        files = rasa.shared.utils.io.list_files(resource_name)

    # yd。功能：如果filename是RASA_YAML格式，则读取文件的内容。
    # 例如如果filename是"data/nlu.yml"，则读取文件内容，解析出样本的意图和实体，进而为每个样本构建Message类对象，并将这些对象保存在TrainingData类对象中
    data_sets = [_load(f, language) for f in files] #yd。由TrainingData类对象组成的list

    #yd。将多个data_sets中的多个TrainingData类对象合并成一个
    training_data_sets: List[TrainingData] = [ds for ds in data_sets if ds]
    if len(training_data_sets) == 0:
        training_data = TrainingData()
    elif len(training_data_sets) == 1:
        training_data = training_data_sets[0]
    else:
        training_data = training_data_sets[0].merge(*training_data_sets[1:])

    return training_data


def _reader_factory(fformat: Text) -> Optional["TrainingDataReader"]:
    """Generates the appropriate reader class based on the file format."""
    from rasa.shared.nlu.training_data.formats import (
        RasaYAMLReader,
        WitReader,
        LuisReader,
        RasaReader,
        DialogflowReader,
    )

    reader: Optional["TrainingDataReader"] = None
    if fformat == LUIS:
        reader = LuisReader()
    elif fformat == WIT:
        reader = WitReader()
    elif fformat in DIALOGFLOW_RELEVANT:
        reader = DialogflowReader()
    elif fformat == RASA:
        reader = RasaReader()
    elif fformat == RASA_YAML:
        reader = RasaYAMLReader()
    return reader


def _load(filename: Text, language: Optional[Text] = "en") -> Optional["TrainingData"]:
    """Loads a single training data file from disk."""
    #yd。功能：如果filename是RASA_YAML格式，则读取文件的内容。
    # 例如如果filename是"data/nlu.yml"，则读取文件内容，解析出nlu意图和实体训练数据，并将这些信息保存在TrainingData类对象中
    fformat = guess_format(filename) #yd。获取file_name对应文件的类型，是UNK还是RASA_YAML
    if fformat == UNK:
        raise ValueError(f"Unknown data format for file '{filename}'.")

    reader = _reader_factory(fformat) #yd。根据文件类型返回对应的reader

    if reader:
        return reader.read(filename, language=language, fformat=fformat)
    else:
        return None


def guess_format(filename: Text) -> Text:
    """
    yd。功能：判断file_name的文件类型是UNK，还是RASA_YAML
    :param filename:
    :return:
    """
    """Applies heuristics to guess the data format of a file.

    Args:
        filename: file whose type should be guessed

    Returns:
        Guessed file format.
    """
    from rasa.shared.nlu.training_data.formats import RasaYAMLReader

    guess = UNK

    if not os.path.isfile(filename): #yd。如果file_name所对应的不是一个文件，则返回UNK，即未知类型
        return guess

    try:
        print("\n-----为执行guess_format()方法，开始读取文件内容")
        content = rasa.shared.utils.io.read_file(filename)
        print("-----为执行guess_format()方法，完成读取文件内容\n")
        js = json.loads(content)
    except ValueError:
        if RasaYAMLReader.is_yaml_nlu_file(filename): #yd。如果file_name对应的文件中，含有关键字KEY_NLU 或 KEY_RESPONSES，则说明其类型为RASA_YAML
            guess = RASA_YAML
    else:
        for file_format, format_heuristic in _json_format_heuristics.items():
            if format_heuristic(js, filename):
                guess = file_format
                break

    logger.debug(f"Training data format of '{filename}' is '{guess}'.")

    return guess
