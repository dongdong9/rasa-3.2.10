from pathlib import Path
from typing import Text

import pytest

from rasa.cli import scaffold
from rasa.shared.importers.importer import TrainingDataImporter


@pytest.mark.parametrize(
    "config_file, domain_file, data_folder, raise_slot_warning",
    [
        (
            "examples/concertbot/config.yml",
            "examples/concertbot/domain.yml",
            "examples/concertbot/data",
            True,
        ),
        (
            "examples/formbot/config.yml",
            "examples/formbot/domain.yml",
            "examples/formbot/data",
            True,
        ),
        (
            "examples/knowledgebasebot/config.yml",
            "examples/knowledgebasebot/domain.yml",
            "examples/knowledgebasebot/data",
            True,
        ),
        (
            "data/test_moodbot/config.yml",
            "data/test_moodbot/domain.yml",
            "data/test_moodbot/data",
            False,
        ),
        (
            "examples/reminderbot/config.yml",
            "examples/reminderbot/domain.yml",
            "examples/reminderbot/data",
            True,
        ),
        (
            "examples/rules/config.yml",
            "examples/rules/domain.yml",
            "examples/rules/data",
            True,
        ),
    ],
)
def test_example_bot_training_data_raises_only_auto_fill_warning(
    config_file: Text,
    domain_file: Text,
    data_folder: Text,
    raise_slot_warning: bool,
):

    importer = TrainingDataImporter.load_from_config(
        config_file, domain_file, [data_folder]
    )

    if raise_slot_warning:
        with pytest.warns(UserWarning) as record:
            importer.get_nlu_data()
            importer.get_stories()

        assert len(record) == 2
        assert all(
            [
                "Slot auto-fill has been removed in 3.0 and replaced with "
                "a new explicit mechanism to set slots." in r.message.args[0]
                for r in record
            ]
        )
    else:
        with pytest.warns(None) as record:
            importer.get_nlu_data()
            importer.get_stories()

        assert len(record) == 0


def test_example_bot_training_on_initial_project(tmp_path: Path):
    # we need to test this one separately, as we can't test it in place
    # configuration suggestions would otherwise change the initial file
    scaffold.create_initial_project(str(tmp_path))

    importer = TrainingDataImporter.load_from_config(
        str(tmp_path / "config.yml"),
        str(tmp_path / "domain.yml"),
        str(tmp_path / "data"),
    )

    with pytest.warns(None) as record:
        importer.get_nlu_data()
        importer.get_stories()

    assert len(record) == 0
