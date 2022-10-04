import sys
import argparse
from pathlib import Path

import pytest
from typing import Callable, Dict

from _pytest.monkeypatch import MonkeyPatch
from _pytest.pytester import RunResult
from aioresponses import aioresponses

import rasa.shared.utils.io
from rasa.cli import x
from rasa.utils.endpoints import EndpointConfig
from rasa.core.utils import AvailableEndpoints
import rasa.version


def test_x_help(run: Callable[..., RunResult]):
    output = run("x", "--help")

    if sys.version_info.minor >= 9:
        # This is required because `argparse` behaves differently on
        # Python 3.9 and above. The difference is the changed formatting of help
        # output for CLI arguments with `nargs="*"
        version_dependent = """[-i INTERFACE] [-p PORT] [-t AUTH_TOKEN] [--cors [CORS ...]]
              [--enable-api] [--response-timeout RESPONSE_TIMEOUT]"""
    else:
        version_dependent = """[-i INTERFACE] [-p PORT] [-t AUTH_TOKEN]
              [--cors [CORS [CORS ...]]] [--enable-api]
              [--response-timeout RESPONSE_TIMEOUT]"""

    help_text = (
        """usage: rasa x [-h] [-v] [-vv] [--quiet] [-m MODEL] [--data DATA [DATA ...]]
              [-c CONFIG] [-d DOMAIN] [--no-prompt] [--production]
              [--rasa-x-port RASA_X_PORT] [--config-endpoint CONFIG_ENDPOINT]
              [--log-file LOG_FILE] [--use-syslog]
              [--syslog-address SYSLOG_ADDRESS] [--syslog-port SYSLOG_PORT]
              [--syslog-protocol SYSLOG_PROTOCOL] [--endpoints ENDPOINTS]
              """
        + version_dependent
        + """
              [--remote-storage REMOTE_STORAGE]
              [--ssl-certificate SSL_CERTIFICATE] [--ssl-keyfile SSL_KEYFILE]
              [--ssl-ca-file SSL_CA_FILE] [--ssl-password SSL_PASSWORD]
              [--credentials CREDENTIALS] [--connector CONNECTOR]
              [--jwt-secret JWT_SECRET] [--jwt-method JWT_METHOD]"""
    )

    lines = help_text.split("\n")
    # expected help text lines should appear somewhere in the output
    printed_help = set(output.outlines)
    for line in lines:
        assert line in printed_help


def test_prepare_credentials_for_rasa_x_if_rasa_channel_not_given(tmpdir: Path):
    credentials_path = str(tmpdir / "credentials.yml")

    rasa.shared.utils.io.write_yaml({}, credentials_path)

    tmp_credentials = x._prepare_credentials_for_rasa_x(
        credentials_path, "http://localhost:5002"
    )

    actual = rasa.shared.utils.io.read_config_file(tmp_credentials)

    assert actual["rasa"]["url"] == "http://localhost:5002"


def test_prepare_credentials_if_already_valid(tmpdir: Path):
    credentials_path = str(tmpdir / "credentials.yml")

    credentials = {
        "rasa": {"url": "my-custom-url"},
        "another-channel": {"url": "some-url"},
    }
    rasa.shared.utils.io.write_yaml(credentials, credentials_path)

    x._prepare_credentials_for_rasa_x(credentials_path)

    actual = rasa.shared.utils.io.read_config_file(credentials_path)

    assert actual == credentials


def test_if_default_endpoint_config_is_valid_in_local_mode():
    event_broker_endpoint = x._get_event_broker_endpoint(None)

    assert x._is_correct_event_broker(event_broker_endpoint)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"type": "mongo", "url": "mongodb://localhost:27017"},
        {"type": "sql", "dialect": "postgresql"},
        {"type": "sql", "dialect": "sqlite", "db": "some.db"},
    ],
)
def test_if_endpoint_config_is_invalid_in_local_mode(kwargs: Dict):
    config = EndpointConfig(**kwargs)
    assert not x._is_correct_event_broker(config)


def test_overwrite_model_server_url():
    endpoint_config = EndpointConfig(url="http://testserver:5002/models/default@latest")
    endpoints = AvailableEndpoints(model=endpoint_config)
    x._overwrite_endpoints_for_local_x(endpoints, "test", "http://localhost")
    assert endpoints.model.url == "http://localhost/models/tags/production"


def test_overwrite_model_server_url_with_no_model_endpoint():
    endpoints = AvailableEndpoints()
    x._overwrite_endpoints_for_local_x(endpoints, "test", "http://localhost")
    assert endpoints.model.url == "http://localhost/models/tags/production"


def test_reuse_wait_time_between_pulls():
    test_wait_time = 5
    endpoint_config = EndpointConfig(
        url="http://localhost:5002/models/default@latest",
        wait_time_between_pulls=test_wait_time,
    )
    endpoints = AvailableEndpoints(model=endpoint_config)
    assert endpoints.model.kwargs["wait_time_between_pulls"] == test_wait_time


def test_default_wait_time_between_pulls():
    endpoint_config = EndpointConfig(url="http://localhost:5002/models/default@latest")
    endpoints = AvailableEndpoints(model=endpoint_config)
    x._overwrite_endpoints_for_local_x(endpoints, "test", "http://localhost")
    assert endpoints.model.kwargs["wait_time_between_pulls"] == 2


def test_default_model_server_url():
    endpoint_config = EndpointConfig()
    endpoints = AvailableEndpoints(model=endpoint_config)
    x._overwrite_endpoints_for_local_x(endpoints, "test", "http://localhost")
    assert endpoints.model.url == "http://localhost/models/tags/production"


async def test_pull_runtime_config_from_server():
    config_url = "http://example.com/api/config?token=token"
    credentials = "rasa: http://example.com:5002/api"
    endpoint_config = """
    event_broker:
        url: http://example.com/event_broker
        username: some_username
        password: PASSWORD
        queue: broker_queue
    """
    with aioresponses() as mocked:
        mocked.get(
            config_url,
            payload={"credentials": credentials, "endpoints": endpoint_config},
        )

        endpoints_path, credentials_path = await x._pull_runtime_config_from_server(
            config_url, 1, 0
        )

        assert rasa.shared.utils.io.read_file(endpoints_path) == endpoint_config
        assert rasa.shared.utils.io.read_file(credentials_path) == credentials


def test_rasa_x_raises_warning_above_version_3(monkeypatch: MonkeyPatch):
    def mock_run_locally(args):
        return None

    monkeypatch.setattr(x, "run_locally", mock_run_locally)

    args = argparse.Namespace(loglevel=None, log_file=None, production=None)
    with pytest.warns(
        UserWarning,
        match=f"Your version of rasa '{rasa.version.__version__}' is currently "
        f"not supported by Rasa X. Running `rasa x` CLI command with rasa "
        f"version higher or equal to 3.0.0 will result in errors.",
    ):
        x.rasa_x(args)

    monkeypatch.setattr(target=rasa.version, name="__version__", value="2.8.0")
    with pytest.warns(None):
        x.rasa_x(args)


def test_rasa_x_does_not_raise_warning_above_version_3_with_production_flag(
    monkeypatch: MonkeyPatch,
):
    def mock_run_in_production(args):
        return None

    monkeypatch.setattr(x, "run_in_production", mock_run_in_production)

    args = argparse.Namespace(loglevel=None, log_file=None, production=True)

    with pytest.warns(None):
        x.rasa_x(args)
