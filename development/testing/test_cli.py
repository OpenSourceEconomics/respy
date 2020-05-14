import pytest
from click.testing import CliRunner
from testing.regression import create
from testing.regression import investigate
from testing.regression import run


pytestmark = pytest.mark.xfail


def test_investigate():
    runner = CliRunner()
    result = runner.invoke(investigate, ["0"])

    assert result.exit_code == 0


def test_run():
    runner = CliRunner()
    result = runner.invoke(run, ["0", "--no-notification"])

    assert result.exit_code == 0


def test_create():
    runner = CliRunner()
    result = runner.invoke(create, ["1", "--no-save"])

    assert result.exit_code == 0
