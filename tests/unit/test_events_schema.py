import json

import pytest
from events_helpers import SCHEMA_PATH, make_event
from jsonschema import Draft202012Validator

from src.events import handler


def test_runtime_schema_matches_contract_copy():
    contract = json.loads(SCHEMA_PATH.read_text())
    assert handler.load_schema() == contract


@pytest.mark.parametrize("kind", ["SCHEDULED", "IN_PROGRESS", "FINAL", "POSTPONED", "CANCELED"])
def test_sample_events_match_contract(kind):
    schema = json.loads(SCHEMA_PATH.read_text())
    Draft202012Validator(schema).validate(make_event(status_kind=kind))


def test_bad_sample_fails_contract():
    schema = json.loads(SCHEMA_PATH.read_text())
    bad = make_event()
    del bad["game"]["homeTeam"]
    assert list(Draft202012Validator(schema).iter_errors(bad))
