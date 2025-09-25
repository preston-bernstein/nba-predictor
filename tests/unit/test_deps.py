import pytest

from src.service.deps import _resolve_for_df


def test_resolve_returns_code_when_present():
    teams = {"NYK","BOS"}
    assert _resolve_for_df("NYK", teams) == "NYK"

def test_resolve_maps_full_name_if_present():
    teams = {"NEW YORK KNICKS","BOSTON CELTICS"}
    assert _resolve_for_df("nyk", teams) == "NEW YORK KNICKS"

def test_resolve_errors_if_unknown_in_data():
    teams = {"NYK","BOS"}
    with pytest.raises(ValueError):
        _resolve_for_df("LAL", teams)