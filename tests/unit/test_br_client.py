import pytest
import requests

from src.data.br_client import fetch_season_html


class DummyResp:
    def __init__(self, text, status=200):
        self.text = text
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}")


def test_fetch_season_html_retries_and_succeeds(monkeypatch):
    calls = {"n": 0}

    def fake_get(url, headers=None, timeout=None):
        calls["n"] += 1
        if calls["n"] < 3:
            raise requests.ConnectionError("boom")
        return DummyResp("<html>ok</html>")

    # stub requests.get and time.sleep (so test is instant)
    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr("time.sleep", lambda *_: None, raising=False)

    html = fetch_season_html(2024, retries=3, timeout=1)
    assert "ok" in html
    assert calls["n"] == 3


def test_fetch_season_html_gives_up(monkeypatch):
    def always_fail(url, headers=None, timeout=None):
        raise requests.Timeout("nope")

    monkeypatch.setattr(requests, "get", always_fail, raising=True)
    monkeypatch.setattr("time.sleep", lambda *_: None, raising=True)

    with pytest.raises(RuntimeError) as excinfo:
        fetch_season_html(2024, retries=2, timeout=1)

    assert str(excinfo.value)  # has some message
