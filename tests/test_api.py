import pytest
from fastapi.testclient import TestClient
from rabbithole import api
from rabbithole.core import Hole

client = TestClient(api.app)


@pytest.fixture
def digs(monkeypatch):

    def install(hole):
        monkeypatch.setattr(api, "dig", lambda term, limit: hole)

    return install


def test_dig_returns_the_top_result_and_the_tail(digs, entry):
    digs(Hole(term="car", top=entry("Car"), more=[entry("Cable car")]))

    response = client.get("/api/dig", params={"term": "car"})

    assert response.status_code == 200
    body = response.json()
    assert body["term"] == "car"
    assert body["top"]["title"] == "Car"
    assert [item["title"] for item in body["more"]] == ["Cable car"]


def test_nothing_found_is_a_404(digs):
    digs(Hole(term="zxcvbnm", top=None, more=[]))

    response = client.get("/api/dig", params={"term": "zxcvbnm"})

    assert response.status_code == 404


def test_term_is_required():
    assert client.get("/api/dig").status_code == 422


@pytest.mark.parametrize("limit", [0, 51])
def test_limit_is_bounded(limit):
    response = client.get("/api/dig", params={"term": "car", "limit": limit})

    assert response.status_code == 422


def test_health_reports_the_version():
    body = client.get("/health").json()

    assert body["status"] == "ok"
    assert body["version"]


def test_root_serves_the_canvas():
    response = client.get("/")

    assert response.status_code == 200
    assert "<title>Rabbit Hole</title>" in response.text


@pytest.mark.network
def test_live_wikipedia_lookup():
    body = client.get("/api/dig", params={"term": "tardigrade"}).json()

    assert body["top"]["title"] == "Tardigrade"
    assert len(body["more"]) > 1
