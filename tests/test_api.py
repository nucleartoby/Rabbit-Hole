import pytest
from fastapi.testclient import TestClient
from rabbithole import api
from rabbithole.core import Hole

client = TestClient(api.app)


@pytest.fixture
def digs(monkeypatch):
    calls = []

    def install(hole):
        def fake(term, limit, stage=None, path=None, exclude=None):
            calls.append({
                "term": term, "limit": limit, "stage": stage,
                "path": path, "exclude": exclude,})
            return hole

        monkeypatch.setattr(api, "dig", fake)
        return calls

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


def test_stage_is_passed_through_and_echoed(digs, entry):
    calls = digs(Hole(term="car", top=entry("Car"), more=[], stage="candidates"))

    body = client.get("/api/dig", params={"term": "car", "stage": "candidates"}).json()

    assert calls[0]["stage"] == "candidates"
    assert body["stage"] == "candidates"


def test_unknown_stage_is_rejected_before_any_lookup(digs, entry):
    calls = digs(Hole(term="car", top=entry("Car"), more=[]))

    response = client.get("/api/dig", params={"term": "car", "stage": "vibes"})

    assert response.status_code == 422
    assert calls == []


def test_visited_path_is_split_for_the_ranker(digs, entry):
    calls = digs(Hole(term="bebop", top=entry("Bebop"), more=[]))

    client.get("/api/dig", params={"term": "bebop", "path": "Jazz|Blues"})

    assert calls[0]["path"] == ["Jazz", "Blues"]


def test_empty_path_is_not_a_phantom_ancestor(digs, entry):
    calls = digs(Hole(term="car", top=entry("Car"), more=[]))

    client.get("/api/dig", params={"term": "car"})

    assert calls[0]["path"] == []


def test_exclusions_are_split_for_the_pool_filter(digs, entry):
    calls = digs(Hole(term="car", top=entry("Car"), more=[]))

    client.get(
        "/api/dig", params={"term": "car", "exclude": "Ford Taurus|Mercury Sable"})

    assert calls[0]["exclude"] == ["Ford Taurus", "Mercury Sable"]


def test_blank_exclude_entries_are_discarded(digs, entry):
    calls = digs(Hole(term="car", top=entry("Car"), more=[]))

    client.get("/api/dig", params={"term": "car", "exclude": "Ford Taurus||  |"})

    assert calls[0]["exclude"] == ["Ford Taurus"]


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
