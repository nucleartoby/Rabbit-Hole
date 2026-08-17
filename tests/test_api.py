import pytest
from fastapi.testclient import TestClient
from rabbithole import api
from rabbithole.core import Hole

client = TestClient(api.app)


@pytest.fixture
def digs(monkeypatch):
    calls = []

    def install(hole):
        def fake(term, limit, stage=None, path=None, exclude=None, taste=None):
            calls.append({
                "term": term, "limit": limit, "stage": stage,
                "path": path, "exclude": exclude, "taste": taste,})
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


def test_taste_is_passed_through_for_the_adaptive_band(digs, entry):
    calls = digs(Hole(term="car", top=entry("Car"), more=[]))

    client.get("/api/dig", params={"term": "car", "taste": 0.3})

    assert calls[0]["taste"] == pytest.approx(0.3)


def test_absent_taste_leaves_the_band_alone(digs, entry):
    calls = digs(Hole(term="car", top=entry("Car"), more=[]))

    client.get("/api/dig", params={"term": "car"})

    assert calls[0]["taste"] is None


@pytest.mark.parametrize("taste", [-0.1, 1.4])
def test_taste_is_bounded_to_a_quantile(taste):
    response = client.get("/api/dig", params={"term": "car", "taste": taste})

    assert response.status_code == 422


def test_why_survives_serialisation(digs):
    from rabbithole.core import Entry

    digs(
        Hole(
            term="jazz",
            top=Entry(title="Jazz", summary="s", url="u"),
            more=[Entry(title="Bebop", summary="s", url="u", why="same movement")],))

    body = client.get("/api/dig", params={"term": "jazz"}).json()

    assert body["more"][0]["why"] == "same movement"


@pytest.mark.parametrize("stage", ["session", "typed", "kge"])
def test_new_stages_are_accepted(digs, entry, stage):
    calls = digs(Hole(term="car", top=entry("Car"), more=[], stage=stage))

    response = client.get("/api/dig", params={"term": "car", "stage": stage})

    assert response.status_code == 200
    assert calls[0]["stage"] == stage


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
