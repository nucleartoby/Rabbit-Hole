import pytest
from rabbithole import core, sources
from rabbithole.core import dig


@pytest.fixture
def wikipedia(monkeypatch):

    def install(hits, detail):
        monkeypatch.setattr(sources, "search", lambda term, limit: hits)
        monkeypatch.setattr(sources, "pages", lambda titles: detail)

    return install


def test_best_match_leads_and_the_rest_follow(wikipedia):
    wikipedia(
        hits=[{"title": "Car"}, {"title": "Cable car"}, {"title": "The Cars"}],
        detail={
            "Car": {"title": "Car", "extract": "A car is a motor vehicle.", "fullurl": "u/car"},
            "Cable car": {"title": "Cable car", "extract": "A cable car is."},
            "The Cars": {"title": "The Cars", "extract": "An American band."},},)

    hole = dig("car")

    assert hole.term == "car"
    assert hole.top.title == "Car"
    assert hole.top.summary == "A car is a motor vehicle."
    assert hole.top.url == "u/car"
    assert [e.title for e in hole.more] == ["Cable car", "The Cars"]


def test_falls_back_to_search_snippet_with_markup_stripped(wikipedia):
    wikipedia(hits=[{"title": "Obscure", "snippet": 'an <span class="hit">obscure</span> thing'}], detail={},)

    hole = dig("obscure")

    assert hole.top.summary == "an obscure thing"
    assert hole.top.url == "https://en.wikipedia.org/wiki/Obscure"
    assert hole.top.thumbnail is None


def test_thumbnail_is_lifted_when_present(wikipedia):
    wikipedia(hits=[{"title": "Car"}], detail={"Car": {"title": "Car", "extract": "x", "thumbnail": {"source": "img.jpg"}}},)

    assert dig("car").top.thumbnail == "img.jpg"


@pytest.mark.parametrize("term", ["", "   "])
def test_blank_term_digs_nothing(term, wikipedia):
    wikipedia(hits=[{"title": "Should not be reached"}], detail={})

    hole = dig(term)

    assert hole.top is None
    assert hole.more == []


def test_no_results_is_an_empty_hole_not_an_error(wikipedia):
    wikipedia(hits=[], detail={})

    hole = dig("zxcvbnmqwerty")

    assert hole.top is None
    assert hole.more == []


def test_url_is_built_from_the_title_when_wikipedia_omits_one():
    assert core._url_for("Cable car") == "https://en.wikipedia.org/wiki/Cable_car"
