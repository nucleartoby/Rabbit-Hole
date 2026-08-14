import pytest
from rabbithole import core, relate, sources
from rabbithole.core import dig


@pytest.fixture
def wikipedia(monkeypatch):
    def install(hits, detail, morelike=None, lead=None, categories=None):
        monkeypatch.setattr(sources, "search", lambda term, limit: hits)
        monkeypatch.setattr(sources, "pages", lambda titles: detail)
        monkeypatch.setattr(
            sources, "morelike", lambda title, limit: morelike or [])
        monkeypatch.setattr(sources, "lead_links", lambda title: lead or [])
        monkeypatch.setattr(
            sources, "categories", lambda titles: categories or {})

    return install


def test_best_match_leads_and_the_rest_follow(wikipedia):
    wikipedia(
        hits=[{"title": "Car"}, {"title": "Cable car"}, {"title": "The Cars"}],
        detail={
            "Car": {"title": "Car", "extract": "A car is a motor vehicle.", "fullurl": "u/car"},
            "Cable car": {"title": "Cable car", "extract": "A cable car is."},
            "The Cars": {"title": "The Cars", "extract": "An American band."},},)

    hole = dig("car", stage=relate.SEARCH)

    assert hole.term == "car"
    assert hole.top.title == "Car"
    assert hole.top.summary == "A car is a motor vehicle."
    assert hole.top.url == "u/car"
    assert [e.title for e in hole.more] == ["Cable car", "The Cars"]


def test_falls_back_to_search_snippet_with_markup_stripped(wikipedia):
    wikipedia(hits=[{"title": "Obscure", "snippet": 'an <span class="hit">obscure</span> thing'}], detail={},)

    hole = dig("obscure", stage=relate.SEARCH)

    assert hole.top.summary == "an obscure thing"
    assert hole.top.url == "https://en.wikipedia.org/wiki/Obscure"
    assert hole.top.thumbnail is None


def test_thumbnail_is_lifted_when_present(wikipedia):
    wikipedia(hits=[{"title": "Car"}], detail={"Car": {"title": "Car", "extract": "x", "thumbnail": {"source": "img.jpg"}}},)

    assert dig("car", stage=relate.SEARCH).top.thumbnail == "img.jpg"


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


def test_unknown_stage_is_rejected(wikipedia):
    wikipedia(hits=[{"title": "Car"}], detail={})

    with pytest.raises(ValueError, match="unknown stage"):
        dig("car", stage="vibes")


class TestCandidateStage:

    def test_lexical_variants_of_the_parent_are_dropped(self, wikipedia):
        wikipedia(
            hits=[{"title": "Car"}],
            detail={
                "Car": {"title": "Car", "extract": "A car is a motor vehicle."},
                "Cable car": {"title": "Cable car", "extract": "A cable car."},
                "The Cars": {"title": "The Cars", "extract": "An American band."},
                "Automobile": {"title": "Automobile", "extract": "A road vehicle."},},
            morelike=[
                {"title": "Cable car"}, {"title": "The Cars"}, {"title": "Automobile"}],)

        titles = [entry.title for entry in dig("car", stage=relate.CANDIDATES).more]

        assert titles == ["Automobile"]

    def test_junk_titles_are_dropped(self, wikipedia):
        wikipedia(
            hits=[{"title": "Jazz"}],
            detail={
                "Jazz": {"title": "Jazz", "extract": "A music genre."},
                "Blues": {"title": "Blues", "extract": "A music genre."},},
            morelike=[
                {"title": "List of jazz musicians"},
                {"title": "Jazz (disambiguation)"},
                {"title": "Outline of jazz"},
                {"title": "Blues"},],)

        assert [e.title for e in dig("jazz", stage=relate.CANDIDATES).more] == ["Blues"]

    def test_lead_links_and_morelike_are_fused_not_concatenated(self, wikipedia):
        wikipedia(
            hits=[{"title": "Jazz"}],
            detail={
                title: {"title": title, "extract": f"About {title}."}
                for title in ("Jazz", "Ragtime", "Blues", "Swing")},
            morelike=[{"title": "Ragtime"}, {"title": "Blues"}],
            lead=["Swing", "Blues"],)

        titles = [entry.title for entry in dig("jazz", stage=relate.CANDIDATES).more]

        assert titles[0] == "Blues"


class TestExclusions:

    @pytest.fixture
    def music(self, wikipedia):
        titles = ("Jazz", "Blues", "Bebop", "Ragtime", "Swing")
        wikipedia(
            hits=[{"title": t} for t in titles],
            detail={t: {"title": t, "extract": f"Music about {t} and rhythm."} for t in titles},
            morelike=[{"title": t} for t in titles[1:]],)

    @pytest.mark.parametrize("stage", [relate.SEARCH, relate.CANDIDATES, relate.RANKED])
    def test_excluded_titles_are_never_offered(self, music, stage):
        hole = dig("jazz", stage=stage, exclude=["Blues", "Bebop"])

        assert {"Blues", "Bebop"}.isdisjoint(entry.title for entry in hole.more)

    @pytest.mark.parametrize("stage", [relate.CANDIDATES, relate.RANKED])
    def test_a_blocked_slot_is_backfilled_not_left_empty(self, music, stage):
        full = dig("jazz", stage=stage, limit=3)
        trimmed = dig("jazz", stage=stage, limit=3, exclude=[full.more[0].title])

        assert len(trimmed.more) == len(full.more)

    @pytest.mark.parametrize("stage", [relate.SEARCH, relate.CANDIDATES, relate.RANKED])
    def test_the_searched_page_itself_survives_exclusion(self, music, stage):
        hole = dig("jazz", stage=stage, exclude=["Jazz"])

        assert hole.top is not None
        assert hole.top.title == "Jazz"

    @pytest.mark.parametrize(
        "spelling", ["blues", "BLUES", "Blues_", " Blues ", "blues_"])
    def test_exclusion_survives_casing_and_underscores(self, music, spelling):
        hole = dig("jazz", stage=relate.CANDIDATES, exclude=[spelling])

        assert "Blues" not in [entry.title for entry in hole.more]

    def test_no_exclusions_changes_nothing(self, music):
        assert (
            [e.title for e in dig("jazz", stage=relate.CANDIDATES, exclude=[]).more]
            == [e.title for e in dig("jazz", stage=relate.CANDIDATES).more])

    def test_the_parent_never_offers_itself_as_a_child(self, wikipedia):
        wikipedia(
            hits=[{"title": "Jazz"}],
            detail={
                t: {"title": t, "extract": f"About {t}."} for t in ("Jazz", "Blues")},
            morelike=[{"title": "Jazz"}, {"title": "Blues"}],)

        assert [e.title for e in dig("jazz", stage=relate.CANDIDATES).more] == ["Blues"]


class TestRankedStage:

    def test_near_duplicates_are_not_all_offered_at_once(self, wikipedia):
        clones = {
            f"Clone {i}": {"title": f"Clone {i}", "extract": "Steam locomotive boiler pressure."}
            for i in range(3)}
        wikipedia(
            hits=[{"title": "Train"}],
            detail={
                "Train": {"title": "Train", "extract": "A train runs on rails."},
                "Weather": {"title": "Weather", "extract": "Rain clouds wind forecast sky."},
                **clones,},
            morelike=[{"title": t} for t in [*clones, "Weather"]],)

        titles = [entry.title for entry in dig("train", stage=relate.RANKED, limit=2).more]

        assert "Weather" in titles

    def test_scores_are_reported_for_every_pick(self, wikipedia):
        wikipedia(
            hits=[{"title": "Jazz"}],
            detail={
                title: {"title": title, "extract": f"Music about {title} and rhythm."}
                for title in ("Jazz", "Blues", "Bebop")},
            morelike=[{"title": "Blues"}, {"title": "Bebop"}],)

        hole = dig("jazz", stage=relate.RANKED)

        assert set(hole.scores) == {entry.title for entry in hole.more}
        assert all(
            {"band", "novelty", "grounding", "crowding"} <= set(parts)
            for parts in hole.scores.values())
