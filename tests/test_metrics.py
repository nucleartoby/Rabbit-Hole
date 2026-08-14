import pytest
from rabbithole import metrics, relate, vectors


def expansion(parent, picks, **kwargs):
    texts = kwargs.pop("texts", None) or {pick: f"About {pick}." for pick in picks}
    return metrics.Expansion(
        parent=parent,
        parent_text=kwargs.pop("parent_text", f"About {parent}."),
        picks=picks,
        texts=texts,
        **kwargs)


class TestLexicalLeak:

    def test_catches_the_failure_it_exists_for(self):
        leak = metrics.lexical_leak(
            expansion("Car", ["Cable car", "The Cars", "Car (disambiguation)", "CAR-15"]))

        assert leak == 1.0

    def test_semantically_related_titles_do_not_count_as_leak(self):
        leak = metrics.lexical_leak(
            expansion("Car", ["Automobile", "Internal combustion engine", "Highway"]))

        assert leak == 0.0

    def test_is_a_proportion_not_a_count(self):
        assert metrics.lexical_leak(expansion("Car", ["Cable car", "Automobile"])) == 0.5

    def test_empty_picks_do_not_divide_by_zero(self):
        assert metrics.lexical_leak(expansion("Car", [])) == 0.0


class TestJunkRate:

    @pytest.mark.parametrize(
        "title",
        ["List of cars", "Car (disambiguation)", "Outline of jazz", "Timeline of flight"],)
    def test_flags_navigational_pages(self, title):
        assert metrics.junk_rate(expansion("Car", [title])) == 1.0

    def test_leaves_real_articles_alone(self):
        assert metrics.junk_rate(expansion("Car", ["Automobile", "Carl Benz"])) == 0.0


class TestGrounding:

    def test_rewards_shared_categories(self):
        scored = metrics.grounding(
            expansion(
                "Car", ["Truck"],
                parent_categories={"Category:Motor vehicles"},
                pick_categories={"Truck": {"Category:Motor vehicles"}},))

        assert scored == 1.0

    def test_is_zero_without_overlap(self):
        scored = metrics.grounding(
            expansion(
                "Car", ["Jazz"],
                parent_categories={"Category:Motor vehicles"},
                pick_categories={"Jazz": {"Category:American music"}},))

        assert scored == 0.0


class TestLinkGrounding:

    def test_counts_picks_the_parent_links_to(self):
        scored = metrics.link_grounding(
            expansion("Car", ["Wheel", "Jazz"], parent_links={"Wheel", "Road"}))

        assert scored == 0.5

    def test_is_zero_when_the_link_set_is_unknown(self):
        assert metrics.link_grounding(expansion("Car", ["Wheel"])) == 0.0


class TestYieldRate:

    def test_short_results_are_penalised(self):
        assert metrics.yield_rate(expansion("Car", ["Wheel", "Road"], requested=8)) == 0.25

    def test_does_not_exceed_one_when_over_delivering(self):
        assert metrics.yield_rate(expansion("Car", ["A", "B", "C"], requested=2)) == 1.0


class TestVectorMetrics:

    def test_redundancy_separates_identical_picks_from_varied_ones(self):
        same = metrics.score(
            expansion(
                "Music", ["A", "B", "C"],
                texts={key: "Jazz trumpet improvisation swing" for key in "ABC"},))
        varied = metrics.score(
            expansion(
                "Music", ["A", "B", "C"],
                texts={
                    "A": "Jazz trumpet improvisation swing",
                    "B": "Granite basalt volcanic rock formation",
                    "C": "Parliament election ballot democracy"},))

        assert same["redundancy"] > varied["redundancy"]

    def test_single_pick_has_no_pairwise_redundancy(self):
        assert metrics.score(expansion("Music", ["A"]))["redundancy"] == 0.0

    def test_band_rate_follows_the_backend_scale(self):
        low, high = vectors.band_for("tfidf")

        assert 0.0 < low < high < 1.0
        assert vectors.band_for("minilm") != vectors.band_for("tfidf")


class TestQuality:

    def test_orients_every_metric_so_higher_is_better(self):
        good = metrics.quality(
            {"lexical_leak": 0.0, "junk_rate": 0.0, "redundancy": 0.0,
             "grounding": 1.0, "link_grounding": 1.0, "band_rate": 1.0, "yield_rate": 1.0})
        bad = metrics.quality(
            {"lexical_leak": 1.0, "junk_rate": 1.0, "redundancy": 1.0,
             "grounding": 0.0, "link_grounding": 0.0, "band_rate": 0.0, "yield_rate": 0.0})

        assert good == pytest.approx(1.0)
        assert bad == pytest.approx(0.0)

    def test_weights_cover_every_metric_exactly_once(self):
        assert set(metrics.QUALITY_WEIGHTS) == set(metrics.HIGHER_IS_BETTER)
        assert sum(metrics.QUALITY_WEIGHTS.values()) == pytest.approx(1.0)


class TestCategoryAffinity:

    def test_word_overlap_survives_where_exact_names_collide_never(self):
        left = {"Category:Jazz musicians", "Category:American jazz composers"}
        right = {"Category:Jazz genres", "Category:Music of New Orleans"}

        assert relate._jaccard(left, right) == 0.0
        assert relate.category_affinity(left, right) > 0.0

    def test_scaffolding_words_do_not_manufacture_affinity(self):
        left = {"Category:1926 establishments in Germany"}
        right = {"Category:1889 establishments in France"}

        assert relate.category_affinity(left, right) == 0.0


class TestVectorizer:

    def test_rows_come_back_l2_normalised(self):
        matrix = vectors.get("tfidf").encode(["jazz music", "rock music", "granite rock"])

        assert all(abs(float((row @ row) ** 0.5) - 1.0) < 1e-5 for row in matrix)

    def test_unknown_backend_is_rejected(self):
        with pytest.raises(ValueError, match="unknown vectorizer"):
            vectors.get("word2vec")

    def test_tfidf_is_always_available(self):
        assert "tfidf" in vectors.available()
