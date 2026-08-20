import numpy as np
import pytest
from rabbithole import labels, metrics, relate, vectors


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


def judged(rating, **scored):
    baseline = dict.fromkeys(metrics.HIGHER_IS_BETTER, 0.5)
    return labels.Judgement(
        parent="Jazz", picks=["Bebop"], rating=rating, scored=baseline | scored)


class TestOrientation:

    def test_a_penalty_metric_is_flipped_and_a_reward_metric_is_not(self):
        features = metrics.oriented({"lexical_leak": 0.25, "grounding": 0.25})

        assert features["lexical_leak"] == 0.75
        assert features["grounding"] == 0.25

    def test_a_missing_metric_reads_as_its_worst_value(self):
        features = metrics.oriented({})

        assert features["grounding"] == 0.0
        assert features["lexical_leak"] == 1.0


class TestSpearman:

    @pytest.mark.parametrize(
        "right,expected", [([1, 2, 3, 4], 1.0), ([4, 3, 2, 1], -1.0)])
    def test_monotone_agreement_reads_plus_or_minus_one(self, right, expected):
        assert metrics.spearman([1, 2, 3, 4], right) == pytest.approx(expected)

    def test_it_is_rank_based_so_scale_does_not_matter(self):
        assert metrics.spearman([1, 2, 3], [10, 1000, 100000]) == pytest.approx(1.0)

    def test_a_flat_series_agrees_with_nothing(self):
        assert metrics.spearman([1, 1, 1], [1, 2, 3]) == 0.0

    @pytest.mark.parametrize("left", [[], [1], [1, 2, 3]])
    def test_too_little_or_mismatched_data_is_zero_not_a_guess(self, left):
        assert metrics.spearman(left, [1, 2]) == 0.0


class TestFitWeights:

    def test_weights_stay_on_the_simplex(self):
        rng = np.random.default_rng(0)
        matrix = rng.random((25, 4))
        fitted = metrics.fit_weights(matrix, rng.random(25))

        assert fitted.sum() == pytest.approx(1.0)
        assert (fitted >= 0.0).all()

    def test_the_feature_that_drives_the_rating_takes_the_weight(self):
        values = np.linspace(0.0, 1.0, 20)
        matrix = np.column_stack([values, np.full(20, 0.5)])

        fitted = metrics.fit_weights(matrix, values)

        assert fitted[0] > 0.9


class TestCalibration:

    def test_it_says_so_when_nobody_has_judged_anything(self):
        calibration = metrics.calibrate([])

        assert not calibration.trusted
        assert "uncalibrated" in calibration.verdict()
        assert calibration.weights == metrics.QUALITY_WEIGHTS

    def test_a_handful_of_labels_is_still_not_a_calibration(self):
        calibration = metrics.calibrate([judged(v) for v in np.linspace(0, 1, 5)])

        assert calibration.samples == 5
        assert not calibration.trusted
        assert f"5/{metrics.MINIMUM_LABELS}" in calibration.verdict()

    def test_it_catches_hand_set_weights_pointing_the_wrong_way(self):
        values = np.linspace(0.0, 1.0, 40) # Rating tracks yield
        judgements = [judged(v, yield_rate=v, lexical_leak=v) for v in values]

        calibration = metrics.calibrate(judgements)

        assert calibration.baseline < 0.0
        assert calibration.agreement > 0.9
        assert calibration.weights["yield_rate"] > metrics.QUALITY_WEIGHTS["yield_rate"]

    def test_enough_agreeing_labels_earns_trust(self):
        values = np.linspace(0.0, 1.0, metrics.MINIMUM_LABELS)
        calibration = metrics.calibrate([judged(v, grounding=v) for v in values])

        assert calibration.trusted
        assert "calibrated on" in calibration.verdict()

    def test_labels_that_agree_with_nothing_are_not_trusted(self):
        ratings = [0.0, 1.0] * 20
        judgements = [
            judged(rating, grounding=value)
            for rating, value in zip(ratings, np.linspace(0, 1, 40), strict=True)]

        calibration = metrics.calibrate(judgements)

        assert calibration.samples == 40
        assert not calibration.trusted
        assert "do not capture" in calibration.verdict()

    def test_judgements_without_stored_metrics_cannot_calibrate_anything(self):
        calibration = metrics.calibrate(
            [labels.Judgement(parent="Jazz", picks=[], rating=1.0) for _ in range(40)])

        assert calibration.samples == 0
        assert not calibration.trusted


class TestQualityWeighting:

    def test_calibrated_weights_can_replace_the_hand_set_ones(self):
        scored = dict.fromkeys(metrics.HIGHER_IS_BETTER, 0.0) | {"yield_rate": 1.0}

        assert metrics.quality(scored) < metrics.quality(scored, {"yield_rate": 1.0})

    def test_aggregate_honours_the_weights_it_is_given(self):
        rows = [dict.fromkeys(metrics.HIGHER_IS_BETTER, 0.0) | {"grounding": 1.0}]

        assert metrics.aggregate(rows, {"grounding": 1.0})["quality"] == pytest.approx(1.0)


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
