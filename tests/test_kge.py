import numpy as np
import pytest
from rabbithole import kge, wikidata


def two_clusters(size=6):
    """Two families of entities, each defined by the relations they share."""
    triples = []
    for index in range(size):
        triples.append((f"Qmusic{index}", "P136", "Qgenre"))
        triples.append((f"Qmusic{index}", "P135", "Qmovement"))
        triples.append((f"Qrock{index}", "P31", "Qmineral"))
        triples.append((f"Qrock{index}", "P17", "Qplace"))

    return triples


class TestTrain:

    def test_entities_reached_the_same_way_end_up_near_each_other(self):
        trained = kge.train(two_clusters(), dim=16, epochs=300)

        together = float(trained["Qmusic0"] @ trained["Qmusic1"])
        apart = float(trained["Qmusic0"] @ trained["Qrock0"])

        assert together > apart

    def test_vectors_come_back_unit_length_so_dot_products_are_cosines(self):
        trained = kge.train(two_clusters(), dim=16, epochs=50)

        assert all(
            float(np.linalg.norm(vector)) == pytest.approx(1.0, abs=1e-5)
            for vector in trained.values())

    def test_the_same_seed_gives_the_same_graph_twice(self):
        first = kge.train(two_clusters(), dim=8, epochs=40, seed=7)
        second = kge.train(two_clusters(), dim=8, epochs=40, seed=7)

        assert all(
            first[key] == pytest.approx(second[key], abs=1e-6) for key in first)

    def test_every_entity_in_the_triples_gets_a_vector(self):
        trained = kge.train([("A", "P31", "B"), ("C", "P31", "B")], dim=8, epochs=10)

        assert set(trained) == {"A", "B", "C"}

    @pytest.mark.parametrize(
        "triples", [[], [("A", "P31", "A")]])
    def test_a_graph_with_nothing_to_learn_trains_nothing(self, triples):
        assert kge.train(triples, dim=8, epochs=10) == {}

    def test_a_degenerate_dimension_is_refused_rather_than_guessed_at(self):
        assert kge.train(two_clusters(), dim=1, epochs=10) == {}


class TestEmbed:

    def test_titles_are_carried_through_to_their_vectors(self):
        graph = wikidata.Graph(
            qids={"Jazz": "Qmusic0", "Bebop": "Qmusic1", "Granite": "Qrock0"},
            triples=two_clusters(),
            parent_qid="Qmusic0",)

        structure = kge.embed(graph, dim=16, epochs=200)

        assert set(structure.by_title) == {"Jazz", "Bebop", "Granite"}
        assert structure.parent is not None

    def test_the_parent_is_closer_to_its_own_cluster(self):
        graph = wikidata.Graph(
            qids={"Jazz": "Qmusic0", "Bebop": "Qmusic1", "Granite": "Qrock0"},
            triples=two_clusters(),
            parent_qid="Qmusic0",)

        structure = kge.embed(graph, dim=16, epochs=300)

        assert structure.similarity("Bebop") > structure.similarity("Granite")

    def test_an_empty_graph_is_no_structure_rather_than_a_crash(self):
        structure = kge.embed(wikidata.Graph())

        assert structure.by_title == {}
        assert structure.parent is None
        assert structure.similarity("Anything") is None

    def test_a_title_outside_the_graph_has_no_similarity(self):
        graph = wikidata.Graph(
            qids={"Jazz": "Qmusic0"}, triples=two_clusters(), parent_qid="Qmusic0")

        structure = kge.embed(graph, dim=8, epochs=20)

        assert structure.similarity("Unmapped") is None
