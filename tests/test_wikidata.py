import pytest
from rabbithole import sources, wikidata


def claim(pid, value):
    return {
        "mainsnak": {
            "snaktype": "value",
            "property": pid,
            "datavalue": {
                "value": {"entity-type": "item", "id": value},
                "type": "wikibase-entityid"},}}


def blank(pid):
    return {"mainsnak": {"snaktype": "novalue", "property": pid}}


class TestParsing:

    def test_entity_ids_are_lifted_out_of_the_snak(self):
        assert wikidata._values([claim("P31", "Q5"), claim("P31", "Q7")]) == {"Q5", "Q7"}

    def test_a_statement_with_no_value_contributes_nothing(self):
        assert wikidata._values([blank("P31"), claim("P31", "Q5")]) == {"Q5"}

    def test_a_quantity_or_date_is_not_mistaken_for_an_entity(self):
        dated = {
            "mainsnak": {
                "snaktype": "value",
                "datavalue": {"value": {"time": "+1926-00-00T00:00:00Z"}, "type": "time"}}}

        assert wikidata._values([dated]) == set()

    def test_relations_outside_the_vocabulary_are_dropped(self):
        parsed = wikidata.parse(
            {"Q1": {"P31": [claim("P31", "Q5")], "P9999": [claim("P9999", "Q8")]}})

        assert parsed["Q1"] == {"P31": {"Q5"}}

    def test_a_property_that_yields_no_entity_is_not_kept_as_an_empty_set(self):
        parsed = wikidata.parse({"Q1": {"P31": [blank("P31")]}})

        assert parsed["Q1"] == {}


class TestEdges:

    def test_a_direct_claim_reads_forwards(self):
        edge = wikidata.edge_between({"P361": {"Q2"}}, {}, "Q1", "Q2")

        assert edge.label == "part of"
        assert edge.family == "part"

    def test_a_claim_pointing_back_is_read_from_the_parents_side(self):
        edge = wikidata.edge_between({}, {"P279": {"Q1"}}, "Q1", "Q2")

        assert edge.label == "a kind of"
        assert edge.key == "~P279"

    def test_a_shared_value_is_the_associative_jump(self):
        edge = wikidata.edge_between({"P135": {"Q9"}}, {"P135": {"Q9"}}, "Q1", "Q2")

        assert edge.label == "same movement"
        assert edge.key == "=P135"

    def test_a_direct_claim_outranks_a_shared_one(self):
        edge = wikidata.edge_between(
            {"P361": {"Q2"}, "P135": {"Q9"}}, {"P135": {"Q9"}}, "Q1", "Q2")

        assert edge.label == "part of"

    def test_unrelated_entities_get_no_edge(self):
        edge = wikidata.edge_between({"P31": {"Q5"}}, {"P31": {"Q7"}}, "Q1", "Q2")

        assert edge.key == ""
        assert edge.label == ""
        assert edge.family == wikidata.UNKNOWN

    def test_both_halves_of_a_part_whole_pair_are_one_axis_of_variety(self):
        assert wikidata.FAMILIES["P361"] == wikidata.FAMILIES["P527"]

    def test_every_labelled_relation_belongs_to_a_family(self):
        labelled = set(wikidata.FORWARD) | set(wikidata.BACKWARD) | set(wikidata.SHARED)

        assert labelled <= set(wikidata.FAMILIES)


class TestTriples:

    def test_claims_become_triples(self):
        found = wikidata.triples_from({"Q1": {"P31": {"Q5"}}}, cap=10)

        assert found == [("Q1", "P31", "Q5")]

    def test_the_cap_is_honoured(self):
        claims = {f"Q{i}": {"P31": {"Q5", "Q6"}} for i in range(20)}

        assert len(wikidata.triples_from(claims, cap=7)) == 7

    def test_extraction_is_deterministic_so_training_is_reproducible(self):
        claims = {"Q2": {"P31": {"Q5", "Q6"}}, "Q1": {"P361": {"Q9"}}}

        assert wikidata.triples_from(claims, cap=99) == wikidata.triples_from(claims, cap=99)


class TestBuild:

    @pytest.fixture
    def wiki(self, monkeypatch):
        def install(qids, entities):
            monkeypatch.setattr(sources, "qids", lambda titles: qids)
            monkeypatch.setattr(sources, "entities", lambda ids: entities)

        return install

    def test_titles_are_wired_up_to_their_relations(self, wiki):
        wiki(
            qids={"Jazz": "Q8341", "Bebop": "Q170972", "Granite": "Q43338"},
            entities={
                "Q8341": {"P136": [claim("P136", "Q638")]},
                "Q170972": {"P279": [claim("P279", "Q8341")]},
                "Q43338": {"P31": [claim("P31", "Q10931")]},},)

        graph = wikidata.build("Jazz", ["Bebop", "Granite"])

        assert graph.edge("Bebop").label == "a kind of"
        assert graph.edge("Granite").key == ""
        assert graph.parent_qid == "Q8341"

    def test_an_article_wikidata_does_not_know_yields_an_empty_graph(self, wiki):
        wiki(qids={"Bebop": "Q170972"}, entities={})

        graph = wikidata.build("Some Very New Page", ["Bebop"])

        assert graph.edges == {}
        assert graph.triples == []

    def test_the_parent_is_never_an_edge_to_itself(self, wiki):
        wiki(
            qids={"Jazz": "Q8341", "Jazz music": "Q8341"},
            entities={"Q8341": {"P136": [claim("P136", "Q638")]}},)

        graph = wikidata.build("Jazz", ["Jazz music"])

        assert "Jazz music" not in graph.edges

    def test_missing_titles_are_simply_absent(self, wiki):
        wiki(qids={"Jazz": "Q8341"}, entities={"Q8341": {}})

        graph = wikidata.build("Jazz", ["Nonexistent"])

        assert graph.edge("Nonexistent").label == ""
