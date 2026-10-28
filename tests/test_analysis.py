import unittest

from mike import text
from mike.analysis import analyze, answered_as, chain_of, opening_identifiers
from mike.evaluate import evaluate_one
from mike.models import ScriptedModel
from mike.world import generate


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.world = generate(42, 6)
        self.task = self.world["tasks"][0]
        self.chain = chain_of(self.task, self.world["records"])
        self.opening = text.opening("text", "tools")

    def run_lines(self, lines):
        return evaluate_one(ScriptedModel(lines), self.task, self.world["records"], True, 6,
                            interface="text", opening=self.opening)

    def test_answered_as_names_the_hop_or_kind(self):
        self.assertEqual(answered_as(self.task["answer"], self.chain), "property")
        self.assertEqual(answered_as(self.chain[1]["object"], self.chain), "station")
        self.assertEqual(answered_as(self.chain[2]["id"], self.chain), "chain_record_id")
        example_ids = opening_identifiers(text.opening("text-example", "tools"))
        self.assertIn("F5F0943517C5BF050", example_ids)
        self.assertIn("OC466B41C0A7A", example_ids)
        self.assertEqual(answered_as("PB716400BF4D4", self.chain, example_ids=example_ids), "example_identifier")
        self.assertEqual(answered_as("PB716400BF4D4", self.chain), "other")
        self.assertEqual(answered_as(None, self.chain), "unknown")

    def test_early_stop_at_station_with_no_citations(self):
        trace = self.run_lines([f"search {self.chain[0]['subject']}", f"search {self.chain[1]['subject']}",
                                f"answer {self.chain[1]['object']}"])
        result = analyze(self.world, [trace])
        self.assertEqual(result["answered_as"], {"station": 1})
        self.assertEqual(result["chain_depth_searched"], {"2": 1})
        self.assertEqual(result["citations"], {"no_citations": 1})
        self.assertEqual(result["first_action"], {"search_start": 1})

    def test_exact_and_example_citations(self):
        searches = [f"search {r['subject']}" for r in self.chain]
        exact = self.run_lines(searches + [f"answer {self.task['answer']} because " + " ".join(self.task["evidence"])])
        copied = self.run_lines(searches + [f"answer {self.task['answer']} because F5F0943517C5BF050"])
        result = analyze(self.world, [exact, copied], text.opening("text-example", "tools"))
        self.assertEqual(result["citations"], {"exact": 1, "cites_example_ids": 1})
        self.assertEqual(result["citation_tokens"], {"expected_record": 4, "example_identifier": 1})
        self.assertEqual(result["chain_depth_searched"], {"4": 2})

    def test_region_cited_as_record(self):
        searches = [f"search {r['subject']}" for r in self.chain]
        region = self.chain[2]["object"]
        trace = self.run_lines(searches + [f"answer {self.task['answer']} because {self.chain[3]['id']}, {region}"])
        result = analyze(self.world, [trace])
        self.assertEqual(result["citation_tokens"], {"expected_record": 1, "region_entity": 1})
        self.assertEqual(result["citations"], {"cites_unobserved": 1})


if __name__ == "__main__":
    unittest.main()
