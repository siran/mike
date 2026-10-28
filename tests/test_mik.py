import json
import unittest

from mik.evaluate import compare_runs, evaluate_one, summarize, symbolic_baseline
from mik.models import ScriptedModel
from mik.runtime import run
from mik.tools import Tools
from mik.world import canonical_hash, generate


class InstrumentTests(unittest.TestCase):
    def setUp(self):
        self.world = generate(42, 3)
        self.task = self.world["tasks"][0]

    def test_world_is_seeded_and_private_identifiers_change(self):
        self.assertEqual(canonical_hash(self.world), canonical_hash(generate(42, 3)))
        self.assertNotEqual(self.task["start"], generate(43, 3)["tasks"][0]["start"])

    def test_search_is_atomic_and_disabled_has_no_records(self):
        tools = Tools(self.world["records"])
        result = tools.execute({"action": "search", "entity": self.task["start"]})
        self.assertEqual(len(result["records"]), 1)
        self.assertEqual(result["records"][0]["relation"], "inside")
        self.assertNotIn("records", Tools(self.world["records"], False).execute({"action": "search", "entity": self.task["start"]}))

    def test_oracle_fixture_scores_chain_and_citations(self):
        by_id = {r["id"]: r for r in self.world["records"]}
        records = [by_id[i] for i in self.task["evidence"]]
        actions = [{"action": "search", "entity": r["subject"]} for r in records]
        actions.append({"action": "answer", "value": self.task["answer"], "evidence": self.task["evidence"]})
        trace = evaluate_one(ScriptedModel(actions), self.task, self.world["records"], True, 6)
        self.assertTrue(trace["score"]["correct"])
        self.assertTrue(trace["score"]["complete_evidence"])
        self.assertEqual(summarize([trace])["correct_with_complete_evidence"], 1)

    def test_citation_cannot_be_fabricated(self):
        action = {"action": "answer", "value": self.task["answer"], "evidence": self.task["evidence"]}
        trace = evaluate_one(ScriptedModel([action]), self.task, self.world["records"], True, 6)
        self.assertTrue(trace["score"]["correct"])
        self.assertFalse(trace["score"]["complete_evidence"])

    def test_budget_and_invalid_output(self):
        model = ScriptedModel([{"action": "search", "entity": "missing"}])
        self.assertEqual(run(model, "q", Tools([]), 1)["result"]["status"], "budget_exhausted")
        class Invalid:
            def decide(self, state):
                return {"raw": "not json"}
        self.assertEqual(run(Invalid(), "q", Tools([]))["result"]["status"], "invalid_action")

    def test_symbolic_baseline_is_transparent_and_tools_matter(self):
        self.assertEqual(symbolic_baseline(self.task, self.world["records"], True)["value"], self.task["answer"])
        self.assertIsNone(symbolic_baseline(self.task, self.world["records"], False)["value"])

    def test_measure_validation(self):
        self.assertEqual(Tools([]).execute({"action": "measure", "x": 2})["y"], 7)
        self.assertIn("error", Tools([]).execute({"action": "measure", "x": True}))

    def test_comparison_pairs_only_same_world_and_common_tasks(self):
        metadata = {"world_sha256": "one", "model": {}, "tools_enabled": True}
        yes = {"task_id": "a", "score": {"correct": True}}
        no = {"task_id": "a", "score": {"correct": False}}
        # compare_runs also summarizes traces, so provide the other measured fields.
        for item in (yes, no):
            item.update({"result": {"status": "answered"}, "steps": [], "wall_seconds": 0, "process_cpu_seconds": 0})
            item["score"].update({"complete_evidence": False, "abstained": False})
        comparison = compare_runs([("a", metadata, [yes], True), ("b", metadata, [no], False)])
        self.assertEqual(comparison["paired"][0]["only_left_correct"], 1)
        self.assertFalse(comparison["runs"][1]["complete"])


if __name__ == "__main__":
    unittest.main()
