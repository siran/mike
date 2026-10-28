import unittest

from mike import text
from mike.evaluate import evaluate_one, evidence_baselines, summarize
from mike.models import ScriptedModel
from mike.runtime import run_text
from mike.tools import Tools
from mike.world import generate


class ParserTests(unittest.TestCase):
    def test_search_command(self):
        self.assertEqual(text.parse_line("search OB23C4574EA97"), {"action": "search", "entity": "OB23C4574EA97"})
        self.assertEqual(text.parse_line("  Search: OB23C4574EA97.  "), {"action": "search", "entity": "OB23C4574EA97"})

    def test_answer_with_citations(self):
        action = text.parse_line("answer P1 because R1, R2 R3 and R4.")
        self.assertEqual(action, {"action": "answer", "value": "P1", "evidence": ["R1", "R2", "R3", "R4"]})
        self.assertEqual(text.parse_line("ANSWER P1")["evidence"], [])

    def test_unknown_abstains(self):
        self.assertEqual(text.parse_line("answer unknown"), {"action": "answer", "value": None, "evidence": []})

    def test_tokens_are_kept_as_written(self):
        # No repair: words in the citation list stay as (unobserved) citations, and extra
        # words make a search line invalid rather than being dropped.
        self.assertEqual(text.parse_line("answer P1 because record R1")["evidence"], ["record", "R1"])
        self.assertEqual(text.parse_line("answer P1, because R1")["value"], "P1,")
        self.assertIsNone(text.parse_line("search for OB23C4574EA97"))
        self.assertIsNone(text.parse_line("I will search OB23C4574EA97"))
        self.assertIsNone(text.parse_line('{"action":"search","entity":"OB23C4574EA97"}'))
        self.assertIsNone(text.parse_line("answer P1 because"))
        self.assertIsNone(text.parse_line("search"))

    def test_first_valid_line_wins(self):
        raw = "Let me look this up.\nsearch O1\nanswer P1 because R1"
        self.assertEqual(text.parse(raw), ({"action": "search", "entity": "O1"}, "search O1"))
        self.assertEqual(text.parse("no command here\nnor here"), (None, None))

    def test_early_stop_only_after_a_terminated_valid_line(self):
        self.assertFalse(text.has_complete_command("search O1"))
        self.assertFalse(text.has_complete_command("Thinking.\nsea"))
        self.assertTrue(text.has_complete_command("search O1\n"))
        self.assertTrue(text.has_complete_command("Thinking.\nsearch O1\nRecord"))


class RenderingTests(unittest.TestCase):
    record = {"id": "R0000000000000001", "subject": "OAAAAAAAAAAAA", "relation": "inside", "object": "BBBBBBBBBBBBB"}

    prop = {"id": "R0000000000000001", "subject": "R123456789ABC", "relation": "has_property", "object": "P123456789ABC"}

    def setUp(self):
        self.switches = (text.TYPE_NOUNS, text.REMIND_QUESTION)

    def tearDown(self):
        text.TYPE_NOUNS, text.REMIND_QUESTION = self.switches

    def test_frozen_rendering(self):
        self.assertEqual((text.TYPE_NOUNS, text.REMIND_QUESTION), (False, False))
        self.assertEqual(text.render_record(self.record),
                         "Record R0000000000000001: OAAAAAAAAAAAA is inside BBBBBBBBBBBBB.")
        self.assertEqual(text.render_record(self.prop), "Record R0000000000000001: R123456789ABC has P123456789ABC.")
        self.assertEqual(text.observation_message("X", {"records": []}, "Q?"), "There are no facts about X.")

    def test_development_switches(self):
        text.TYPE_NOUNS = True
        self.assertEqual(text.render_record(self.record),
                         "Record R0000000000000001: object OAAAAAAAAAAAA is inside box BBBBBBBBBBBBB.")
        self.assertEqual(text.render_record(self.prop),
                         "Record R0000000000000001: region R123456789ABC has property P123456789ABC.")
        # Unknown prefixes and non-entity lengths (record IDs are 17 characters) get no type noun.
        unknown = dict(self.record, subject="D123456789ABC", relation="related_to", object="R0000000000000002")
        self.assertEqual(text.render_record(unknown),
                         "Record R0000000000000001: D123456789ABC is related to R0000000000000002.")
        text.REMIND_QUESTION = True
        self.assertEqual(text.observation_message("X", {"records": []}, "Q?"), "There are no facts about X.\n\nQuestion: Q?")

    def test_observations(self):
        self.assertEqual(text.render_observation("OAAAAAAAAAAAA", {"records": [self.record]}),
                         "Facts about OAAAAAAAAAAAA:\n"
                         "Record R0000000000000001: OAAAAAAAAAAAA is inside BBBBBBBBBBBBB.")
        self.assertEqual(text.render_observation("X", {"records": []}), "There are no facts about X.")
        self.assertEqual(text.render_observation("X", {"error": "tools_disabled"}),
                         "Search is disabled. No facts are available.")
        self.assertEqual(text.render_observation("X", {"error": "entity_must_be_a_string"}),
                         "Search failed: entity must be a string.")

    def test_prompts(self):
        self.assertEqual(text.opening("text", "tools"), [{"role": "system", "content": text.SYSTEM_TOOLS}])
        self.assertNotIn("search <ID>", text.system_prompt("text", "evidence"))
        example = text.opening("text-example", "tools")
        self.assertEqual([m["role"] for m in example],
                         ["system", "user"] + ["assistant", "user"] * 4 + ["assistant"])
        self.assertEqual(example[2]["content"], "search OC466B41C0A7A")
        self.assertTrue(example[-1]["content"].startswith("answer PB716400BF4D4 because F5F0943517C5BF050, "))
        for message in example[1:]:
            self.assertEqual(text.parse(message["content"])[0] is not None, message["role"] == "assistant")
        evidence = text.opening("text-example", "evidence")
        self.assertEqual([m["role"] for m in evidence], ["system", "user", "assistant"])
        self.assertTrue(evidence[1]["content"].startswith("Facts:\nRecord "))
        with self.assertRaises(ValueError):
            text.opening("json", "tools")

    def test_example_ids_are_not_in_generated_worlds(self):
        for seed in (42, 10728, 31415):
            self.assertTrue(text.examples_disjoint(generate(seed, 5)))
        poisoned = {"records": [{"id": "F5F0943517C5BF050", "subject": "x", "relation": "inside", "object": "y"}]}
        self.assertFalse(text.examples_disjoint(poisoned))


class TextRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.world = generate(42, 6)
        self.task = self.world["tasks"][0]
        by_id = {r["id"]: r for r in self.world["records"]}
        self.chain = [by_id[i] for i in self.task["evidence"]]
        self.opening = text.opening("text", "tools")

    def oracle_lines(self):
        lines = [f"search {r['subject']}" for r in self.chain]
        return lines + [f"answer {self.task['answer']} because " + ", ".join(self.task["evidence"])]

    def test_oracle_scores_and_history_is_chat_turns(self):
        model = ScriptedModel(self.oracle_lines())
        trace = evaluate_one(model, self.task, self.world["records"], True, 6, interface="text", opening=self.opening)
        self.assertTrue(trace["score"]["correct"] and trace["score"]["complete_evidence"])
        self.assertTrue(trace["score"]["exact_evidence"])
        last = model.seen[-1]
        self.assertEqual([m["role"] for m in last],
                         ["system", "user"] + ["assistant", "user"] * 4)
        self.assertEqual(last[1]["content"], self.task["question"])
        self.assertEqual(last[2]["content"], f"search {self.chain[0]['subject']}")
        self.assertIn(f"Record {self.chain[0]['id']}:", last[3]["content"])

    def test_prose_and_hallucinated_continuation_are_recorded_not_used(self):
        raw = f"I need to look it up.\nsearch {self.chain[0]['subject']}\nRecord RFAKE: made up."
        model = ScriptedModel([raw, "answer unknown"])
        trace = run_text(model, self.task["question"], Tools(self.world["records"]), 6, opening=self.opening)
        self.assertEqual(trace["steps"][0]["generation"]["raw"], raw)
        self.assertEqual(trace["steps"][0]["line"], f"search {self.chain[0]['subject']}")
        self.assertEqual(model.seen[1][2]["content"], f"search {self.chain[0]['subject']}")
        self.assertEqual(trace["result"]["status"], "answered")
        self.assertIsNone(trace["result"]["value"])

    def test_invalid_without_retry_ends_the_task(self):
        trace = run_text(ScriptedModel(["hello"]), "q", Tools([]), 6, opening=self.opening)
        self.assertEqual(trace["result"]["status"], "invalid_action")
        self.assertEqual(trace["retries"], 0)

    def test_one_explicit_retry_is_counted_and_shown(self):
        model = ScriptedModel(["hello", "answer unknown"])
        trace = run_text(model, "q", Tools([]), 6, opening=self.opening, retry=True)
        self.assertEqual(trace["retries"], 1)
        self.assertEqual(trace["steps"][0]["feedback"], text.RETRY_MESSAGE)
        self.assertTrue(trace["steps"][1]["after_retry_message"])
        self.assertEqual(model.seen[1][-2:], [{"role": "assistant", "content": "hello"},
                                              {"role": "user", "content": text.RETRY_MESSAGE}])
        self.assertEqual(trace["result"]["status"], "answered")
        self.assertEqual(summarize([dict(trace, score={"correct": False, "complete_evidence": False,
                                                       "abstained": True})])["retries"], 1)

    def test_retry_is_bounded_and_uses_budget(self):
        trace = run_text(ScriptedModel(["hello", "still no"]), "q", Tools([]), 6, opening=self.opening, retry=True)
        self.assertEqual(trace["result"]["status"], "invalid_action")
        self.assertEqual(len(trace["steps"]), 2)
        trace = run_text(ScriptedModel(["search X", "hello"]), "q", Tools([]), 2, opening=self.opening, retry=True)
        self.assertEqual((trace["retries"], trace["result"]["status"]), (0, "invalid_action"))

    def test_budget_exhausted(self):
        trace = run_text(ScriptedModel(["search X"] * 6), "q", Tools([]), 6, opening=self.opening)
        self.assertEqual(trace["result"]["status"], "budget_exhausted")

    def test_unobserved_citations_are_not_supported(self):
        line = f"answer {self.task['answer']} because " + ", ".join(self.task["evidence"])
        trace = evaluate_one(ScriptedModel([line]), self.task, self.world["records"], True, 6,
                             interface="text", opening=self.opening)
        self.assertTrue(trace["score"]["correct"])
        self.assertFalse(trace["score"]["complete_evidence"])


class EvidenceConditionTests(unittest.TestCase):
    def setUp(self):
        self.world = generate(42, 6)
        self.task = self.world["tasks"][0]

    def test_facts_are_deterministic_and_balanced(self):
        facts = text.supplied_facts(self.world, self.task)
        self.assertEqual(facts, text.supplied_facts(self.world, self.task))
        self.assertEqual(len(facts), 4 + text.DISTRACTORS_PER_TASK)
        ids = {r["id"] for r in facts}
        self.assertTrue(set(self.task["evidence"]) <= ids)
        relations = sorted(r["relation"] for r in facts)
        self.assertEqual(relations, sorted(list(text.CHAIN_RELATIONS) * 2))
        donors = [t for t in self.world["tasks"] if set(t["evidence"]) & ids and t is not self.task]
        self.assertEqual(len(donors), 4)

    def test_supplied_facts_count_as_observed_and_search_is_disabled(self):
        facts = text.supplied_facts(self.world, self.task)
        opening = text.opening("text", "evidence")
        line = f"answer {self.task['answer']} because " + " ".join(self.task["evidence"])
        model = ScriptedModel([line])
        trace = evaluate_one(model, self.task, self.world["records"], False, 6, interface="text",
                             opening=opening, facts=facts)
        self.assertTrue(trace["score"]["correct"] and trace["score"]["complete_evidence"])
        self.assertTrue(model.seen[0][1]["content"].startswith("Facts:\nRecord "))
        self.assertTrue(model.seen[0][1]["content"].endswith("Question: " + self.task["question"]))
        searched = run_text(ScriptedModel(["search " + self.task["start"], "answer unknown"]), "q",
                            Tools(self.world["records"], False), 6, opening=opening, facts=facts)
        self.assertEqual(searched["steps"][0]["observation_text"], "Search is disabled. No facts are available.")

    def test_citing_every_supplied_fact_is_complete_but_not_exact(self):
        facts = text.supplied_facts(self.world, self.task)
        line = f"answer {self.task['answer']} because " + ", ".join(r["id"] for r in facts)
        trace = evaluate_one(ScriptedModel([line]), self.task, self.world["records"], False, 6, interface="text",
                             opening=text.opening("text", "evidence"), facts=facts)
        self.assertTrue(trace["score"]["complete_evidence"])
        self.assertFalse(trace["score"]["exact_evidence"])

    def test_controls(self):
        result = evidence_baselines(self.world)
        self.assertEqual(result["join_over_supplied_facts"], 6)
        self.assertLessEqual(result["first_property_listed"], 6)


if __name__ == "__main__":
    unittest.main()
