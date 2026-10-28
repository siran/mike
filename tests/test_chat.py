import json
import os
import tempfile
import time
import unittest
from pathlib import Path

from mike.body import Body, has_command, parse_reply
from mike.chat import EXAMPLES, Session, Transcript, converse, read_script
from mike.models import ScriptedModel


class ParseTests(unittest.TestCase):
    def test_speech_then_command_keeps_reply_up_to_the_command(self):
        said, command, kept = parse_reply('I\'ll use my calculator.\n$ python3 -c "print(2+1)"\noutput:\n3')
        self.assertEqual(said, ["I'll use my calculator."])
        self.assertEqual(command, 'python3 -c "print(2+1)"')
        self.assertEqual(kept, 'I\'ll use my calculator.\n$ python3 -c "print(2+1)"')

    def test_lines_that_are_not_commands(self):
        for text in ("OK", "$python3", "$ ", "the price is $ 5? no", "  "):
            self.assertIsNone(parse_reply(text)[1], text)
        self.assertEqual(parse_reply("   $ ls -la  ")[1], "ls -la")

    def test_first_command_wins(self):
        self.assertEqual(parse_reply("$ ls\n$ rm -r x")[1], "ls")

    def test_heredoc_runs_to_its_terminator(self):
        text = "Writing it.\n$ cat > ~/work/a.py <<'EOF'\nfor i in range(3):\n    print(i)\nEOF\n$ python3 ~/work/a.py"
        said, command, kept = parse_reply(text)
        self.assertEqual(command, "cat > ~/work/a.py <<'EOF'\nfor i in range(3):\n    print(i)\nEOF")
        self.assertTrue(kept.endswith("EOF"))
        self.assertEqual(parse_reply('$ cat <<< "hi"\nhi')[1], 'cat <<< "hi"')

    def test_open_quote_continues_like_bash(self):
        text = '$ python3 -c "\nfor i in range(2):\n    print(i)\n"\nDone.'
        said, command, kept = parse_reply(text)
        self.assertEqual(command, 'python3 -c "\nfor i in range(2):\n    print(i)\n"')
        self.assertEqual(kept, text[:-len("\nDone.")])
        self.assertEqual(parse_reply("$ echo \"it's\" # it's fine\nok")[1], "echo \"it's\" # it's fine")

    def test_early_stop_waits_for_a_whole_command(self):
        self.assertFalse(has_command("$ ls"))
        self.assertTrue(has_command("$ ls\n"))
        self.assertFalse(has_command("Hello\n"))
        self.assertFalse(has_command("$ cat > f <<EOF\nline\n"))
        self.assertTrue(has_command("$ cat > f <<EOF\nline\nEOF\n"))
        self.assertFalse(has_command('$ echo "one\n'))
        self.assertTrue(has_command('$ echo "one\ntwo" >> f\n'))


@unittest.skipUnless(os.name == "posix", "the body runs bash")
class BodyTests(unittest.TestCase):
    def setUp(self):
        self.home = tempfile.mkdtemp()
        self.body = Body(self.home, timeout=2.0, max_output=100, grace=0.3)

    def test_runs_in_home_with_clean_environment(self):
        result = self.body.run('pwd; echo "$HOME"; echo "${PYTHONPATH:-none}"; cat; echo $?')
        self.assertEqual(result["output"].split(), [self.home, self.home, "none", "0"])
        self.assertEqual(result["exit_code"], 0)

    def test_exit_status_and_stderr(self):
        result = self.body.run("echo oops >&2; exit 3")
        self.assertEqual((result["output"], result["exit_code"]), ("oops\n", 3))
        self.assertTrue(self.body.observation(result).endswith("(exit status 3)"))

    def test_timeout_kills_the_process_group(self):
        start = time.monotonic()
        result = self.body.run("sleep 30 & sleep 30; echo never")
        self.assertLess(time.monotonic() - start, 5)
        self.assertTrue(result["timed_out"])
        self.assertIsNone(result["exit_code"])
        self.assertIn("(stopped after 2 seconds)", self.body.observation(result))

    def test_output_is_capped(self):
        result = self.body.run("yes x | head -c 100000")
        self.assertTrue(result["truncated"])
        self.assertEqual(len(result["output"]), 100)
        self.assertEqual(result["output_bytes"], 100000)
        self.assertIn("[cut: only the first 100 characters are shown]", self.body.observation(result))

    def test_background_child_does_not_hold_the_turn(self):
        start = time.monotonic()
        result = self.body.run("sleep 20 & echo started")
        self.assertLess(time.monotonic() - start, 1.5)
        self.assertEqual((result["output"], result["exit_code"], result["timed_out"]), ("started\n", 0, False))

    def test_empty_output(self):
        self.assertEqual(self.body.observation(self.body.run("true")), "output:\n(no output)")

    def test_heredoc_writes_a_file(self):
        command = parse_reply("$ cat > a.py <<'EOF'\nprint(6*7)\nEOF")[1]
        self.body.run(command)
        self.assertEqual(self.body.run("python3 a.py")["output"], "42\n")


@unittest.skipUnless(os.name == "posix", "the body runs bash")
class SessionTests(unittest.TestCase):
    def setUp(self):
        self.home = tempfile.mkdtemp()
        self.body = Body(self.home, timeout=2.0, max_output=200)
        self.transcript = Transcript(Path(self.home) / "conversations")

    def session(self, replies, **kwargs):
        model = ScriptedModel(replies)
        return model, Session(model, self.body, self.transcript, system="SYSTEM", **kwargs)

    def test_command_output_returns_to_the_model_then_it_answers(self):
        model, session = self.session(["OK"] + ['Calculator.\n$ python3 -c "print(2+1)"\n3', "It is 3."])
        session.turn("remember 2")
        report = session.turn("add 1?")
        self.assertEqual(report["commands"], ['python3 -c "print(2+1)"'])
        self.assertEqual(report["said"], ["Calculator.", "It is 3."])
        last = model.seen[-1]
        self.assertEqual(last[0], {"role": "system", "content": "SYSTEM"})
        self.assertEqual(last[1:1 + sum(map(len, EXAMPLES))], [m for e in EXAMPLES for m in e])
        self.assertEqual(last[-2:], [{"role": "assistant", "content": 'Calculator.\n$ python3 -c "print(2+1)"'},
                                     {"role": "user", "content": "output:\n3"}])

    def test_step_limit(self):
        _, session = self.session(["$ echo hi"] * 4, max_steps=3)
        report = session.turn("loop")
        self.assertEqual(len(report["commands"]), 3)
        self.assertEqual(len(report["model_calls"]), 4)
        self.assertIn("(not run: the limit is 3 commands per turn)", session.history[-1]["content"])
        self.assertIn("(that was your last command for this turn)", session.history[-3]["content"])

    def test_transcripts_record_everything(self):
        _, session = self.session(["$ echo hi", "done"])
        session.turn("go")
        session.close()
        events = [json.loads(line) for line in self.transcript.raw.read_text().splitlines()]
        self.assertEqual([e["type"] for e in events],
                         ["session", "user", "model", "command", "model", "turn_end", "end"])
        self.assertEqual(events[3]["output"], "hi\n")
        self.assertEqual(self.transcript.text.read_text().splitlines()[1:],
                         ["user: go", "mike: $ echo hi", "output:", "hi", "mike: done"])

    def test_sliding_window_drops_whole_old_turns_and_keeps_the_opening(self):
        _, session = self.session(["a" * 400, "b" * 400, "c"], context_tokens=0)
        for text in ("one", "two", "three"):
            session.turn(text)
        self.assertEqual(session.window()[len(session.opening)]["content"], "three")
        self.assertEqual(session.window()[:len(session.opening)], session.opening)


@unittest.skipUnless(os.name == "posix", "the body runs bash")
class ScriptTests(unittest.TestCase):
    def test_script_checks_and_sessions(self):
        items = read_script("# c\nhi\nexpect: hello | hey\n/new\nadd?\nexpect-command: python3\n"
                            "expect-start: yes\nif-asked: details\nmax-new-tokens: 60\nwhat?\n")
        self.assertEqual([i[0] for i in items], ["say", "expect", "new", "say", "expect-command",
                                                  "expect-start", "if-asked", "max-new-tokens", "say"])
        body = Body(tempfile.mkdtemp(), timeout=2.0)
        model = ScriptedModel(["Hey!", "Yes, a sum.\n$ python3 -c 'print(1)'", "1", "Which one?", "fine"])
        sessions = []

        def new():
            sessions.append(Session(model, body, None, system="S"))
            return sessions[-1]
        result = converse(new, items, out=lambda *_: None)
        self.assertEqual(len(sessions), 2)
        self.assertEqual([c["passed"] for c in result["checks"]], [True, True, True])
        self.assertEqual(len(result["turns"]), 3)
        self.assertEqual(result["turns"][2]["user"], "what?")


if __name__ == "__main__":
    unittest.main()
