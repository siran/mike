# Protocol: MIK-e-1 plain-text retrieval

This protocol was written after development on a separate world (seed 31415; see
[dev-log-mike1.md](dev-log-mike1.md)) and before the held-out world below was generated. Nothing in it was chosen
by looking at held-out data. It inherits everything from [protocol.md](protocol.md) (MIK-0) and
[evaluation.md](evaluation.md) that it does not change; the changes are listed explicitly.

FROZEN: STATUS-PLACEHOLDER

## What is tested

The task family is MIK-0's, unchanged: a question about a randomly generated object, answerable only by following
four typed facts (object → box → station → region → property). MIK-0 failed before any reasoning was tested,
because neither model ever issued a search under a JSON interface. MIK-e-1 asks two narrow questions:

1. With a plain-text interface, do the same frozen models look facts up and answer with correct, cited evidence?
2. Given the four needed facts directly in context (with distractors), can they join them without any lookup?

A pass or fail applies to these two models, this generator and this interface only.

## Models and decoding (as MIK-0 unless stated)

* `HuggingFaceTB/SmolLM2-135M-Instruct` at revision `12fd25f77366fa6b3b4b768ec3050bf629380bac` and
  `HuggingFaceTB/SmolLM2-360M-Instruct` at `a10cc1512eabd3dde888204e902eca88bddb4951` (MIK-0's resolved snapshots),
  frozen, float32, CPU, greedy decoding, 4 torch threads, one run at a time.
* At most **6 model decisions** per task, including the final answer (as MIK-0).
* At most **128 new tokens** per decision, for every interface (MIK-0: 96). Reason: a fully cited answer with these
  identifiers is 97 tokens as compact JSON and 83–87 tokens as a text command, so MIK-0's cap could not hold a
  supported JSON answer. Using one cap for all interfaces keeps the interface the only difference between them.
* Text interfaces stop decoding once a newline-terminated generated line parses as a command. Because the parser
  takes the first valid line and greedy decoding is deterministic, this cannot change the parsed action. The JSON
  interface has no early stop.

## World

`mike generate --seed 20261101 --count 100 --schema 2 --out data/heldout-20261101.json`, generated only after this
protocol is frozen: 100 four-fact chains with 48-bit identifiers, 2,000 unrelated facts, shuffled records. Its
canonical SHA-256 is recorded in the report. Every text run refuses to start if any worked-example identifier
occurs in the world.

**One change to MIK-0's generator:** world schema 2 names record IDs `F` + 16 hex digits instead of `R` + 16.
In MIK-0's worlds a record ID and a region entity (`R` + 12 hex) look alike, and during development both models
cited region IDs as if they were records (360M: 10 of its 21 correct but unsupported answers on the 40-task
development world). Since evidence-supported accuracy is the gate, this artefact was removed before any held-out
world existed. The random draws are unchanged, so a schema-2 world has the same entities, questions and answers as
the schema-1 world of the same seed; schema 1 (`--schema 1`) still regenerates MIK-0's worlds exactly. All MIK-e-1
held-out runs, including the JSON reference, use the same schema-2 world; MIK-0's reported results used schema 1.

## The frozen interface: `text-example`

Defined by `src/mike/text.py` with `TYPE_NOUNS = False` and `REMIND_QUESTION = False`, and no retry.

**Opening (tools and no-tools conditions).** A system message, then one worked example as prior chat turns, then
the task. The example uses made-up identifiers from `random.Random("mike1-example-ids-not-from-any-world")`:

```text
[system]
You answer questions about a database of facts. You do not know these facts; you have to look them up.

To look up an ID, reply with one line:
search <ID>
I will reply with the facts about that ID. Each fact starts with its record ID.

Look up one ID at a time. Follow the IDs that the facts give you until you reach what the question asks for.

When you know the answer, reply with one line:
answer <ID> because <record IDs>
List the record IDs of the facts you used. If you cannot find the answer, reply: answer unknown
[user]       What property characterizes the region containing the station associated with object OC466B41C0A7A?
[assistant]  search OC466B41C0A7A
[user]       Facts about OC466B41C0A7A:
             Record F5F0943517C5BF050: OC466B41C0A7A is inside B17EAFA897FE4.
[assistant]  search B17EAFA897FE4
[user]       Facts about B17EAFA897FE4:
             Record FBFFC342D72A3C297: B17EAFA897FE4 belongs to S0A547DC6F7DA.
[assistant]  search S0A547DC6F7DA
[user]       Facts about S0A547DC6F7DA:
             Record F33983177FAC87BB6: S0A547DC6F7DA is located in R11ABC5B76157.
[assistant]  search R11ABC5B76157
[user]       Facts about R11ABC5B76157:
             Record F02513CB5017CC1E0: R11ABC5B76157 has PB716400BF4D4.
[assistant]  answer PB716400BF4D4 because F5F0943517C5BF050, FBFFC342D72A3C297, F33983177FAC87BB6, F02513CB5017CC1E0
```

(The indentation above is for reading; each message's exact text is in the code and in every run's metadata,
with a SHA-256 of the whole opening.) The task's question then follows as a user message, verbatim.

**Commands.** The parser reads the output line by line and takes the first line that matches; the rest is
recorded but not used. After stripping surrounding whitespace and at most one trailing period, a line is:

* `search <token>`, or
* `answer <token>`, optionally followed by `because` and one or more tokens separated by commas and/or spaces
  (the word `and` between them is a separator). `answer unknown` is an abstention.

Command words are case-insensitive and may be followed by a colon. Every other token is kept exactly as written:
a word in the citation list is a citation, a trailing comma stays part of the value, `search for X` is not a
command. No line matching means an invalid action, which ends the task. There is no retry.

**Observations** come back as the next user message, one sentence per record, carrying its record ID:
`Facts about X:` followed by lines `Record <record id>: <subject> <phrase> <object>.` with the phrases
`is inside`, `belongs to`, `is located in`, `has`, `is related to`. A search with no records gets
`There are no facts about X.`; with search disabled, `Search is disabled. No facts are available.`

**History** is real chat turns: the matched command line (verbatim, stripped) as the assistant message and the
observation as the next user message. Any other text the model wrote in that output is recorded in the trace
but not shown back to it.

## Conditions

All three use the same tasks and decoding.

* **tools** — search enabled; opening as above.
* **no-tools** — the same opening; every search returns the disabled sentence. A sanity check: the facts are
  random, so correct answers are not expected.
* **evidence** — no search (disabled as above). The first user message is `Facts:` followed by eight record
  sentences, a blank line and `Question: <question>`. The eight are the task's four records and one record of
  each chain relation (inside, belongs_to, located_in, has_property) taken from four other tasks, shuffled with
  `random.Random("mike1-evidence:<world seed>:<task id>")`, so each relation appears twice and exactly one chain
  starts at the question's object. The system message and worked example are the evidence variants:

```text
[system]
You answer questions about a database of facts. You do not know these facts yourself. The facts you need are
listed before the question, mixed with some facts you do not need. Each fact starts with its record ID.

Follow the facts from the ID in the question until you reach what the question asks for.

Reply with one line:
answer <ID> because <record IDs>
List the record IDs of the facts you used. If the facts do not give the answer, reply: answer unknown
[user]       Facts: (the example's four records and four made-up distractors, one per relation, in a fixed order)
             Question: (the example question)
[assistant]  answer PB716400BF4D4 because F5F0943517C5BF050, FBFFC342D72A3C297, F33983177FAC87BB6, F02513CB5017CC1E0
```

(The system message has no line breaks inside its first paragraph; see `text.SYSTEM_EVIDENCE`.)

## Reference interfaces (tools condition only)

* **json** — MIK-0's interface exactly (prompt `models.SYSTEM`, the whole state as one JSON user message,
  strict JSON parsing), but with the 128-token cap above.
* **text** — the frozen system message with no worked example (zero-shot).

## Runs, in this order

| # | model | interface | condition |
| --- | --- | --- | --- |
| 0 | — | handwritten traversal; evidence controls | tools, no-tools, evidence |
| 1–2 | 135M, 360M | text-example | tools |
| 3–4 | 135M, 360M | text-example | evidence |
| 5–6 | 135M, 360M | text-example | no-tools |
| 7–8 | 135M, 360M | json | tools |
| 9–10 | 135M, 360M | text | tools |

Each run evaluates all 100 tasks and writes `traces/heldout-mike1/<model>-<interface>-<condition>.jsonl`.
If a run is interrupted, its partial file is kept and renamed `…-interrupted.jsonl` and the run is repeated from the
first task; both are reported. Compute estimate from development rates: ESTIMATE-PLACEHOLDER

## Scoring

As [evaluation.md](evaluation.md): plain accuracy (exact match of the property identifier); evidence-supported
accuracy (correct, all four expected record IDs cited, every citation observed, extra observed citations allowed);
abstention; invalid action; budget exhaustion. In the evidence condition the eight supplied facts count as
observed. Added for MIK-e-1: **exact evidence** (cited set equals the four expected IDs), because in the evidence
condition citing all eight supplied facts would otherwise count as complete. Costs are reported per run: input and
output tokens, decisions, searches, wall and process CPU time; model loading and world generation excluded.

Failure analysis uses `mike.analysis`, fixed before the held-out run: what the answered value is (which hop of the
chain, a chain record ID, a distractor, a worked-example identifier, other); how far the chain was searched in
order; the citation profile (exact, complete with extra observed, cites example IDs, cites unobserved, incomplete,
none); and the first action.

## Criteria

1. **Feasibility gate (MIK-0's, unchanged):** for each model, `text-example` in the tools condition with at least
   **50/100 correct answers supported by all four observed record IDs**. Reported per model as pass or fail.
2. **Evidence condition:** the same gate is reported. Separately, plain accuracy is compared with the no-join control
   (the first property listed) on the same tasks: the models are said to join supplied facts better than the
   control only if the lower Wilson 95% bound of their plain accuracy exceeds the control's rate. Because two
   properties are always listed, picking one at random is right about half the time.
3. **No-tools:** expected 0/100 correct; any correct answer is investigated and reported.
4. **Interfaces:** paired per-task outcomes (correct and supported) for `text-example` against `json` and `text`
   per model, from `mike compare`. These are descriptive counts on one world, not tests of a general effect.

Wilson 95% intervals are reported for plain and supported accuracy; as in MIK-0 they are marginal summaries over
tasks from one generator and one world.

## What will not be concluded

No result says anything about general reasoning, minimal capacity, or model size as a cause (two models from one
family differ in more than size). A pass would show that these frozen models can operate this lookup through this
interface on this synthetic chain task, not that they reason about the chain: each search returns only the next
fact, so following the chain needs no comparison among facts. The worked example demonstrates the exact four-step
procedure of this task family with different identifiers; success with it is success at following a demonstrated
procedure on new facts, not discovering one. The interface was developed on one 40-task world, with small and
noisy differences between variants; it may be overfitted to that world. Persistent notes, contradiction,
discovery, calibration and multiple held-out worlds remain untested.
