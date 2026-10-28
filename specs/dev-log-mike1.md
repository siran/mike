# MIK-e-1 development log

Development used only the development world below. No held-out world existed while the interface was changed.
Trace files are under `traces/dev-mike1/` (local, not committed); every run listed here is kept there, including
failed and superseded ones. Counts below are on development tasks and are for choosing an interface, not results.

## Setup

* Development world: `mike generate --seed 31415 --count 40 --out data/dev-31415.json`,
  canonical SHA-256 `4e2f62e5d94365cfa90b87707c4222759764b99f836f45f7093d690ae287fe49`.
* Controls on it: handwritten traversal 40/40 with search, 0/40 without. Evidence-supplied controls: a four-step join
  over the supplied facts 40/40; picking the first property listed (no join) 20/40, i.e. chance by design.
* Models and decoding as MIK-0: SmolLM2-135M-Instruct `12fd25f…` and SmolLM2-360M-Instruct `a10cc15…`, frozen,
  float32, greedy, CPU, 4 torch threads, one run at a time, at most 6 decisions per task.

## Budget finding: MIK-0's 96-token cap could not hold a fully cited answer

With these identifiers, a correct answer citing all four records is 97 tokens as compact JSON and 104 tokens as
`json.dumps` default JSON (SmolLM2 tokenizer), so MIK-0's cap of 96 new tokens could never have produced an
evidence-supported JSON answer. It did not bind in MIK-0 because no model got that far (the longest outputs were
about 44 tokens). The text form is 83–87 tokens. MIK-e-1 therefore uses **128 new tokens per decision for every
interface, including the JSON reference**, so that the interface is the only difference between them.

The text interfaces also stop decoding early once a newline-terminated generated line parses as a command. Since
greedy decoding is deterministic and the parser takes the first valid line, this cannot change which action is
parsed; it only shortens the recorded raw output and saves compute. The JSON interface has no early stop (a JSON
action must be the whole output, so stopping early could change its validity).

## Getting a first search

* **d01** — `text` (plain-English system prompt, question as the user turn), 360M, 3 tasks: 0 searches, 3/3 invalid.
  The model answered from world knowledge in prose: "… is its altitude", "… is the "latitude" property".
* **d02** — `text-example` with the worked example written **inside the system prompt**, 360M, 3 tasks: 0 searches.
  Twice the model continued the system prompt verbatim, ending on the template line `answer <ID> because <record IDs>`,
  which the grammar accepts: it was scored as an (incorrect) answer `<ID>`, not repaired. Once it narrated an
  invented trajectory in prose.
* **p01** (first decision only, 5 tasks, 360M; `traces/dev-mike1/probes/p01-360-v1.json`) — counts of
  first decisions that were `search <the question's object>`:

  | variant | 360M |
  | --- | ---: |
  | A `text` | 0/5 |
  | B example inside system prompt | 0/5 |
  | C shorter system prompt, no placeholders | 0/5 |
  | D C + example inside system prompt | 3/5 (prose line, then a valid command line) |
  | E example as prior chat turns | 5/5 |
  | F D + user turn "Question: …\nYour reply:" mirroring the example | 5/5 |

* **p02** (same, 135M; `probes/p02-135-v2.json`): B 0/5, **E 5/5**, F 0/5 (it copied the example's
  "My reply:" label and invented an observation), and three zero-shot variants with the cue or an explicit
  "Start by searching the ID in the question" 0/5 each.

**Decision:** the worked example is given as **prior chat turns** (user question → assistant `search …` →
user observation → … → assistant `answer … because …`), after the system prompt. This departs from the brief's
wording ("add one worked example to the system prompt"): with the example inside the system prompt, 135M searched
in 0/5 (B) and 0/5 (F) first decisions; 360M in 0/5 (B), 3/5 after shortening the system prompt (D), and 5/5 only
when the user turn was also reshaped to mirror the example (F). Prior chat turns are the form a chat model's own
template is built for, and the same example in that form gave 5/5 for each model. The zero-shot `text` interface
never produced a first search in any wording tried: 0/25 first decisions (360M: A, C; 135M: G, H, I).

## Full trajectories: where they fail

* **d03** — `text-example`, tools, 10 tasks: 360M 5/10 correct, 1/10 evidence-supported; 135M 2/10, 0/10.
  All 20 first decisions searched the right object; there were no invalid actions. Failure analysis
  (`mike.analysis`): 135M answered the **station** in 8/10 (two searches, then `answer S…`), and never wrote a
  `because` clause. 360M answered the station in 4/10 and the box once; its citations duplicated or dropped records
  and in 3/10 included the worked example's record IDs (copied from the example's answer line).
* **d04** — same, observations name each entity's type from its identifier prefix ("object O… is inside box B…").
  135M 5/10 correct (0 supported); 360M **0/10**: it now answered with the record ID of the station fact after two
  searches in 7/10. The brief's own sample observation names types ("object OB23… is associated with station S9F…"),
  and the type is already encoded in each generated identifier (O/B/S/R/P), so this is labelling, not traversal.
* **Sensitivity.** d04 also changed the `has_property` phrase from "has property" to "has" (so that typed output
  reads "region R… has property P…", not "has property property P…"); untyped output then reads "R… has P…". In the
  20-task screen (d05) that one word, which first appears in the worked example's last observation, moved 135M on
  the first 10 tasks from 2/10 (d03) to 5/10 correct, and 360M from 5/10 to 6/10. For 135M the change acted at the
  decision after the second search, before the task's own fourth observation existed, so it came from the example
  alone. Small wording changes swing these models by several tasks in ten; 10-task differences are not evidence.
  The d05 runs with this wording are named `…-n0-r0-hasP`.

## Screen of two rendering switches (d05, 20 tasks, `text-example`, tools)

| observation rendering | 135M correct (supported) | 360M correct (supported) |
| --- | ---: | ---: |
| untyped, "R… has P…" (`n0-r0-hasP`) | 8/20 (0) | **12/20 (2)** |
| types named, "object O… is inside box B…" (`n1-r0`) | 7/20 (0) | 0/20 (0) |
| untyped + the question restated after every observation (`n0-r1`) | 0/20 (0) | 0/20 (0) |
| types + restated question (`n1-r1`) | stopped after 1 task (`-interrupted`) | stopped after 0 tasks (`-interrupted`) |

* Naming types made 360M stop after the second search in 15/20 tasks, now usually answering the record ID of the
  station fact: the observation literally said "station S…", matching the question's "the station associated with
  object…".
* Restating the question after each observation made both models answer immediately: 360M answered the box after
  one search in 20/20; 135M answered the box in 12/20 and repeated the same search until the budget ran out in 5/20.
  A question at the end of a user turn reads to these models as "answer now".
* The fourth arm was stopped deliberately once the other three showed that neither switch could help both models;
  its partial files are kept.

**Decision:** both switches off (`text.TYPE_NOUNS = False`, `text.REMIND_QUESTION = False`), "has" wording. The frozen
`text-example` opening is byte-identical to the one recorded in `d05-360-…-n0-r0-hasP`.

What remains after this screen is not the interface: every first decision searched the right object, invalid
actions were 0/80 in the untyped arms, and in every run of d03–d05 except the restated-question arm, every search
was the next subject of the chain (0 repeated or off-chain searches; with the restated question, 135M repeated a
search 12 times and left the chain 3 times).
The losses are (a) answering after the second search, the station, which the question names ("the station
associated with object…"), and (b) citations: 135M almost never writes a `because` clause; 360M copies record IDs
accurately but drops one, repeats one, cites a **region** ID (regions are `R` + 12 hex, records `R` + 16 hex, a
MIK-0 generator choice), or copies the worked example's record IDs.

## Evidence supplied, retry, and no-tools probes (d06)

* **Evidence supplied, 20 tasks.** Eight facts in the first user turn (the task's four plus one fact of each chain
  relation from four other tasks, shuffled), no search.
  * `text-example` (the example given as one prior user turn of facts and question, then the answer line):
    135M 4/20 correct, 0 supported; 360M 4/20 correct, 0 supported. The no-join control (first property listed)
    gets 10/20 on these tasks. Both models picked between the two listed properties without regard to the
    chain (135M: first 4, second 5; 360M: first 5, second 9; no position bias) or answered a non-property entity.
    135M copied the example's record IDs in 10/20 citations and often filled the 128-token cap with repeated IDs.
  * `text` (zero-shot): 135M 20/20 invalid; 360M 17/20 invalid, 3 answered, 0 correct. The invalid outputs were
    prose ("… is PE76A869B1041."). Read as a diagnostic only (nothing is scored or repaired from prose), the prose
    named the correct property as often as the distractor property (135M 5 vs 6, 360M 4 vs 4): the format failure
    is not hiding above-chance joining.
  * Reading: in the tools condition each search returns only the next fact, so the subject-indexed lookup turns
    the four-fact join into four steps of copying the newest identifier. With the facts side by side and no lookup,
    neither model selects the right chain. No change to the evidence interface was made in response.
* **Retry variant (`text --retry`, tools, 10 tasks).** After the error sentence, 135M still never searched
  (10/10 invalid, 20 decisions). 360M searched in 6/10 (`Search: O…`, accepted by the declared colon form) but then
  wrote prose (`Answer: O… is inside B….`, invalid) or repeated the system prompt's template line (scored as the
  answer `<ID>`): 0 correct. With `text-example` invalid outputs were already rare (0/80 untyped), so a retry would
  almost never trigger. **Decision:** no retry in the frozen protocol.
* **No-tools (`text-example`, 10 tasks).** 135M kept searching until the budget ran out (9/10; 55 searches, about
  15 s per task). 360M searched once, got "Search is disabled. No facts are available.", then apologised in prose
  (10/10 invalid, about 16 s per task). Zero correct, as expected.

## All 40 development tasks, frozen candidate (d07)

Interface as decided above (`text-example`, both switches off, no retry), 128 new tokens, 6 decisions. The first
20 tasks of both `text-example` runs were byte-identical to the d05 `…-hasP` runs (deterministic decoding,
same opening).

| model | interface | condition | correct | evidence-supported | invalid | searches | decisions | wall s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 135M | text-example | tools | 17/40 | 0/40 | 0 | 114 | 154 | 469 |
| 360M | text-example | tools | 24/40 | 3/40 | 0 | 137 | 177 | 1,404 |
| 135M | json (MIK-0) | tools | 0/40 | 0/40 | 40 | 0 | 40 | 89 |
| 360M | json (MIK-0) | tools | 0/40 | 0/40 | 28 | 0 | 40 | 203 |
| 135M | text (zero-shot) | tools | 0/40 | 0/40 | 40 | 0 | 40 | 153 |
| 360M | text (zero-shot) | tools | 0/40 | 0/40 | 40 | 0 | 40 | 160 |

* 135M: answered the station after two searches in 23/40 and the property after four in 17/40, nothing else;
  32/40 answers had no `because` clause.
* 360M: answered the property in 24/40, the region in 9/40, the station in 7/40. Of 141 cited tokens, 107 were
  expected records, **10 were region identifiers cited as records**, 11 other entity IDs, 11 worked-example IDs.
* JSON at 128 tokens reproduced MIK-0's failure on new tasks: 0 searches; the 360M model answered the question's
  own object in 11/40.
* The `text-example` evidence run on 40 tasks was stopped after its first task (`…-interrupted`) when the record
  prefix amendment below was made; the 20-task d06 runs stand for the old prefix.

## Amendment before the held-out world: record IDs get their own prefix

MIK-0's generator names record IDs `R` + 16 hex and region entities `R` + 12 hex. In the observations a region
looks like a record ID, and the models cite regions as records. Example (d07 360M, task-0000, correct answer):
`answer PE76A869B1041 because R5FFD023DBABBA7C7, RCE758941A85DF819, R5FFD023DBABBA7C7, R8BB90AE6AAD6`, a repeated
record plus the region `R8BB90AE6AAD6`. In d07, 10 of 360M's 21 correct-but-unsupported answers cited a region ID
(135M: 7 of 17). Evidence-supported accuracy is the gate, so this generator artefact could decide the result.

**Change:** `world.generate` gained a `schema` argument. Schema 2 (now the default) names records `F` + 16 hex;
no entity type uses `F` (entities use O, B, S, R, P, D, N). Nothing else changes: the random draws, entities,
questions and answers are identical, only the record prefix differs (tested). Schema 1 still regenerates the
MIK-0 worlds exactly (checked: seeds 20261029, 10728 and the old dev world reproduce their canonical hashes). The
worked example's record IDs were changed to `F` the same way. `mike.analysis` now reads worked-example identifiers
from each run's recorded opening, so old and new traces are both classified correctly.

The schema-2 development world is `data/dev-31415-s2.json` (same seed and tasks, SHA-256
`6fb4cf8117124aedf2f1bd2985d12f39cf8e368fd7a22a0599fff22dbf3a922c`); controls unchanged (traversal 40/40,
evidence join 40/40, first property listed 20/40). Because only the prefix differs, the re-check below is paired
task by task with d07.
