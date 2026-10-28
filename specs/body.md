# MIK's body: a research direction

MIK's model need not carry a world of facts in its weights. It could learn how to understand and express questions, use its surroundings, compare what it finds, form and test ideas, and keep notes. These are research goals, not capabilities established by the first experiment. Asking useful questions, choosing actions, comparing evidence, and deciding what to believe are themselves substantive abilities: access to a body does not provide them automatically.

The pretrained models used so far already contain knowledge and learned behaviors. Randomly generated facts make the benchmark's answers new to them, but do not make the models language-only. Training a model specifically for language and inquiry remains a later experiment, after the body and evaluation are useful enough to test it.

## Inspiration from eGPT

The eGPT checkout supplies the structure; the `~/.egpt` profile is the lived-in body. In the actual profile there are rooms, conversations, agents, documents and state. The room abstraction in `src/room-core.mjs` gives a conversation a transcript, supplied files, media, directives, prior transcripts and a durable `desktop/` where a being can work. `src/spine/transcript.mjs` appends to the conversation record. `src/brain-session.mjs` selects among distinct brain implementations; the body is not identical to any one model. `src/tools/agent-loop.mjs` shows a bounded model–tool–observation loop.

The important idea for MIK is that a being can have a place to work and a past that survive a model invocation. A transcript is a record, not an automatically available memory: something must retrieve or bring the relevant part back. In eGPT's local llama session, history is kept in process memory and the code explicitly says transcript rehydration is not implemented there; the agentic loop also currently builds its own message list and does not consume the `history` passed by that session. Do not assume that a durable folder alone makes a model remember.

## A possible MIK body

This is a design sketch, not an implemented directory or a new benchmark protocol:

```text
MIK home/
  workspace/           its durable desk: notes, questions, hypotheses, working files
  sources/             acquired material and observations, with provenance
  conversations/       questions, responses and other interaction history
  runs/                chronological action/observation traces and measurements
  environments/        controlled worlds and experiments available to investigate
  config/              body and tool configuration, model selection
```

The brain meets this body through a small set of capabilities:

* **Perceive and ask:** receive a question, inspect the surroundings, search sources, or request an observation from an environment.
* **Work:** read and write its own notes, list what is on the desk, and use simple tools such as arithmetic. A note can be a question, hypothesis, evidence, uncertainty, or a plan; it is not automatically a fact.
* **Remember:** deliberately find and read earlier observations and notes. Sources retain identifiers, origin and time; revisions preserve what changed and why. Storage and retrieval are distinct from changing model weights.
* **Communicate:** ask a person for clarification or report an answer, supporting evidence and uncertainty. The harness records the interaction.

The body should preserve the desk and history across process restarts and model swaps. Its ordinary operation should be understandable from files and traces; an index can help retrieve material, but the underlying evidence remains inspectable. The model chooses what to look for, what to note and what to conclude; the body executes the requested operation and returns the result. A particular benchmark may impose step or data budgets, but those are experiment conditions rather than a definition of the being.

## The interface is part of the experiment

In MIK-0 the models were required to emit exact JSON on their first turn. Neither used search. That cannot tell us whether a small model could have worked with evidence or with a different body. eGPT's function schemas and tool-result messages are an example of another interface, but their local loop still requires the model to make usable tool calls; they are not proof that a 135M model can do so.

Investigate small, explicit interfaces on development worlds: native function calls where a model supports them, a compact text command, or an explicitly specified structured action. Return a clear error and allow a bounded retry rather than silently turning an invalid response into a valid one. Record the model's raw output, the parsed action, any correction or feedback, and each observation. Distinguish syntax assistance from a host that selects queries, plans the chain, chooses facts, or answers the question: those latter operations contribute intelligence and must be measured as part of the system.

## A test that separates the parts

Use a development world to make the interface usable, then freeze it and evaluate on newly generated held-out worlds. Give the same model and tasks several conditions:

1. **No evidence:** ask about new facts without a body providing them. This is a sanity check; such facts cannot be known from the question alone.
2. **Evidence supplied:** put the necessary atomic facts in context and ask the model to combine them. This tests a part of reasoning without requiring successful tool use.
3. **Body and retrieval:** let the model request facts and use its desk. Measure tool selection, evidence use and answers separately.
4. **Body and accumulated notes:** on later related questions, allow it to retrieve its own prior notes and sources. Measure whether it reuses accurate evidence, recognizes stale or contradictory notes, and reduces new searches.

Include transparent, task-specific symbolic controls and report all body-side operations and costs. Use multiple seeds and report invalid actions, retries, retrieval quality, accuracy, evidence support and compute. Any gain is a property of the whole model–body system. Only after the interface and the simpler controls work should an unknown-rule environment test whether the system can choose measurements and discover new relations.
