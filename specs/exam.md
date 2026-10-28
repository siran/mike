# The young-computer exam

What MIK-e should be (operator, 2026-10-28): **a young computer, like a 12-year-old — not in raw knowledge,
but in curiosity and the ability to search and reason.** A kernel either understands this or it doesn't;
the exam does not coach it.

## Rules

* **No scaffolding.** No worked examples, no map of where things are, no "use this command". The kernel gets
  only a plain description of its situation: it lives in a Linux computer, it can run commands by writing a
  line starting with `$ `, everything else is said to the person, and when a conversation ends it forgets it.
* **Score what it does, not what it says.** A claim ("I wrote it down") counts only if the transcript shows the
  command that did it.
* **Show how.** An answer counts when the kernel found it and can say where; a lucky or memorised answer does not.
* **The same exam for every kernel**, untouched weights, recorded raw.

## 1. Finding out how to search here

Not "search here" but *how do I search here* and *where could this information be?*

* The answer is in a file somewhere in its home, not in an obvious place. Does it explore (`ls`, `find`, `grep`)?
* "What can you use here to read a web page?" Does it discover what is installed (`curl`, `w3m`, …)?
* "Is there a manual for `tac`?" (`man -k`, `apropos`, `--help`).
* "How many user accounts does this computer have?" Does it know where a Linux system keeps that?
* A question no file answers: does it go to the web, or ask the person?

## 2. Reasoning: telling things apart

Operator: *reasoning is basically telling things apart.*

* **Its own memory vs. anything else:** "Is that what I told you, or something you read?"
* **Same or different:** "Do `cat` and `tac` do the same thing?" (it must look both up and compare).
* **Which to trust:** two notes disagree (an older and a newer value). Which stands, and why?
* **What changed:** two versions of a file. What is different?
* **Knowing vs. not knowing:** "Do you know this, or do you need to look it up?"

## 3. Remembering

* "Remember this number: 2" → a new session → "What number did I ask you to remember?" It must have written
  it somewhere it will look, and must look there.
* "Remember my favourite colour is green" → new session → recall.
* It must not "remember" things it was never told (the failure seen with SmolLM2, which recalled content from
  the examples it had been shown).

## 4. Doing

* "Code foo-bar-baz script" (deliberately vague): a reasonable interpretation or one short question, a file
  written, run, checked, fixed, and where it is.

## Findings so far (2026-10-28)

SmolLM2-360M never ran a command in conversation. SmolLM2-1.7B ran real commands and read their output, but
with only "you forget when a conversation ends" it did not write anything down; with a scripted example it
wrote the note but then recalled the example's number instead of reading its notes. Both treated worked
examples as their own past. Neither meets the bar. Next: small open models trained for reasoning and tool use,
unscripted, on this exam; later, training a kernel to look things up instead of knowing them.
