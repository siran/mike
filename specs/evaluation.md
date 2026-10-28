# Measurement

Each JSONL evaluation begins with runtime metadata, continues with one complete trace per attempted task, and ends with an aggregate summary if the run finishes. Each trace contains raw model output, parsed actions, observations, inference token counts and timing, tool timing, process CPU time, result status and scoring. The metadata records the world hash, model snapshot, package versions and runtime settings. Final answer keys remain in the local world file, never in the model state or a search response.

* Accuracy: exact equality of the final answer and generated property identifier. An invalid action or exhausted budget is incorrect.
* Evidence-supported accuracy: the answer is correct, every expected record ID is cited, and every citation appeared in an observation. Extra observed citations are permitted. This checks provenance, not validity of reasoning.
* Abstention: an explicit answer with null value; budget exhaustion and invalid output are separate failures. All retrieval tasks in this benchmark are answerable with enabled tools; with tools disabled all are unanswerable from available observations.
* Cost proxies: total input/output tokens, search calls, wall time, process CPU time, and model parameter count. Model loading and benchmark generation are excluded from per-task timings; tool index construction is also excluded. Total training cost, energy, FLOPs, memory usage, implementation cost, and storage footprint are not yet measured.
* Wilson intervals are marginal binomial summaries; tasks share one generator and model and are not independent samples from a broad universe of tasks. They do not quantify model-size effects. One world seed limits generalization.

The protocol does not yet produce calibrated confidence, verify linguistic explanations, measure experiment-selection value, or establish causal separation of factual knowledge and reasoning.
