# Hypothesis and scope

MIK investigates whether external information and interaction reduce the learned capacity required for specified tasks. The present experiment tests a narrow implication: can frozen small models retrieve and join four previously unknown facts? A success is evidence only for this synthetic task distribution, not a minimum for general intelligence. A failure might reflect action formatting, language, sequence length, decoding, or reasoning, not necessarily an intrinsic capacity limit.

The kernel makes each action decision. The host enforces a step budget and provides a subject-indexed database lookup. The lookup returns only records directly attached to the requested subject; it does not traverse relations or select an answer. The task-specific symbolic comparator does traverse the prescribed relation sequence and is explicitly identified as such.

We distinguish model parameters from system cost. Report parameter count, model revision and dtype, inference tokens, wall and process CPU time, lookup count, task generation, and host logic. These are incomplete proxies: training cost, energy, peak RAM, FLOPs and human implementation effort are not currently measured. No efficiency conclusion follows from parameter counts alone.
