---
description: Familiarize yourself with the Kahne-Bench codebase by reading all documentation
allowed-tools: Read, Glob, Grep, Task
---

# Learn Kahne-Bench Codebase

Read all files in the `docs/` directory to understand the theoretical foundation and specification:

1. Read `docs/Kahne-Bench_ A Framework for Implementing the Kahneman-Tversky Behavioral Bias Benchmark in Python.md` - the complete specification
2. Read `docs/IMPROVEMENT_TRACKER.md` - recent changes and future work

Then read `CLAUDE.md` to understand:
- Project architecture and module structure
- The 69 cognitive biases and 16 categories
- The 6 evaluation metrics (BMS, BCI, BMP, HAS, RCI, CAS)
- Multi-scale testing methodology (MICRO, MESO, MACRO, META)
- Development commands and testing patterns

Explore the codebase to understand:
- Core types in `src/kahne_bench/core.py`
- Bias taxonomy in `src/kahne_bench/biases/taxonomy.py`
- Test generators in `src/kahne_bench/engines/generator.py`
- Evaluation engine in `src/kahne_bench/engines/evaluator.py`
- Metrics in `src/kahne_bench/metrics/core.py`

Once you have reviewed thoroughly, provide a summary confirming your understanding and say "We will begin once you confirm the focus area."
