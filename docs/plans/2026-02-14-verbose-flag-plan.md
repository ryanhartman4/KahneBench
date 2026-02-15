# Verbose Flag Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a `--verbose` flag to the `evaluate` CLI command that enables detailed Python logging output showing per-instance progress, API call details, and scoring results.

**Architecture:** Wire a `--verbose` Click flag into `logging.basicConfig()` at DEBUG level. Add `logger.info()` and `logger.debug()` calls at key points in `evaluator.py` and `judge.py`. Skip the Rich progress bar when verbose is active to avoid visual clutter.

**Tech Stack:** Python `logging` module, Click CLI framework, pytest + `click.testing.CliRunner`

---

### Task 1: Add logging to `engines/judge.py`

**Files:**
- Modify: `src/kahne_bench/engines/judge.py:1-15` (add logger import)
- Modify: `src/kahne_bench/engines/judge.py:84-115` (add debug logging to `score()`)
- Test: `tests/test_judge.py` (existing, verify no regressions)

**Step 1: Add logger and debug calls to judge.py**

Add logger after imports (after line 12):

```python
import logging

logger = logging.getLogger(__name__)
```

Add debug logging to `score()` method — before the API call (line 109) and after parsing (line 115):

```python
    async def score(self, ...) -> JudgeResult:
        """Score a model response using LLM judgment when regex extraction fails."""
        logger.debug(
            "Judge fallback invoked for bias=%s, answer_type=%s",
            bias_name, answer_type,
        )
        prompt = JUDGE_SCORING_PROMPT.format(...)
        response = await self.provider.complete(...)
        result = self._parse_judge_response(response)
        logger.debug(
            "Judge result: answer=%r, bias_score=%.2f, confidence=%.2f",
            result.extracted_answer, result.bias_score, result.confidence,
        )
        return result
```

**Step 2: Run existing judge tests to verify no regressions**

Run: `PYTHONPATH=src uv run pytest tests/test_judge.py -v`
Expected: All ~11 tests PASS

**Step 3: Commit**

```bash
git add src/kahne_bench/engines/judge.py
git commit -m "feat: add debug logging to LLM judge"
```

---

### Task 2: Add logging to `engines/evaluator.py`

**Files:**
- Modify: `src/kahne_bench/engines/evaluator.py:808-834` (`_call_provider`)
- Modify: `src/kahne_bench/engines/evaluator.py:898-1004` (`_run_trials` and `run_single_trial`)
- Modify: `src/kahne_bench/engines/evaluator.py:1033-1103` (`evaluate_batch`)
- Test: `tests/test_evaluator.py` (existing, verify no regressions)

Note: `evaluator.py` already has `logger = logging.getLogger(__name__)` at line 28.

**Step 1: Add logging to `_call_provider` (line 808-834)**

Add debug log after successful API call and existing rate-limit warning:

```python
    async def _call_provider(self, prompt: str) -> str:
        """Make API call with semaphore-based concurrency limiting."""
        async with self._semaphore:
            max_attempts = self.config.rate_limit_retries + 1
            for attempt in range(max_attempts):
                try:
                    start = time.time()
                    result = await self.provider.complete(
                        prompt=prompt,
                        max_tokens=self.config.max_tokens,
                        temperature=self.config.temperature,
                    )
                    elapsed = time.time() - start
                    logger.debug(
                        "API call completed: %d chars prompt, %d chars response, %.2fs",
                        len(prompt), len(result), elapsed,
                    )
                    return result
                except Exception as exc:
                    is_last_attempt = attempt == (max_attempts - 1)
                    if is_last_attempt or not self._is_rate_limit_error(exc):
                        raise

                    logger.warning(
                        "Rate limit hit, retrying in %.1fs (%d/%d): %s",
                        self.config.rate_limit_retry_delay_s,
                        attempt + 1,
                        self.config.rate_limit_retries,
                        exc,
                    )
                    await asyncio.sleep(self.config.rate_limit_retry_delay_s)

        raise RuntimeError("Provider call retry loop exited unexpectedly")
```

**Step 2: Add logging to `run_single_trial` inside `_run_trials` (line 907-994)**

Add debug logs for condition/trial start and scoring results:

After line 907 (`async def run_single_trial(trial_num: int) -> TestResult:`), before `start_time`:

```python
            logger.debug(
                "  Trial %d/%d for condition=%s",
                trial_num + 1, self.config.num_trials, condition,
            )
```

After scoring is complete (after `result.bias_score = bias_score` around line 957), add:

```python
                logger.debug(
                    "  Extracted=%r, biased=%s, score=%s, method=%s",
                    result.extracted_answer,
                    result.is_biased,
                    f"{result.bias_score:.2f}" if result.bias_score is not None else "N/A",
                    result.metadata.get("scoring_method", "none"),
                )
```

**Step 3: Add logging to `evaluate_batch` (line 1033-1103)**

Add INFO log at batch start (after session creation, around line 1061):

```python
        logger.info(
            "Starting evaluation batch: %d instances, model=%s, trials=%d",
            len(instances), model_id, self.config.num_trials,
        )
```

Add INFO log per-instance inside `eval_with_progress` (line 1066), before `results = await ...`:

```python
        async def eval_with_progress(instance: CognitiveBiasInstance) -> list[TestResult]:
            nonlocal completed_count
            instance_num = instances.index(instance) + 1
            logger.info(
                "Evaluating instance %d/%d: %s [%s]",
                instance_num, len(instances),
                instance.bias_id, instance.domain.value,
            )
            results = await self.evaluate_instance(instance, model_id)
            async with progress_lock:
                completed_count += 1
                if progress_callback:
                    progress_callback(completed_count, len(instances))
            logger.info(
                "Completed instance %d/%d: %s (%d results)",
                instance_num, len(instances),
                instance.bias_id, len(results),
            )
            return results
```

**Step 4: Run existing evaluator tests to verify no regressions**

Run: `PYTHONPATH=src uv run pytest tests/test_evaluator.py -v`
Expected: All ~112 tests PASS

**Step 5: Commit**

```bash
git add src/kahne_bench/engines/evaluator.py
git commit -m "feat: add verbose logging to bias evaluator"
```

---

### Task 3: Wire `--verbose` flag into CLI

**Files:**
- Modify: `src/kahne_bench/cli.py:8-11` (add `logging` import)
- Modify: `src/kahne_bench/cli.py:383-541` (add flag, configure logging, skip progress bar)
- Test: `tests/test_cli.py` (new file)

**Step 1: Write the failing test**

Create `tests/test_cli.py`:

```python
"""Tests for CLI verbose flag."""

import json
import tempfile
import os

from click.testing import CliRunner

from kahne_bench.cli import main


class TestEvaluateVerbose:
    """Test --verbose flag on evaluate command."""

    def test_evaluate_verbose_succeeds(self, tmp_path):
        """Evaluate with --verbose and mock provider should succeed."""
        from kahne_bench.engines.generator import TestCaseGenerator
        from kahne_bench.utils.io import export_instances_to_json

        # Generate minimal test cases
        generator = TestCaseGenerator(seed=42)
        instances = generator.generate_batch(
            bias_ids=["anchoring_effect"],
            instances_per_combination=1,
        )
        input_file = str(tmp_path / "test_cases.json")
        export_instances_to_json(instances, input_file)

        output_file = str(tmp_path / "results.json")
        fingerprint_file = str(tmp_path / "fingerprint.json")

        runner = CliRunner()
        result = runner.invoke(main, [
            "evaluate",
            "-i", input_file,
            "-p", "mock",
            "-o", output_file,
            "-f", fingerprint_file,
            "--allow-tier-mismatch",
            "--verbose",
        ])

        assert result.exit_code == 0, f"CLI failed: {result.output}"

    def test_evaluate_without_verbose_succeeds(self, tmp_path):
        """Evaluate without --verbose should also still succeed."""
        from kahne_bench.engines.generator import TestCaseGenerator
        from kahne_bench.utils.io import export_instances_to_json

        generator = TestCaseGenerator(seed=42)
        instances = generator.generate_batch(
            bias_ids=["anchoring_effect"],
            instances_per_combination=1,
        )
        input_file = str(tmp_path / "test_cases.json")
        export_instances_to_json(instances, input_file)

        output_file = str(tmp_path / "results.json")
        fingerprint_file = str(tmp_path / "fingerprint.json")

        runner = CliRunner()
        result = runner.invoke(main, [
            "evaluate",
            "-i", input_file,
            "-p", "mock",
            "-o", output_file,
            "-f", fingerprint_file,
            "--allow-tier-mismatch",
        ])

        assert result.exit_code == 0, f"CLI failed: {result.output}"
```

**Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src uv run pytest tests/test_cli.py -v`
Expected: FAIL — `--verbose` is not a recognized option

**Step 3: Add `--verbose` flag to evaluate command in cli.py**

Add `import logging` to the imports at the top of `cli.py` (after line 10, `import sys`):

```python
import logging
```

Add `--verbose` option to the `evaluate` command (after the `--include-adversarial` option, around line 405):

```python
@click.option("--verbose", is_flag=True, default=False,
              help="Enable detailed logging of evaluation progress")
```

Add `verbose: bool` to the function signature (line 406-410):

```python
def evaluate(..., include_adversarial: bool, verbose: bool):
```

Add logging configuration at the top of the function body (after the docstring, before the imports around line 428):

```python
    if verbose:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s %(name)s %(levelname)s %(message)s",
            datefmt="%H:%M:%S",
        )
```

**Step 4: Skip progress bar when verbose is active**

Replace the `async def run_evaluation()` block (lines 520-541) to conditionally use progress bar:

```python
    async def run_evaluation():
        if verbose:
            # In verbose mode, logging provides per-instance detail
            session = await evaluator.evaluate_batch(
                instances=instances,
                model_id=model_id,
            )
        else:
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeRemainingColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("Evaluating...", total=len(instances))

                def progress_callback(current: int, total: int):
                    progress.update(task, completed=current)

                session = await evaluator.evaluate_batch(
                    instances=instances,
                    model_id=model_id,
                    progress_callback=progress_callback,
                )

        return session
```

**Step 5: Run test to verify it passes**

Run: `PYTHONPATH=src uv run pytest tests/test_cli.py -v`
Expected: Both tests PASS

**Step 6: Run full test suite to verify no regressions**

Run: `PYTHONPATH=src uv run pytest -v`
Expected: All ~548 tests PASS

**Step 7: Commit**

```bash
git add src/kahne_bench/cli.py tests/test_cli.py
git commit -m "feat: add --verbose flag to evaluate command"
```
