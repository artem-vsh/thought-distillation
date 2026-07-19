#!/usr/bin/env python3
"""RL training on arithmetic problems with a Tinker-served LLM judge.

**Input:** ``data/arithmetic_operations.csv`` (operation, solution), same prompts
as ``ask_arithmetic.make_prompt``.

**Judge:** ``openai/gpt-oss-20b`` with renderer ``gpt_oss_high_reasoning``
(high reasoning effort). Rewards are parsed from ``<score>0|1</score>``.

**Policy (trained):** defaults to the same base model in *instant* mode (same as
``ask_arithmetic --instant``): renderer ``gpt_oss_no_sysprompt`` plus final-channel
prefill so rollouts skip CoT. Override with ``model_name`` / ``renderer_name`` /
``generation_prefill``.

Example::

    source .venv/bin/activate
    python train_math_llm_judge.py \\
        groups_per_batch=8 \\
        group_size=4 \\
        max_steps=2 \\
        behavior_if_log_dir_exists=delete
"""

from __future__ import annotations

import asyncio
import csv
import logging
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import chz
import tinker
from tinker.types import LossFnType

from ask_arithmetic import (
    FINAL_CHANNEL_PREFILL,
    INPUT_CSV as DEFAULT_DATA_CSV,
    RENDERER_BY_EFFORT,
    load_hf_token,
    load_tinker_api_key,
    make_prompt,
)
from tinker_cookbook import checkpoint_utils, cli_utils, model_info
from tinker_cookbook.completers import MessageCompleter, TinkerMessageCompleter
from tinker_cookbook.renderers import Renderer, get_renderer, get_text_content
from tinker_cookbook.rl.train import AsyncConfig, Config, main as rl_main
from tinker_cookbook.rl.types import (
    Action,
    ActionExtra,
    Env,
    EnvGroupBuilder,
    Metrics,
    Observation,
    RLDataset,
    RLDatasetBuilder,
    StepResult,
    Trajectory,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import logtree
from tinker_cookbook.utils.logtree_formatters import ConversationFormatter

logger = logging.getLogger(__name__)

DEFAULT_JUDGE_MODEL = "openai/gpt-oss-20b"
DEFAULT_JUDGE_RENDERER = "gpt_oss_high_reasoning"
# High reasoning spends tokens in the analysis channel before the final score;
# 2k often truncates mid-CoT and never emits <score>. Match ask_arithmetic headroom.
DEFAULT_JUDGE_MAX_TOKENS = 16384
DEFAULT_POLICY_MODEL = "openai/gpt-oss-20b"
# Match ask_arithmetic --instant: no Reasoning system line + force final channel.
DEFAULT_POLICY_REASONING_EFFORT = "instant"
DEFAULT_POLICY_RENDERER = RENDERER_BY_EFFORT[DEFAULT_POLICY_REASONING_EFFORT]
DEFAULT_POLICY_PREFILL = FINAL_CHANNEL_PREFILL

# Parse <score>0</score> or <score>1</score> (and floats in between).
_SCORE_RE = re.compile(r"<score>\s*([01](?:\.\d+)?|0?\.\d+)\s*</score>", re.IGNORECASE)
# Fallback: bare yes/no or correct/incorrect near the end of the judge reply.
_YES_NO_RE = re.compile(
    r"\b(yes|no|correct|incorrect|true|false)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class MathProblem:
    """One arithmetic problem and its reference solution."""

    operation: str
    solution: str


def load_math_problems(input_csv: Path) -> list[MathProblem]:
    """Load operation/solution rows from the arithmetic processor CSV."""
    with input_csv.open(newline="", encoding="utf-8-sig") as input_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames is None:
            raise ValueError(f"Input CSV is empty: {input_csv}")
        required = {"operation", "solution"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"Input CSV must contain columns {sorted(required)}; "
                f"missing {sorted(missing)}: {input_csv}"
            )

        problems: list[MathProblem] = []
        for line_number, row in enumerate(reader, start=2):
            operation = (row.get("operation") or "").strip()
            solution = (row.get("solution") or "").strip()
            if not operation:
                raise ValueError(f"CSV row {line_number} has an empty operation")
            if not solution:
                raise ValueError(f"CSV row {line_number} has an empty solution")
            problems.append(MathProblem(operation=operation, solution=solution))
    if not problems:
        raise ValueError(f"No problems found in {input_csv}")
    return problems


def make_judge_messages(
    operation: str,
    solution: str,
    policy_response: str,
) -> list[dict[str, str]]:
    """Build the user message for the LLM judge.

    The judge sees the same policy prompt used by ``ask_arithmetic``, the
    reference solution from the data file, and the policy's free-form answer.
    """
    problem_prompt = make_prompt(operation)
    content = "\n".join(
        [
            "You are grading a model's answer to an arithmetic problem.",
            "Decide whether the model's answer is numerically equivalent to the reference solution.",
            "Ignore minor formatting differences (whitespace, trailing zeros, signs like +42).",
            "If the model output contains reasoning, grade only the final numeric result.",
            "",
            f"Problem prompt: {problem_prompt}",
            f"Expression: {operation}",
            f"Reference solution: {solution}",
            f"Model answer: {policy_response}",
            "",
            "Respond with <score>1</score> if the answer is correct, or <score>0</score> if incorrect.",
            "Do not wrap any other text inside the score tags.",
        ]
    )
    return [{"role": "user", "content": content}]


def extract_judge_score(response: str) -> float:
    """Extract a [0, 1] score from the judge's free-form reply."""
    text = response.strip()
    if not text:
        return 0.0

    match = _SCORE_RE.search(text)
    if match is not None:
        try:
            score = float(match.group(1))
        except ValueError:
            score = 0.0
        return min(1.0, max(0.0, score))

    # Prefer the last yes/no-style token if the model forgot the tags.
    matches = list(_YES_NO_RE.finditer(text))
    if matches:
        token = matches[-1].group(1).lower()
        if token in {"yes", "correct", "true"}:
            return 1.0
        if token in {"no", "incorrect", "false"}:
            return 0.0

    logger.warning("Failed to parse judge score; defaulting to 0. Response: %s", text[:500])
    return 0.0


def create_tinker_judge(
    *,
    judge_model: str,
    judge_renderer_name: str,
    max_tokens: int,
    temperature: float = 0.0,
    base_url: str | None = None,
    judge_model_path: str | None = None,
) -> MessageCompleter:
    """Serve the LLM judge from Tinker (sampling client + renderer).

    Args:
        judge_model: Base model name (tokenizer / renderer identity).
        judge_renderer_name: Usually high-reasoning (e.g. gpt_oss_high_reasoning).
        judge_model_path: Optional ``tinker://…`` sampler weights. When set, the
            judge samples from this checkpoint instead of the base model (used
            by the autoresearch loop after the first train run).
    """
    tokenizer = get_tokenizer(judge_model)
    renderer = get_renderer(
        judge_renderer_name,
        tokenizer=tokenizer,
        model_name=judge_model,
    )
    service_client = tinker.ServiceClient(base_url=base_url)
    if judge_model_path:
        logger.info("Judge sampling from checkpoint: %s", judge_model_path)
        sampling_client = service_client.create_sampling_client(
            model_path=judge_model_path
        )
    else:
        logger.info("Judge sampling from base model: %s", judge_model)
        sampling_client = service_client.create_sampling_client(base_model=judge_model)
    return TinkerMessageCompleter(
        sampling_client=sampling_client,
        renderer=renderer,
        max_tokens=max_tokens,
        temperature=temperature,
    )


class MathLLMJudgeEnv(Env):
    """Single-turn math problem graded by a Tinker-served LLM judge."""

    def __init__(
        self,
        problem: MathProblem,
        renderer: Renderer,
        judge_llm: MessageCompleter,
        format_coef: float = 0.1,
        require_stop_sequence_for_format: bool = True,
        generation_prefill: str | None = DEFAULT_POLICY_PREFILL,
    ):
        self.problem = problem
        self.renderer = renderer
        self.judge_llm = judge_llm
        self.format_coef = format_coef
        self.require_stop_sequence_for_format = require_stop_sequence_for_format
        self.generation_prefill = generation_prefill

    @property
    def stop_condition(self):
        return self.renderer.get_stop_sequences()

    def get_question(self) -> str:
        """Policy user message — same wording as ``ask_arithmetic.make_prompt``."""
        return make_prompt(self.problem.operation)

    def _convo_prefix(self) -> list[dict[str, str]]:
        return [{"role": "user", "content": self.get_question()}]

    async def initial_observation(self) -> tuple[Observation, Any]:
        # Same prompt construction as ask_arithmetic instant mode (renderer + prefill).
        return (
            self.renderer.build_generation_prompt(
                self._convo_prefix(),
                prefill=self.generation_prefill,
            ),
            self.stop_condition,
        )

    async def step(self, action: Action, *, extra: ActionExtra | None = None) -> StepResult:
        convo = self._convo_prefix()
        message, termination = self.renderer.parse_response(action)
        content = get_text_content(message)

        well_formed = (
            termination.is_stop_sequence
            if self.require_stop_sequence_for_format
            else termination.is_clean
        )
        format_score = float(well_formed)

        judge_messages = make_judge_messages(
            operation=self.problem.operation,
            solution=self.problem.solution,
            policy_response=content,
        )
        judge_response = await self.judge_llm(judge_messages)
        judge_text = get_text_content(judge_response)
        answer_score = extract_judge_score(judge_text)

        # Same reward shape as ProblemEnv: format penalty + correctness.
        total_reward = self.format_coef * (format_score - 1.0) + answer_score

        with logtree.scope_header("Prompt"):
            logtree.log_formatter(ConversationFormatter(messages=convo))
        with logtree.scope_header("Policy Response"):
            logtree.log_formatter(ConversationFormatter(messages=[message]))
        with logtree.scope_header("LLM Judge"):
            logtree.table_from_dict(
                {
                    "operation": self.problem.operation,
                    "reference_solution": self.problem.solution,
                    "judge_score": f"{answer_score:.3f}",
                    "format_valid": bool(format_score),
                    "reward": f"{total_reward:.3f}",
                },
                caption="Judge reward components",
            )
            logtree.details(judge_text, summary="Judge output", pre=True)

        return StepResult(
            reward=total_reward,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics={
                "format": format_score,
                "correct": answer_score,
            },
            logs={
                "operation": self.problem.operation,
                "solution": self.problem.solution,
                "judge_score": answer_score,
            },
        )


@dataclass(frozen=True)
class MathLLMJudgeGroupBuilder(EnvGroupBuilder):
    """Group of identical math problems for GRPO-style multi-sample rollouts."""

    problem: MathProblem
    renderer: Renderer
    judge_llm: MessageCompleter
    group_size: int
    format_coef: float = 0.1
    generation_prefill: str | None = DEFAULT_POLICY_PREFILL
    dataset_name: str = "arithmetic_llm_judge"

    async def make_envs(self) -> Sequence[Env]:
        return [
            MathLLMJudgeEnv(
                problem=self.problem,
                renderer=self.renderer,
                judge_llm=self.judge_llm,
                format_coef=self.format_coef,
                generation_prefill=self.generation_prefill,
            )
            for _ in range(self.group_size)
        ]

    async def compute_group_rewards(
        self, trajectory_group: list[Trajectory], env_group: Sequence[Env]
    ) -> list[tuple[float, Metrics]]:
        return [(0.0, {}) for _ in range(len(trajectory_group))]

    def logging_tags(self) -> list[str]:
        return [self.dataset_name]


class MathLLMJudgeDataset(RLDataset):
    """Batches of arithmetic problems graded by the LLM judge."""

    def __init__(
        self,
        problems: Sequence[MathProblem],
        batch_size: int,
        group_size: int,
        renderer: Renderer,
        judge_llm: MessageCompleter,
        format_coef: float = 0.1,
        generation_prefill: str | None = DEFAULT_POLICY_PREFILL,
    ):
        self.problems = list(problems)
        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer = renderer
        self.judge_llm = judge_llm
        self.format_coef = format_coef
        self.generation_prefill = generation_prefill

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        batch_start = index * self.batch_size
        batch_end = min((index + 1) * self.batch_size, len(self.problems))
        assert batch_start < batch_end, "Incorrect batch size"
        return [
            MathLLMJudgeGroupBuilder(
                problem=problem,
                renderer=self.renderer,
                judge_llm=self.judge_llm,
                group_size=self.group_size,
                format_coef=self.format_coef,
                generation_prefill=self.generation_prefill,
            )
            for problem in self.problems[batch_start:batch_end]
        ]

    def __len__(self) -> int:
        return math.ceil(len(self.problems) / self.batch_size)


@chz.chz
class MathLLMJudgeDatasetBuilder(RLDatasetBuilder):
    """Build train/test RL datasets from the arithmetic CSV + Tinker LLM judge."""

    data_path: str = str(DEFAULT_DATA_CSV)
    batch_size: int
    group_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    judge_model: str = DEFAULT_JUDGE_MODEL
    judge_renderer_name: str = DEFAULT_JUDGE_RENDERER
    judge_max_tokens: int = DEFAULT_JUDGE_MAX_TOKENS
    judge_temperature: float = 0.0
    # Optional tinker:// sampler path — judge from a trained checkpoint.
    judge_model_path: str | None = None
    test_fraction: float = 0.05
    seed: int = 0
    format_coef: float = 0.1
    generation_prefill: str | None = DEFAULT_POLICY_PREFILL
    base_url: str | None = None
    test_group_size: int = 1

    async def __call__(self) -> tuple[MathLLMJudgeDataset, MathLLMJudgeDataset | None]:
        problems = load_math_problems(Path(self.data_path))
        train_problems, test_problems = _split_problems(
            problems, test_fraction=self.test_fraction, seed=self.seed
        )

        policy_tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        policy_renderer = get_renderer(
            self.renderer_name,
            tokenizer=policy_tokenizer,
            model_name=self.model_name_for_tokenizer,
        )
        judge_llm = create_tinker_judge(
            judge_model=self.judge_model,
            judge_renderer_name=self.judge_renderer_name,
            max_tokens=self.judge_max_tokens,
            temperature=self.judge_temperature,
            base_url=self.base_url,
            judge_model_path=self.judge_model_path,
        )

        train_dataset = MathLLMJudgeDataset(
            problems=train_problems,
            batch_size=self.batch_size,
            group_size=self.group_size,
            renderer=policy_renderer,
            judge_llm=judge_llm,
            format_coef=self.format_coef,
            generation_prefill=self.generation_prefill,
        )
        if not test_problems:
            return train_dataset, None

        test_dataset = MathLLMJudgeDataset(
            problems=test_problems,
            batch_size=max(1, min(self.batch_size, len(test_problems))),
            group_size=self.test_group_size,
            renderer=policy_renderer,
            judge_llm=judge_llm,
            format_coef=self.format_coef,
            generation_prefill=self.generation_prefill,
        )
        return train_dataset, test_dataset


def _split_problems(
    problems: Sequence[MathProblem],
    *,
    test_fraction: float,
    seed: int,
) -> tuple[list[MathProblem], list[MathProblem]]:
    """Deterministic train/test split (shuffle by seed, then cut)."""
    if not 0.0 <= test_fraction < 1.0:
        raise ValueError(f"test_fraction must be in [0, 1), got {test_fraction}")

    indexed = list(enumerate(problems))
    # Stable shuffle without numpy dependency for this helper.
    rng = _SeededRNG(seed)
    rng.shuffle(indexed)
    ordered = [problem for _, problem in indexed]

    if test_fraction == 0.0 or len(ordered) < 2:
        return ordered, []

    n_test = max(1, int(round(len(ordered) * test_fraction)))
    n_test = min(n_test, len(ordered) - 1)
    test_problems = ordered[:n_test]
    train_problems = ordered[n_test:]
    return train_problems, test_problems


class _SeededRNG:
    """Tiny deterministic PRNG for dataset splits (stdlib only)."""

    def __init__(self, seed: int):
        self._state = seed & 0xFFFFFFFF

    def _next(self) -> int:
        # Numerical Recipes LCG
        self._state = (1664525 * self._state + 1013904223) & 0xFFFFFFFF
        return self._state

    def shuffle(self, items: list[Any]) -> None:
        for i in range(len(items) - 1, 0, -1):
            j = self._next() % (i + 1)
            items[i], items[j] = items[j], items[i]


@chz.chz
class CLIConfig:
    """CLI configuration for math RL with a Tinker LLM judge."""

    # Policy model (trained) — defaults match ask_arithmetic --instant
    model_name: str = DEFAULT_POLICY_MODEL
    renderer_name: str | None = DEFAULT_POLICY_RENDERER
    # Final-channel prefill skips CoT; set to None to disable (e.g. low/medium/high).
    generation_prefill: str | None = DEFAULT_POLICY_PREFILL
    load_checkpoint_path: str | None = None
    lora_rank: int = 32

    # Data / prompt source (math processor CSV)
    data_path: str = str(DEFAULT_DATA_CSV)
    test_fraction: float = 0.05
    seed: int = 0

    # LLM judge served from Tinker (GPT-OSS-20b @ high reasoning by default)
    judge_model: str = DEFAULT_JUDGE_MODEL
    judge_renderer_name: str = DEFAULT_JUDGE_RENDERER
    judge_max_tokens: int = DEFAULT_JUDGE_MAX_TOKENS
    judge_temperature: float = 0.0
    # When set (e.g. previous loop sampler_path), judge uses those weights with
    # high-reasoning renderer instead of the base model.
    judge_model_path: str | None = None

    # Training hyperparameters
    group_size: int = 4
    groups_per_batch: int = 16
    learning_rate: float = 1e-5
    max_tokens: int = 128
    temperature: float = 1.0
    kl_penalty_coef: float = 0.0
    format_coef: float = 0.1
    num_substeps: int = 1
    loss_fn: LossFnType = "importance_sampling"
    loss_fn_config: dict[str, Any] | None = None

    # Logging / checkpoints
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    compute_post_kl: bool = False
    eval_every: int = 20
    save_every: int = 20
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"

    # Service
    base_url: str | None = None
    max_steps_off_policy: int | None = None
    max_steps: int | None = None


def get_dataset_builder(cli_config: CLIConfig, renderer_name: str) -> RLDatasetBuilder:
    return MathLLMJudgeDatasetBuilder(
        data_path=cli_config.data_path,
        batch_size=cli_config.groups_per_batch,
        group_size=cli_config.group_size,
        model_name_for_tokenizer=cli_config.model_name,
        renderer_name=renderer_name,
        judge_model=cli_config.judge_model,
        judge_renderer_name=cli_config.judge_renderer_name,
        judge_max_tokens=cli_config.judge_max_tokens,
        judge_temperature=cli_config.judge_temperature,
        judge_model_path=cli_config.judge_model_path,
        test_fraction=cli_config.test_fraction,
        seed=cli_config.seed,
        format_coef=cli_config.format_coef,
        generation_prefill=cli_config.generation_prefill,
        base_url=cli_config.base_url,
    )


async def cli_main(cli_config: CLIConfig) -> None:
    """Convert CLI config into an RL Config and run Tinker training."""
    load_tinker_api_key()
    load_hf_token()

    renderer_name = await checkpoint_utils.resolve_renderer_name_from_checkpoint_or_default_async(
        model_name=cli_config.model_name,
        explicit_renderer_name=cli_config.renderer_name,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        base_url=cli_config.base_url,
    )
    # Fall back to recommended renderer if resolve returns None somehow.
    if not renderer_name:
        renderer_name = model_info.get_recommended_renderer_name(cli_config.model_name)

    model_tag = cli_config.model_name.replace("/", "-")
    run_name = (
        f"math-llm-judge-{model_tag}-{cli_config.lora_rank}rank-"
        f"{cli_config.learning_rate}lr-{cli_config.group_size}group-"
        f"{cli_config.groups_per_batch}batch-{cli_config.loss_fn}-"
        f"seed{cli_config.seed}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    )
    _project_root = Path(__file__).resolve().parent
    log_path = cli_config.log_path or str(
        _project_root / "output" / "math_llm_judge" / run_name
    )
    wandb_name = cli_config.wandb_name or run_name

    logger.info(
        "Training policy=%s renderer=%s prefill=%r load_ckpt=%r | "
        "judge=%s renderer=%s judge_path=%r | data=%s",
        cli_config.model_name,
        renderer_name,
        cli_config.generation_prefill,
        cli_config.load_checkpoint_path,
        cli_config.judge_model,
        cli_config.judge_renderer_name,
        cli_config.judge_model_path,
        cli_config.data_path,
    )

    config = Config(
        learning_rate=cli_config.learning_rate,
        dataset_builder=get_dataset_builder(cli_config, renderer_name),
        model_name=cli_config.model_name,
        recipe_name="recipe_math_llm_judge",
        renderer_name=renderer_name,
        lora_rank=cli_config.lora_rank,
        max_tokens=cli_config.max_tokens,
        temperature=cli_config.temperature,
        wandb_project=cli_config.wandb_project,
        wandb_name=wandb_name,
        log_path=log_path,
        base_url=cli_config.base_url,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        compute_post_kl=cli_config.compute_post_kl,
        kl_penalty_coef=cli_config.kl_penalty_coef,
        num_substeps=cli_config.num_substeps,
        eval_every=cli_config.eval_every,
        save_every=cli_config.save_every,
        async_config=AsyncConfig(
            max_steps_off_policy=cli_config.max_steps_off_policy,
            groups_per_batch=cli_config.groups_per_batch,
        )
        if cli_config.max_steps_off_policy is not None
        else None,
        loss_fn=cli_config.loss_fn,
        loss_fn_config=cli_config.loss_fn_config,
        max_steps=cli_config.max_steps,
    )

    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)
    await rl_main(config)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(cli_config))


if __name__ == "__main__":
    main()
