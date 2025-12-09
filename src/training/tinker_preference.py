"""
Types for Prometheus-Eval that takes in a rubric, two model outputs, and an optional reference
to return reasoning and pairwise preference.

Reference: https://github.com/prometheus-eval/prometheus-eval
"""

import logging
from dataclasses import dataclass

import torch
from tinker import SamplingClient, types
from tinker_cookbook import renderers
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.tokenizer_utils import Tokenizer
from tinker_cookbook.rl.types import EnvGroupBuilder, RLDataset, RLDatasetBuilder
from datasets import Dataset, load_dataset
logger = logging.getLogger(__name__)




class SummarizationDataset(RLDataset):
    def __init__(
        self,
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        split: Literal["train", "test"] = "train",
        seed: int = 0,
    ):
        if split not in ("train", "test"):
            raise ValueError("split must be 'train' or 'test'")

        self.ds = cast(Dataset, load_dataset("openai/gsm8k", name="main", split=split))
        if split == "train":
            self.ds = self.ds.shuffle(seed=seed)
        self.batch_size = batch_size
        self.group_size = group_size if split == "train" else 1
        self.renderer = renderer

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        batch_start = index * self.batch_size
        batch_end = min((index + 1) * self.batch_size, len(self.ds))
        assert batch_start < batch_end, "Incorrect batch size"
        return [
            builder
            for row in self.ds.select(range(batch_start, batch_end))
            if (builder := self._make_env_group_builder(row, self.group_size)) is not None  # pyright: ignore[reportArgumentType]
        ]
    def __len__(self) -> int:
        return math.ceil(len(self.ds) / self.batch_size)
    def _make_env_group_builder(
        self, x: dict[str, str], group_size: int
    ) -> SummarizationGroupBuilder | None:
        try:
            problem = x["question"]
            answer = extract_gsm8k_final_answer(x["answer"])
        except Exception as e:
            logger.warning(f"Failed to parse SLF5K row: {e}")
            return None
        return SummarizationGroupBuilder(
            env_thunk=partial(
                SummarizationEnv, 
            ),
            num_envs=group_size,
        )



@dataclass
class PrometheusEvalComparison:
    prompt_conversation: list[renderers.Message]
    completion_A: list[renderers.Message]
    completion_B: list[renderers.Message]
    rubric: str
    reference: str | None = None

    def swap(self) -> "PrometheusEvalComparison":
        return PrometheusEvalComparison(
            prompt_conversation=self.prompt_conversation,
            completion_A=self.completion_B,
            completion_B=self.completion_A,
            rubric=self.rubric,
            reference=self.reference,
        )


@dataclass
class LabeledPrometheusEvalComparison:
    comparison: PrometheusEvalComparison
    label: str  # "{reasoning} [RESULT] (Either "A" or "B")"


class PrometheusEvalPreferenceModel:
    async def __call__(self, comparison: PrometheusEvalComparison) -> float:
        """
        1: A is strongly preferred
        0: Tie
        -1: B is strongly preferred

        Caveat: Prometheus-eval training data do not include examples that are tied.
        """
        raise NotImplementedError


class PrometheusEvalComparisonRenderer:
    def build_generation_prompt(self, comparison: PrometheusEvalComparison) -> types.ModelInput:
        raise NotImplementedError

    def to_tokens_weights(
        self, labeled_comparison: LabeledPrometheusEvalComparison
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @property
    def tokenizer(self) -> Tokenizer:
        raise NotImplementedError


class PrometheusEvalComparisonRendererFromChatRenderer(PrometheusEvalComparisonRenderer):
    def __init__(self, convo_renderer: renderers.Renderer):
        self.convo_renderer = convo_renderer

    def _comparison_to_convo(self, comparison: PrometheusEvalComparison, use_metrics = True) -> list[renderers.Message]:
        comparison_prompt_template = """\
You are a fair judge assistant assigned to deliver insightful feedback that compares individual performances, highlighting how each stands relative to others within the same cohort.

###Task Description:
An instruction which is a summarization task, two summaries to evaluate (denoted as Summary A and Summary B), and some evaluation metrics are given. Each metric has a description and corresponding good and bad behaviours.
1. Write a detailed feedback that assess the quality of the two summaries strictly based on the given evaluation metrics, not evaluating in general.
2. Make comparisons between Summary A and Summary B. Instead of examining Summary A and Summary B separately, go straight to the point and mention about the commonalities and differences between them based on the evaluation metrics. Give a score for each summary based on each metric and the weightage associated with that metric and give a reasoning for the same. Also provide a final combined score at the end out of 10.
3. After writing the feedback and the reasoning for the score of each summary, indicate the better summary, either "A" or "B".
4. The output format should look as follows: "Feedback: (write a feedback for criteria) Final Score: A <score_A>/10, B <score_B>/10 [RESULT] (Either "A" or "B")"
5. Please do not generate any other opening, closing, and explanations.
6. Please ensure that the output format in point 4 is strictly followed. Example output - "Feedback: <your feedback> Final Score: A 7.2/10, B 6.1/10 [RESULT] A"

###The instruction to evaluate:
{instruction}

###Summary A to evaluate:
{completion_A}

###Summary B to evaluate:
{completion_B}

###Evaluation Metrics:
{rubric}

###Feedback:
"""
        if not use_metrics:
            comparison_prompt_template = """\
You are a fair judge assistant assigned to deliver insightful feedback that compares individual performances, highlighting how each stands relative to others within the same cohort.

###Task Description:
An instruction which is a summarization task, two summaries to evaluate (denoted as Summary A and Summary B).
1. Write a detailed feedback that assess the quality of the two summaries.
2. Make comparisons between Summary A and Summary B. Instead of examining Summary A and Summary B separately, go straight to the point and mention about the commonalities and differences between them based on the evaluation metrics.
3. After writing the feedback and the reasoning for the score of each summary, indicate the better summary, either "A" or "B".
4. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (Either "A" or "B")"
5. Please do not generate any other opening, closing, and explanations.
6. Please ensure that the output format in point 4 is strictly followed. Example output - "Feedback: <your feedback> [RESULT] A"
###The instruction to evaluate:
{instruction}

###Summary A to evaluate:
{completion_A}

###Summary B to evaluate:
{completion_B}

###Feedback:
"""


        instruction = ""
        for msg in comparison.prompt_conversation:
            if msg["role"] == "user" or msg["role"] == "assistant":
                instruction += f"{msg['role']}: {msg['content']}\n"
        comparison_prompt = comparison_prompt_template.format(
            instruction=instruction,
            completion_A=comparison.completion_A[0]["content"],
            completion_B=comparison.completion_B[0]["content"],
            reference=comparison.reference if comparison.reference else "None",
            rubric=comparison.rubric,
        )
        return [{"role": "user", "content": comparison_prompt}]

    def build_generation_prompt(self, comparison: PrometheusEvalComparison) -> types.ModelInput:
        return self.convo_renderer.build_generation_prompt(self._comparison_to_convo(comparison))

    def to_tokens_weights(
        self, labeled_comparison: LabeledPrometheusEvalComparison
    ) -> tuple[torch.Tensor, torch.Tensor]:
        convo = self._comparison_to_convo(labeled_comparison.comparison)
        convo_with_pref = convo + [{"role": "assistant", "content": labeled_comparison.label}]
        tokens, weights = self.convo_renderer.build_supervised_example(
            convo_with_pref, train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES
        )
        return tokens, weights

    @property
    def tokenizer(self) -> Tokenizer:
        return self.convo_renderer.tokenizer


class PrometheusEvalPreferenceModelFromChatRenderer(PrometheusEvalPreferenceModel):
    def __init__(self, convo_renderer: renderers.Renderer, sampling_client: SamplingClient):
        self.comparison_renderer = PrometheusEvalComparisonRendererFromChatRenderer(convo_renderer)
        self.sampling_client = sampling_client

    async def __call__(self, comparison: PrometheusEvalComparison) -> float:
        pm_input = self.comparison_renderer.build_generation_prompt(comparison)
        response = await self.sampling_client.sample_async(
            pm_input,
            num_samples=1,
            sampling_params=types.SamplingParams(
                temperature=0.0,
                max_tokens=1024,
                stop=self.comparison_renderer.convo_renderer.get_stop_sequences(),
            ),
        )
        str_output = self.comparison_renderer.tokenizer.decode(response.sequences[0].tokens).strip()
        # Expected output format: "{feedback} [RESULT] (Either "A" or "B")"
        if "[RESULT]" not in str_output:
            logger.warning(f"Invalid output preference model output: '{str_output}'")
            return 0.0
        result = str_output.split("[RESULT]")[-1]
        if "A" in result:
            return -1.0
        elif "B" in result:
            return 1.0
        else:
            logger.warning(f"Invalid output preference model output: '{str_output}'")
            return 0.0
