
import logging
from dataclasses import dataclass
from typing import cast
import re
import asyncio
import torch
import chz
from tinker import SamplingClient, types
from tinker_cookbook import renderers
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.tokenizer_utils import Tokenizer
from tinker_cookbook.rl.types import EnvGroupBuilder, RLDataset, RLDatasetBuilder
from datasets import Dataset, load_dataset
from math import ceil
from typing import Literal, Sequence, Callable
from functools import partial
import tinker
from tinker_cookbook.completers import StopCondition
from tinker_cookbook.rl.types import (
    Action,
    Env,
    EnvGroupBuilder,
    Metrics,
    Observation,
    StepResult,
    Trajectory,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer
import asyncio

import chz
import sys
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.rl import train
import json
from pathlib import Path
from tinker.types.sample_response import SampleResponse
from concurrent.futures import Future
from retrying import retry

service_client = tinker.ServiceClient()


logger = logging.getLogger(__name__)


class RewardModel():
    def __init__(
        self,
        renderer: renderers.Renderer,
        sampling_params: types.SamplingParams,
        sampling_client: SamplingClient,
        rubric: str,
    ):
        self.renderer = renderer
        self.sampling_params = sampling_params
        self.sampling_client = sampling_client
        self.rubric = rubric

    def get_reward_model_prompt(self, summary_prompt, model_response) -> types.ModelInput:

        prompt_template = """\
You are a fair judge assistant assigned to deliver insightful feedback model generated summary for a summarization task. You use the metrics provided to evaluate the summary and provide a score out of 10 based on those metrics.

###Task Description:
An instruction which is a summarization task, a generated summary to evaluate, and some evaluation metrics are given. Each metric has a description and corresponding good and bad behaviours.
1. Write a detailed feedback that assess the quality of the given summary strictly based on the given evaluation metrics, not evaluating in general.
2. Give a score for the summary based on each metric and the weightage associated with that metric and give a reasoning for the same. Also provide a final combined score at the end out of 10.
4. The output format should look as follows: "Feedback: (write a feedback for criteria) Final Score: <score>/10"
5. Please do not generate any other opening, closing, and explanations.
6. Please ensure that the output format in point 4 is strictly followed. Example output - "Feedback: <your feedback> Final Score: 7.2/10"

###The instruction to evaluate:
{instruction}

###Summary to evaluate:
{summary}

###Evaluation Metrics:
{rubric}

###Feedback:
"""
        return [
            {"role": "system", "content": "You are a reward model that scores the quality of summaries based on the provided evaluation metrics."},
            {"role": "user", "content": prompt_template.format(
            instruction=summary_prompt,
            summary=model_response,
            rubric=self.rubric)}]

    def get_prompt_template_without_metrics(self, summary_prompt: str, model_response: str) -> str:
        prompt_template = """\
        You are a fair judge assistant assigned to deliver insightful feedback model generated summary for a summarization task. You provide a score out of 10 on the quality of summary.

###Task Description:
An instruction which is a summarization task, a generated summary to evaluate.
1. The output format should look as follows: "Feedback: (write a feedback for criteria) Final Score: <score>/10"
2. Please do not generate any other opening, closing, and explanations.
3. Please ensure that the output format in point 1 is strictly followed. Example output - "Feedback: <your feedback> Final Score: 7/10"

###The instruction to evaluate:
{instruction}

###Summary to evaluate:
{summary}

###Feedback:
"""
        return [
            {"role": "system", "content": "You are a reward model that scores the quality of summaries."},
            {"role": "user", "content": prompt_template.format(
            instruction=summary_prompt,
            summary=model_response,
            rubric=self.rubric)}]

    async def score2(self, summary_prompt, model_response) -> Future[SampleResponse]:
        prompt_messages = self.get_reward_model_prompt(summary_prompt, model_response)
        return self.sampling_client.sample(prompt_messages, sampling_params=self.sampling_params, num_samples=1)

    @retry(stop_max_attempt_number=5, wait_fixed=2000)
    async def generate(self, messages: list[renderers.Message]) -> renderers.Message:
        
        print("Generating reward model response...")
        prompt = self.renderer.build_generation_prompt(messages)
        output = self.sampling_client.sample(prompt, sampling_params=self.sampling_params, num_samples=1).result()
        sampled_message, parse_success = self.renderer.parse_response(output.sequences[0].tokens)
        print("Generated reward model response.")
        if not parse_success:
            raise Exception("Failed to reder response.")
        return sampled_message
    
    async def score(self, summary_prompt, model_response, with_metrics = True) -> float:
        if with_metrics:
            prompt_messages = self.get_reward_model_prompt(summary_prompt, model_response)
        else:
            prompt_messages = self.get_prompt_template_without_metrics(summary_prompt, model_response)
        response_message = await self.generate(prompt_messages)
        #print("Reward model response message:", response_message)
        # Extract score from response_message
        match = re.search(r'Final Score:\s*(\d+(\.\d+)?)\/10', response_message['content'])
        if match:
            score = float(match.group(1))
            return score/10.0  # Normalize to [0, 1]
        else:
            logger.warning("Failed to extract score from reward model response.")
            return 0.0
        


def get_reward_model(model_name: str) -> RewardModel:

    tokenizer = get_tokenizer(model_name) if model_name!="meta-llama/Llama-3.1-8B-Instruct" else get_tokenizer('thinkingmachineslabinc/meta-llama-3-tokenizer')
    reward_model_renderer_name = model_info.get_recommended_renderer_name(model_name)
    reward_model_renderer = renderers.get_renderer(name=reward_model_renderer_name, tokenizer=tokenizer)
    sampling_params = types.SamplingParams(
            max_tokens=1024,
            temperature=0.9,
            top_p=1,
            top_k=-1,
            stop=reward_model_renderer.get_stop_sequences(),
    )
    sampling_client = service_client.create_sampling_client(
            model_path=None,
            base_model=model_name,
        )
    rubric = """
Use the following metrics to evaluate the quality of a summary. Each metric has a name, description, weightage and associated good and bad behaviours -
"""

    metrics_path = Path('/Users/atharvachougule/CS329X/autolibra/.data/metrics/slf5k_summarization/11_02_17_43/metrics/')

    for i, metric_file in enumerate(metrics_path.glob('*.json')):
        with open(metric_file, 'r') as f:
            metric_data = json.load(f)
            
            # Format metric information
            rubric += f"""
({i+1}) Metric Name: {metric_data['name']}
Description: {metric_data['explanation']}
Weightage: {metric_data['weightage']}
Good Behaviors:
{chr(10).join('- ' + behavior for behavior in metric_data.get('good_behaviors', []))}
Bad Behaviors:
{chr(10).join('- ' + behavior for behavior in metric_data.get('bad_behaviors', []))}

"""
    reward_model = RewardModel(
        renderer=reward_model_renderer,
        sampling_params=sampling_params,
        sampling_client=sampling_client,
        rubric=rubric,
    )
    return reward_model

class SummarizationEnv(Env):
    def __init__(
        self,
        summary_prompt: str,
        renderer:  renderers.Renderer,
        reward_model: RewardModel,
        convo_prefix: list[renderers.Message] | None = None,
    ):
        self.summary_prompt = summary_prompt
        self.renderer = renderer
        self.reward_model = reward_model
        self.convo_prefix = convo_prefix or []

    @property
    def stop_condition(self) -> StopCondition:
        return self.renderer.get_stop_sequences()
    
    @classmethod
    def _suffix(cls) -> str:
        return "Provide an accurate summary based on the above reddit post. \n\n Summary: "

    def get_question(self) -> str:
        return self.summary_prompt + self._suffix()

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        convo = self.convo_prefix + [
            {"role": "user", "content": self.get_question()},
        ]
        return self.renderer.build_generation_prompt(convo), self.stop_condition
    

    async def step(self, action: Action) -> StepResult:
        #message, parse_success = self.renderer.parse_response(action)
        total_reward = 0.0
        # if not parse_success:
        #     total_reward = 0.0
        # else:
        #     response = message["content"]
        #     total_reward = await self.reward_model.score(self.summary_prompt, response)
        
        return StepResult(
            reward=total_reward,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics={},
        )


@dataclass(frozen=True)
class SummarizationGroupBuilder(EnvGroupBuilder):
    renderer: renderers.Renderer
    summary_prompt: str
    reward_model: RewardModel
    convo_prefix: list[renderers.Message] | None 
    suffix: str
    env_thunk: Callable[[], SummarizationEnv]
    num_envs: int
    dataset_name: str = "summarization"

    async def make_envs(self) -> Sequence[Env]:
        return [self.env_thunk() for _ in range(self.num_envs)]

    async def compute_group_rewards(
        self, trajectory_group: list[Trajectory]
    ) -> list[tuple[float, Metrics]]:
        tasks = []
        async def zero_reward():
            return 0.0
        for trajectory in trajectory_group:
            message, parse_success = self.renderer.parse_response(trajectory.transitions[0].ac.tokens)
            if not parse_success:
                tasks.append(zero_reward())
            else:
                response = message["content"]
                tasks.append(self.reward_model.score(self.summary_prompt, response))
        scores = await asyncio.gather(*tasks)
        final_scores = [score for score in scores]
        return [(final_score, {}) for final_score in final_scores]

    def logging_tags(self) -> list[str]:
        return [self.dataset_name]



class SummarizationDataset(RLDataset):
    def __init__(
        self,
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        reward_model: RewardModel,
        convo_prefix: list[renderers.Message] | None = None,
        split: Literal["train", "test"] = "train",
        seed: int = 0,
    ):
        if split not in ("train", "test"):
            raise ValueError("split must be 'train' or 'test'")
        if split == "test":
            return None
        self.ds = cast(Dataset, load_dataset("json", data_files=".data/raw/slf5k_dataset/slf5k_data.jsonl", split="train"))
        self.ds = self.ds.shuffle(seed=seed)
        self.batch_size = batch_size
        self.group_size = group_size if split == "train" else 1
        self.renderer = renderer
        self.convo_prefix = convo_prefix
        self.reward_model = reward_model

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        batch_start = index * self.batch_size
        batch_end = min(batch_start + self.batch_size // 8, len(self.ds))
        assert batch_start < batch_end, "Incorrect batch size"
        return [
            builder
            for row in self.ds.select(range(batch_start, batch_end))
            if (builder := self._make_env_group_builder(row, self.group_size)) is not None  # pyright: ignore[reportArgumentType]
        ]
    def __len__(self) -> int:
        return ceil(len(self.ds) / self.batch_size)
    def _make_env_group_builder(
        self, x: dict[str, str], group_size: int
    ) ->   None:
        try:
            summary_prompt = x['summary_prompt']
        except Exception as e:
            logger.warning(f"Failed to parse SLF5K row: {e}")
            return None
        return SummarizationGroupBuilder(
            renderer = self.renderer,
            summary_prompt=summary_prompt,
            reward_model=self.reward_model,
            convo_prefix=self.convo_prefix,
            suffix="Provide an accurate summary based on the above reddit post. \n\n Summary: ",
            env_thunk=partial(
                SummarizationEnv,
                summary_prompt=summary_prompt,
                renderer=self.renderer,
                reward_model=self.reward_model,
                convo_prefix=self.convo_prefix,
            ),
            num_envs=group_size,
        )


@chz.chz 
class SummarizationDatasetBuilder(RLDatasetBuilder):
    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int
    seed: int = 0
    reward_model_name: str

    async def __call__(self) -> tuple[SummarizationDataset, SummarizationDataset]:
        
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer  = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)
        datasets = [
            SummarizationDataset(
                batch_size=self.batch_size,
                group_size=self.group_size,
                renderer=renderer,
                split=split,
                seed=self.seed,
                reward_model=get_reward_model(self.reward_model_name),
                convo_prefix=[
                    {"role": "system", "content": "You are an expert summarization assistant. Write a summary based on the given context."},]
            )
            for split in ("train", "test")
        ]
        return (datasets[0], None)




def build_config_blueprint() -> chz.Blueprint[train.Config]:
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    reward_model_name = "meta-llama/Llama-3.3-70B-Instruct"

    renderer_name = model_info.get_recommended_renderer_name(model_name)
    builder = SummarizationDatasetBuilder(
        batch_size=16,
        group_size=4,
        renderer_name=renderer_name,
        model_name_for_tokenizer=model_name,
        reward_model_name=reward_model_name,
    )

    return chz.Blueprint(train.Config).apply(
        {
            "model_name": model_name,
            "log_path": "logs_ppo",
            "dataset_builder": builder,
            "learning_rate": 4e-5,
            "max_tokens": 256,
            "eval_every": 0,
            "loss_fn": "ppo",
            "wandb_project": "tinker_personalization",
            "wandb_name": "summarization_rlnlhf",
        }
    )
def main(config: train.Config):
    # Avoid clobbering log dir from your previous run:
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="ask")
    asyncio.run(train.main(config))


if __name__ == "__main__":
    blueprint = build_config_blueprint()
    print(sys.argv[1:])
    blueprint.make_from_argv(sys.argv[1:])
    main(blueprint.make())

# if __name__ == "__main__":
#     Dataset = SummarizationDataset(batch_size=4, group_size=2, renderer=None)  # type: ignore
#     print(len(Dataset))
#     batch = Dataset.get_batch(0)

#     print(asyncio.run(batch[0].make_envs()))
#     print(batch)

    
