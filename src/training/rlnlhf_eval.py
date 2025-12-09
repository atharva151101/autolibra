import pandas as pd
import tinker

service_client = tinker.ServiceClient()
from tinker import types
from tinker_cookbook import renderers
from tinker_cookbook.model_info import get_recommended_renderer_name
from tinker_cookbook.tokenizer_utils import get_tokenizer
import json
import asyncio
from pathlib import Path 
from tqdm import tqdm
import re
from datasets import load_dataset, Dataset
from typing import cast
class TinkerSampler():
    """A simple wrapper around Tinker ServiceClient to do sampling."""
    def __init__(
        self,
        model_name: str,
        model_path: str | None = None,  # tinker://..., obtained from Tinker training job
        temperature: float = 0.9,
        max_tokens=1024,
        top_p=1,
        top_k=-1,  # -1 means no limit
    ):
        tokenizer = get_tokenizer(model_name) if model_name!="meta-llama/Llama-3.1-8B-Instruct" else get_tokenizer('thinkingmachineslabinc/meta-llama-3-tokenizer')
        renderer_name = get_recommended_renderer_name(model_name)
        # Read https://tinker-docs.thinkingmachines.ai/rendering to understand what renderer is
        self.renderer = renderers.get_renderer(name=renderer_name, tokenizer=tokenizer)
        self.sampling_params = types.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop=self.renderer.get_stop_sequences(),
        )
        self.sampling_client = service_client.create_sampling_client(
            model_path=model_path,
            base_model=model_name,
        )
        
    async def generate(self, messages: list[renderers.Message]) -> renderers.Message:
        # TODO: add your code here (10 points
        prompt = self.renderer.build_generation_prompt(messages)
        output = self.sampling_client.sample(prompt, sampling_params=self.sampling_params, num_samples=1).result()
        sampled_message, parse_success = self.renderer.parse_response(output.sequences[0].tokens)
        if not parse_success:
            raise Exception("Failed to reder response.")
        return sampled_message


async def main() -> None:


    dataset_path = f'../RLHF_NLFeedback/data/development.jsonl'

    #ds = cast(Dataset, load_dataset("json", data_files=dataset_path, split="train"))
    df = pd.read_json(dataset_path, lines=True)

    print(df.head())

    grader_sampler = TinkerSampler(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    model_path="tinker://f434b567-4738-52c4-8d62-ae33590ffaa3:train:0/sampler_weights/final",#"tinker://8e6ea5b5-2ba3-5bfb-be6d-e60f2431b4d8:train:0/sampler_weights/000020", #tinker://bf9be244-84a5-55b7-b9e2-6f794e7c57d6:train:0/sampler_weights/final",
    temperature=1.0,
    max_tokens=2048,
     )


    convo_prefix = [{"role": "system", "content": "You are an expert summarization assistant. Write a summary based on the given context."},]



    count = 0
    tasks = []
    ground_truths = []

    total = 50
    pbar = tqdm(
            total=total,
            bar_format="{l_bar}{bar} {n}/{total} Rows processed [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        )
    df = df[:total] 
    
    async def process_row(messages, grader_sampler):
        result = await grader_sampler.generate(messages)
        pbar.update(1)
        return result

    for idx, row in df.sample(n=total).iterrows():
        count += 1
        convo = convo_prefix + [
            {"role": "user", "content": row['summary_prompt'] + "Provide an accurate concise summary for the above context. \n\n Summary: "},
        ]

        task = process_row(convo, grader_sampler)
        tasks.append(task)
        if count>total:
            break
    
    results = await asyncio.gather(*tasks)

    for i, df_idx in enumerate(df.index):
        df.at[df_idx, "generated"] = results[i]["content"]
        print(df.at[df_idx, "generated"], "\n\n")

    # Keep only the requested columns (as specified)
    df = df[["summary_prompt", "ideal_human_summary", "generated"]]

    # Write out as jsonl (one record per line)
    df.to_json("rlhf_nlfeedback_generated.jsonl", orient="records", lines=True)
    pbar.close()

if __name__ == "__main__":
    import argparse

    asyncio.run(main())

