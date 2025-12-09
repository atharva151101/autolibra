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
from ppo import RewardModel, get_reward_model



async def main():
    dataset_path = f'../RLHF_NLFeedback/data/development.jsonl'

    #ds = cast(Dataset, load_dataset("json", data_files=dataset_path, split="train"))
    df = pd.read_json(dataset_path, lines=True)

    rm = get_reward_model("meta-llama/Llama-3.1-8B-Instruct")
    count = 0

    total = 200
    tasks2 = []
    
    pbar = tqdm(
            total=total,
            bar_format="{l_bar}{bar} {n}/{total} Rows processed [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        )
    async def process_row(row, grader_sampler: RewardModel, with_metrics=True) -> float:
        score1 = await grader_sampler.score(row['summary_prompt'], row['generated_summary_for_comparison_A'], with_metrics)
        score2 = await grader_sampler.score(row['summary_prompt'], row['generated_summary_for_comparison_B'], with_metrics)
        answer = 'Summary A' if score1 >= score2 else 'Summary B'
        pbar.update(1)
        if answer != row['comparison_preference']:
            print(f"Prompt: {row['summary_prompt']}")
            print(f"Generated A: {row['generated_summary_for_comparison_A']}")
            print(f"Generated B: {row['generated_summary_for_comparison_B']}")
            print(f"Scores: A={score1}, B={score2}, Chose: {answer}, Ground Truth: {row['comparison_preference']}\n\n")
        return 1 if answer == row['comparison_preference'] else 0, abs(score1 - score2)
    

    for idx, row in df.iterrows():
        count += 1
        tasks2.append(process_row(row, rm, with_metrics=True))
        if count >= total:
            break
    
    scores = await asyncio.gather(*tasks2)
    scores2 = [s[0] for s in scores]
    score_diffs = [s[1] for s in scores]
    diff_total = 0.0
    for i, diff in enumerate(score_diffs):
        if scores2[i]==0:
            diff_total += diff
    print(f"Average score difference for incorrect cases: {diff_total/(total - sum(scores2)) if (total - sum(scores2))>0 else 0.0:.2f}")
    pbar.close()

    print(f"Accuracy: {sum(scores2)}/{total} = {sum(scores2)/total:.2f}")

if __name__ == "__main__":
    asyncio.run(main())