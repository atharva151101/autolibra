import pandas as pd
from tinker_preference import PrometheusEvalComparisonRendererFromChatRenderer, PrometheusEvalComparison
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
        tokenizer = get_tokenizer(model_name)
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


async def main2(dataset_name: str) -> None:

    use_metrics = False

    dataset_path = f'.data/raw/{dataset_name}/slf5k_data.jsonl'

    df = pd.read_json(dataset_path, lines=True)

    print(df.head())

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

    print("Generated Rubric:")
    print(rubric)

    grader_sampler = TinkerSampler(
    model_name="meta-llama/Llama-3.3-70B-Instruct",
    temperature=1.0,
    max_tokens=2048,
     )
    pairwise_renderer = PrometheusEvalComparisonRendererFromChatRenderer(convo_renderer=grader_sampler.renderer)

    count = 0
    tasks = []
    ground_truths = []

    total = 100
    pbar = tqdm(
            total=total,
            bar_format="{l_bar}{bar} {n}/{total} Rows processed [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        )
    
    
    async def process_row(messages, grader_sampler):
        result = await grader_sampler.generate(messages)
        pbar.update(1)
        return result

    for idx, row in df.sample(total, random_state=42).iterrows():
        prompt = row["summary_prompt"]
        respons_a = row["generated_summary_for_comparison_A"]
        respons_b = row["generated_summary_for_comparison_B"]
        preference = row["comparison_preference"]
        comparison = PrometheusEvalComparison(
            prompt_conversation=[renderers.Message(role="user", content=prompt)],
            completion_A=[renderers.Message(role="assistant", content=respons_a)],
            completion_B=[renderers.Message(role="assistant", content=respons_b)],
            rubric=rubric,
            reference=None,
        )
        messages = pairwise_renderer._comparison_to_convo(comparison, use_metrics=use_metrics)
        #print("Messages: ", messages)
        tasks.append(process_row(messages, grader_sampler))
        ground_truths.append(preference)
        count += 1
        if count >= total:
            break
    
        
    results = await asyncio.gather(*tasks)
    pbar.close()
    correct = 0
    unkown = 0
    sum = 0
    score_pattern = r"Final Score:.*?A\s+(\d+\.?\d*)/10.*?B\s+(\d+\.?\d*)/10"
    for i, result in enumerate(results):
        print(f"Result {i+1}: {result}")
        predicted_preference = "A" if "[RESULT] A" in result["content"] or "RESULT: A" in result["content"] else "B" if "[RESULT] B" in result["content"] or "RESULT: B" in result["content"] else "Unknown"
        ground_truth = "A" if "Summary A" in ground_truths[i] else "B" if "Summary B" in ground_truths[i] else "Unknown"
        print(f"Predicted Preference: {predicted_preference}, Ground Truth: {ground_truth}")
        if predicted_preference == ground_truth and predicted_preference != "Unknown":
            correct += 1
        if predicted_preference == "Unknown" or ground_truth == "Unknown":
            unkown += 1
        if use_metrics and predicted_preference != ground_truth and predicted_preference != "Unknown" and ground_truth != "Unknown":
            if match := re.search(score_pattern, result["content"]):
                score_a = float(match.group(1))
                score_b = float(match.group(2))
                
                sum += abs(score_a - score_b)
                

    print(f"Accuracy: {correct}/{len(results)} = {correct/(len(results)):.2f}, Unknowns: {unkown}")
    if use_metrics:
        print(f"Average score difference for incorrect predictions: {sum/(len(results)-correct-unkown) if (len(results)-correct-unkown)>0 else 0:.2f}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Balrog Converter")
    parser.add_argument(
        "--filename",
        type=str,
        required=True,
        help="The name of the folder containing the data for the given run",
    )

    filename = parser.parse_args().filename

    asyncio.run(main2(filename))

