from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import requests, json
from osw_data import MultiAgentDataset, AgentMetadata, PointType, MediaType
from .base import BaseConverter, run_converter
from osw_data.annotation import AnnotationSystem, AnnotationSpan 


class SLF5KConverter(BaseConverter):
    """Handles converting SLF5K summarization data to our dataset format"""

    def __init__(self, output_path: Path, source_path: Path, annotation_path: Path | None = None) -> None:
        super().__init__(output_path, source_path)
        self.dataset_url = "https://huggingface.co/datasets/JeremyAlain/SLF5K/resolve/main/train.jsonl"
        self.annotation_path = annotation_path

    def download_data(self) -> None:
        """Download SLF5K dataset files from Hugging Face"""
        self.logger.info("Downloading SLF5K dataset from Hugging Face...")
        self.source_path.mkdir(parents=True, exist_ok=True)
        
        # Download the JSONL file
        response = requests.get(self.dataset_url)
        if response.status_code == 200:
            jsonl_path = self.source_path / "slf5k_data.jsonl"
            with open(jsonl_path, "wb") as f:
                f.write(response.content)
            self.logger.info(f"Dataset downloaded to {jsonl_path}")
        else:
            raise Exception(f"Failed to download dataset: {response.status_code}")
        

    def convert_to_dataset(self) -> None:
        """Convert SLF5K data to autolibra dataset format"""
        self.logger.info("Creating SLF5K dataset...")

        ref_time = datetime.now()

        # Initialize dataset and annotation system
        dataset = MultiAgentDataset(
            name="SLF5K-Summarization",
            base_path=self.output_path,
            description="Summarization trajectories from SLF5K dataset with human feedback",
        )

        annotation_system = AnnotationSystem(
            base_path=self.annotation_path,
            project_name="SLF5K-Summarization", 
            description="Human feedback annotations for SLF5K summarization dataset",
        )

        annotation_system.add_annotator(
            annotator_id="original_annotator",
            name="Original SLF5K Annotator",
            role="human_annotator",
            expertise_level="expert",
            metadata={"source": "SLF5K Dataset"}
        )

        # Read the JSONL file
        df = pd.read_json(self.source_path / "slf5k_data.jsonl", lines=True)

        for idx, row in df.iterrows():
            # Create agent metadata
            agents_metadata = {
                "summarizer": AgentMetadata(
                    agent_id="summarizer",
                    agent_type="language_model",
                    capabilities=["text_generation", "summarization"],
                )
            }

            # Create instance metadata (without feedback)
            instance_metadata = {
                "task": "text_summarization",
                "prompt": row["summary_prompt"],
            }

            # Create new instance
            instance_id = dataset.create_instance(
                agents_metadata=agents_metadata,
                instance_metadata=instance_metadata
            )

            self.logger.info(f"Created instance {instance_id}")

            # Add observation point (input text)
            dataset.add_data_point(
                instance_id=instance_id,
                agent_id="summarizer",
                timestamp=ref_time + timedelta(seconds=idx*2),
                point_type=PointType.OBSERVATION,
                data={"text": row["summary_prompt"]},
                media_type=MediaType.JSON,
            )

            # Add action point (generated summary without feedback)
            dataset.add_data_point(
                instance_id=instance_id,
                agent_id="summarizer",
                timestamp=ref_time + timedelta(seconds=idx*2 + 1),
                point_type=PointType.ACTION,
                data={"text": row["generated_summary_for_feedback"]},
                media_type=MediaType.JSON,
            )

            # Add feedback as annotation
            annotation_system.add_annotation(
                instance_id=instance_id,
                agent_id="summarizer",
                annotator_id="original_annotator",
                content={
                    "feedback": row["feedback"],
                    "timestamp": ref_time + timedelta(seconds=idx*2 + 1)
                },
            )

        self.logger.info("Dataset conversion complete")
        dataset.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SLF5K Summarization Converter")
    parser.add_argument(
        "--filename",
        type=str,
        required=True,
        help="Path to the SLF5K summarization CSV file",
    )

    filename = parser.parse_args().filename

    source_path = Path(f".data/raw/{filename}")
    output_path = Path(".data/slf5k_summarization")
    annotation_path = Path(".data/annotations/slf5k_summarization")

    run_converter(SLF5KConverter, output_path, source_path, annotation_path=annotation_path)