import os
import json
import random
import wandb
import logging
import argparse

from omegaconf import OmegaConf
from typing import List, Dict, Optional
from vllm import LLM, SamplingParams
from utils import is_correct, majority_vote
from parser import extract_latex_answer
from datasets import load_dataset


logging.getLogger("vllm").setLevel(logging.CRITICAL)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class TrajectoryCollector:
    def __init__(
        self,
        model_name: str,
        data_path: str,
        output_dir: str,
        batch_size: int,
        num_rollouts: int,
        max_new_tokens: int,
        use_wandb: bool,
        max_samples: Optional[int] = None,
        project_name: Optional[str] = None,
        wandb_api_key: Optional[str] = None,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_rollouts = num_rollouts
        self.max_samples = max_samples
        self.output_dir = output_dir
        self.use_wandb = use_wandb
        self.data = self.load_dataset(data_path, max_samples)
        
        self.checkpoint_file = os.path.join(output_dir, f"reasoning_trace_checkpoint.json")
        self.output_file = os.path.join(output_dir, f"reasoning_trace_results.json")

        if self.use_wandb:
            if project_name is None or wandb_api_key is None:
                raise ValueError(
                    "Project name and wandb api key are required when use_wandb is True"
                )

            wandb.login(key=wandb_api_key)
            wandb.init(project=project_name)

        self.llm = LLM(
            model=model_name,
            max_model_len=2048,
            gpu_memory_utilization=0.8,
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.sampling_params = SamplingParams(
            n=self.num_rollouts,
            max_tokens=max_new_tokens,
        )


    def chat_template(self, problem: str) -> str:
        return problem
        # messages = [
        #     {"role": "user", "content": "Please reason step by step, and put your final answer within \\boxed{}.\n\n" + problem},
        # ]

        # prompt = self.tokenizer.apply_chat_template(
        #     messages, 
        #     tokenize=False, 
        #     add_generation_prompt=True,
        # )

        # return prompt


    def load_dataset(self, path: str, max_samples: Optional[int] = None) -> List[Dict]:
        dataset = load_dataset(path, name="en", split="train")

        if max_samples is not None:
            dataset = dataset.select(range(max_samples))

        dataset = [{"prompt": data["prompt"], "true_answer": data["solution"]} for data in dataset]
        return dataset


    def load_checkpoint(self) -> Dict:
        if not os.path.exists(self.checkpoint_file):
            return {
                "next_index": 0,
                "completed": False
            }
        
        try:
            with open(self.checkpoint_file, "r", encoding="utf-8") as f:
                checkpoint = json.load(f)
            logger.info(f"Checkpoint loaded: {checkpoint['next_index']}/{len(self.data)} problems processed")
            return checkpoint
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}, starting from scratch")
            return {
                "next_index": 0,
                "completed": False
            }


    def save_checkpoint(self, checkpoint_data: Dict):
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            with open(self.checkpoint_file, "w", encoding="utf-8") as f:
                json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")


    def load_existing_traces(self) -> List[Dict]:
        if not os.path.exists(self.output_file):
            return []
        
        try:
            with open(self.output_file, "r", encoding="utf-8") as f:
                traces = json.load(f)
            logger.info(f"Loaded {len(traces)} traces from existing result file")
            return traces
        except Exception as e:
            logger.warning(f"Failed to load traces from existing result file: {e}")
            return []


    def solve_problems_batch(
        self,
        problems_batch: List[Dict],
    ) -> List[Dict]:
        all_prompts = [self.chat_template(problem["prompt"]) for problem in problems_batch]
        outputs = self.llm.generate(
            all_prompts,
            self.sampling_params,
            use_tqdm=False,
        )

        results = []
        for i, problem_data in enumerate(problems_batch):
            ground_truth = problem_data["true_answer"]
            solutions = []
            correct_count = 0
            problem_output = outputs[i]

            for output_choice in problem_output.outputs:
                solution_text = output_choice.text.strip()

                predicted_answer = extract_latex_answer(solution_text)
                if predicted_answer is None:
                    continue

                valid = is_correct(predicted_answer, ground_truth)

                if valid:
                    correct_count += 1

                solutions.append(
                    {
                        "solution": solution_text,
                        "predicted_answer": predicted_answer,
                        "is_correct": valid,
                    }
                )

            correctness_rate = correct_count / self.num_rollouts
            results.append(
                {
                    "problem": problem_data["prompt"],
                    "true_answer": problem_data["true_answer"],
                    "solutions": solutions,
                    "correctness_rate": correctness_rate,
                }
            )

        return results

    def collect(self) -> List[Dict]:
        checkpoint = self.load_checkpoint()
        if checkpoint["completed"]:
            logger.info("All work is already completed. Returning existing results.")
            return self.load_existing_traces()
            
        start_index = checkpoint["next_index"]
        collected_traces = self.load_existing_traces()
            
        if start_index > 0:
            logger.info(f"Restarting from previous state: {start_index}/{len(self.data)} problems processed")

        for idx in range(start_index, len(self.data), self.batch_size):
            end_index = min(idx + self.batch_size, len(self.data))
            current_batch = self.data[idx:end_index]
            batch_results = self.solve_problems_batch(current_batch)

            for result in batch_results:
                current_traces = []
                correctness_rate = result["correctness_rate"]

                breakpoint()
                if correctness_rate <= 0.25 or correctness_rate >= 0.75:
                    continue

                correct_trace = random.choice(
                    [sol for sol in result["solutions"] if sol["is_correct"]]
                )

                correct_sample_data = {
                    "problem": result["prompt"],
                    "true_answer": result["true_answer"],
                    "predicted_answer": correct_trace["predicted_answer"],
                    "solution": correct_trace["solution"],
                    "is_correct": correct_trace["is_correct"],
                }
                current_traces.append(correct_sample_data)

                wrong_traces = [
                    sol for sol in result["solutions"] if not sol["is_correct"]
                ]
                majority_answer = majority_vote(
                    [wrong_trace["predicted_answer"] for wrong_trace in wrong_traces]
                )
                majority_traces = [
                    trace
                    for trace in wrong_traces
                    if trace["predicted_answer"] == majority_answer
                ]
                wrong_trace = random.choice(majority_traces)
                wrong_sample_data = {
                    "problem": result["prompt"],
                    "true_answer": result["true_answer"],
                    "predicted_answer": wrong_trace["predicted_answer"],
                    "solution": wrong_trace["solution"],
                    "is_correct": wrong_trace["is_correct"],
                }
                current_traces.append(wrong_sample_data)

                collected_traces.extend(current_traces)
                self.save_traces(collected_traces)

            checkpoint_data = {
                "next_index": end_index + 1,
                "completed": (end_index + 1) >= len(self.data),
            }
            self.save_checkpoint(checkpoint_data)

            progress_percent = (end_index / len(self.data)) * 100
            logger.info(
                f"Progress: {end_index}/{len(self.data)} ({progress_percent:.2f}%)"
            )
            logger.info(f"Collected Traces: {len(current_traces)}")
            logger.info(f"Total Collected: {len(collected_traces)}")

            if self.use_wandb:
                wandb.log(
                    {
                        "progress_percent": progress_percent,
                    }
                )


        if self.use_wandb:
            wandb.finish()

        return collected_traces

    def save_traces(self, traces: List[Dict]):
        os.makedirs(self.output_dir, exist_ok=True)
        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(traces, f, ensure_ascii=False, indent=2)

    def shutdown(self):
        self.llm.stop_profile()
        if self.use_wandb:
            wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = OmegaConf.load(args.config)

    collector = TrajectoryCollector(
        model_name=config["model"],
        num_rollouts=config["num_rollouts"],
        batch_size=config["batch_size"],
        max_new_tokens=config["max_new_tokens"],
        data_path=config["data_path"],
        output_dir=config["output_dir"],
        use_wandb=config["use_wandb"],
        max_samples=config.get("max_samples", None),
        project_name=config.get("project_name", None),
        wandb_api_key=config.get("wandb_api_key", None),
    )

    try:
        traces = collector.collect()
        logger.info(f"Collection completed: Total {len(traces)} traces collected.")
    except (Exception, KeyboardInterrupt) as e:
        logger.error(f"Error collecting reasoning trace: {e}")
    finally:    
        collector.shutdown()
