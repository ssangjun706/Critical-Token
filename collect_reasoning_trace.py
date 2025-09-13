import os
import json
import random
import wandb
import logging
import argparse

from datetime import datetime
from omegaconf import OmegaConf
from typing import List, Dict, Optional
from vllm import LLM, SamplingParams
from utils import is_correct, majority_vote, set_random_seeds
from parser import extract_latex_answer


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
        num_rollouts: int = 32,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        use_wandb: bool = False,
        gpu_memory_utilization: float = 0.7,
        project_name: Optional[str] = None,
        wandb_api_key: Optional[str] = None,
        seed: int = 42,
        resume: bool = True,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_rollouts = num_rollouts
        self.output_dir = output_dir
        self.use_wandb = use_wandb
        self.resume = resume
        self.data = self.load_dataset(data_path)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.checkpoint_file = os.path.join(output_dir, f"checkpoint_{timestamp}.json")
        self.output_file = os.path.join(output_dir, f"reasoning_trace_{timestamp}.json")

        if self.use_wandb:
            if project_name is None or wandb_api_key is None:
                raise ValueError(
                    "Project name and wandb api key are required when use_wandb is True"
                )

            wandb.login(key=wandb_api_key)
            wandb.init(project=project_name)

        self.llm = LLM(
            model=model_name,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype="bfloat16",
        )

        self.sampling_params = SamplingParams(
            n=self.num_rollouts,
            temperature=temperature,
            max_tokens=max_new_tokens,
        )

        set_random_seeds(seed=seed)

    def load_dataset(self, path: str) -> List[Dict]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not (isinstance(data, list) and isinstance(data[0], dict)):
            raise ValueError("Data must be a list of dictionaries")
        else:
            return data

    def load_checkpoint(self) -> Dict:
        if not os.path.exists(self.checkpoint_file):
            return {
                "last_processed_index": 0,
                "collected_traces": [],
                "batch_count": 0,
                "completed": False
            }
        
        try:
            with open(self.checkpoint_file, "r", encoding="utf-8") as f:
                checkpoint = json.load(f)
            logger.info(f"Checkpoint loaded: {checkpoint['last_processed_index']}/{len(self.data)} problems processed")
            return checkpoint
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}, starting from scratch")
            return {
                "last_processed_index": 0,
                "collected_traces": [],
                "batch_count": 0,
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
        all_prompts = [problem["problem"] for problem in problems_batch]
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
                if predicted_answer == "":
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
                    **problem_data,
                    "solutions": solutions,
                    "correctness_rate": correctness_rate,
                }
            )

        return results

    def collect_reasoning_trace(self) -> List[Dict]:
        problem_pool = self.data
        total_batches = (len(problem_pool) + self.batch_size - 1) // self.batch_size
        
        if self.resume:
            checkpoint = self.load_checkpoint()
            if checkpoint["completed"]:
                logger.info("All work is already completed. Returning existing results.")
                return self.load_existing_traces()
            
            problem_index = checkpoint["last_processed_index"]
            batch_count = checkpoint["batch_count"]
            collected_traces = self.load_existing_traces()
            
            if problem_index > 0:
                logger.info(f"Restarting from previous state: {problem_index}/{len(problem_pool)} problems processed")
        else:
            collected_traces = []
            problem_index = 0
            batch_count = 0

        while problem_index < len(problem_pool):
            end_index = min(problem_index + self.batch_size, len(problem_pool))
            current_batch = problem_pool[problem_index:end_index]

            if not current_batch:
                break

            batch_results = self.solve_problems_batch(current_batch)
            batch_count += 1
            skipped_count = 0

            for result in batch_results:
                current_traces = []
                correctness_rate = result["correctness_rate"]

                if correctness_rate <= 0.01 or correctness_rate >= 0.99:
                    skipped_count += 1
                    continue

                correct_traces = random.choices(
                    [sol for sol in result["solutions"] if sol["is_correct"]], k=2
                )

                for correct_trace in correct_traces:
                    correct_sample_data = {
                        "problem": result["problem"],
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
                    if trace["predicted_answer"] in majority_answer
                ]
                wrong_traces = random.choices(majority_traces, k=2)

                for wrong_trace in wrong_traces:
                    wrong_sample_data = {
                        "problem": result["problem"],
                        "true_answer": result["true_answer"],
                        "predicted_answer": wrong_trace["predicted_answer"],
                        "solution": wrong_trace["solution"],
                        "is_correct": wrong_trace["is_correct"],
                    }
                    current_traces.append(wrong_sample_data)

                collected_traces.extend(current_traces)
                self.save_traces(collected_traces)

            checkpoint_data = {
                "last_processed_index": end_index,
                "batch_count": batch_count,
                "total_problems": len(problem_pool),
                "total_batches": total_batches,
                "collected_traces_count": len(collected_traces),
                "completed": end_index >= len(problem_pool),
            }
            self.save_checkpoint(checkpoint_data)

            progress_percent = (batch_count / total_batches) * 100
            logger.info(
                f"Progress: {batch_count}/{total_batches} ({progress_percent:.2f}%)"
            )
            logger.info(f"Collected Traces: {len(current_traces)}")
            logger.info(f"Skipped Traces: {skipped_count}")
            logger.info(f"Total Collected: {len(collected_traces)}")

            if self.use_wandb:
                wandb.log(
                    {
                        "batch": batch_count,
                        "total_batches": total_batches,
                        "progress_percent": progress_percent,
                        "collected_traces": len(collected_traces),
                    }
                )

            problem_index = end_index

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
        data_path=config["data_path"],
        output_dir=config["output_dir"],
        use_wandb=config["use_wandb"],
        gpu_memory_utilization=config["gpu_memory_utilization"],
        project_name=config["project_name"],
        wandb_api_key=config["wandb_api_key"],
        seed=config["seed"],
        resume=config["resume"],
    )

    try:
        traces = collector.collect_reasoning_trace()
        logger.info(f"Collection completed: Total {len(traces)} traces collected.")
    except (Exception, KeyboardInterrupt) as e:
        logger.error(f"Error collecting reasoning trace: {e}")
    finally:    
        collector.shutdown()
