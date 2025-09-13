import json
import os
import wandb
import logging
import argparse
import numpy as np

from omegaconf import OmegaConf
from typing import List, Dict, Optional
from vllm import LLM, SamplingParams
from utils import set_random_seeds, is_correct
from parser import extract_latex_answer


logging.getLogger("vllm").setLevel(logging.CRITICAL)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class CombinedCollector:
    def __init__(
        self,
        model_name: str,
        data_path: str,
        output_dir: str,
        temperature: float = 0.7,
        top_k: int = 20,
        max_logprobs: int = 32768,
        num_rollouts: int = 32,
        max_new_tokens: int = 1024,
        prompt_chunk_size: int = 256,
        use_wandb: bool = False,
        project_name: Optional[str] = None,
        wandb_api_key: Optional[str] = None,
        resume: bool = True,
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        self.use_wandb = use_wandb
        self.top_k = top_k
        self.num_rollouts = num_rollouts
        self.prompt_chunk_size = max(1, int(prompt_chunk_size))
        self.data = self.load_dataset(data_path)

        if self.use_wandb:
            if project_name is None or wandb_api_key is None:
                raise ValueError(
                    "Project name and wandb api key are required when use_wandb is True"
                )

            wandb.login(key=wandb_api_key)
            self.run = wandb.init(project=project_name)

        self.checkpoint_file = os.path.join(output_dir, f"combined_checkpoint.json")
        self.output_file = os.path.join(output_dir, f"combined_results.json")
        self.resume = resume

        self.llm = LLM(
            model=model_name,
            gpu_memory_utilization=0.95,
            dtype="bfloat16",
            max_model_len=4096,
            max_logprobs=max_logprobs,
        )

        self.tokenizer = self.llm.get_tokenizer()

        self.logps_params = SamplingParams(
            temperature=temperature,
            max_tokens=1,
            logprobs=max_logprobs,
        )

        self.rollout_params = SamplingParams(
            n=self.num_rollouts,
            temperature=temperature,
            max_tokens=max_new_tokens,
        )

        set_random_seeds(seed=42)

    def load_checkpoint(self) -> Dict:
        if not os.path.exists(self.checkpoint_file):
            return {
                "next_index": 0,
                "completed": False,
            }

        try:
            with open(self.checkpoint_file, "r", encoding="utf-8") as f:
                checkpoint = json.load(f)

            logger.info(
                f"Checkpoint loaded: {checkpoint['next_index']}/{len(self.data)} samples processed"
            )
            return checkpoint
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}, starting from scratch")
            return {
                "next_index": 0,
                "completed": False,
            }

    def save_checkpoint(self, checkpoint_data: Dict):
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            with open(self.checkpoint_file, "w", encoding="utf-8") as f:
                json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def load_existing_results(self) -> List[Dict]:
        if not os.path.exists(self.output_file):
            return []

        try:
            with open(self.output_file, "r", encoding="utf-8") as f:
                results = json.load(f)
            logger.info(f"Loaded {len(results)} results from existing output file")
            return results
        except Exception as e:
            logger.warning(f"Failed to load results from existing output file: {e}")
            return []

    def chat_template(self, problem: str, partial_solution: str) -> str:
        messages = [
            {"role": "user", "content": problem},
        ]

        if len(partial_solution) > 0:
            messages.append({"role": "assistant", "content": partial_solution})

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        return prompt

    def load_dataset(self, path: str) -> List[Dict]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not (isinstance(data, list) and isinstance(data[0], dict)):
            raise ValueError("Data must be a list of dictionaries")

        return data

    def _normalize_logprobs_stable(self, logps: np.ndarray) -> np.ndarray:
        if logps.size == 0:
            return logps

        max_logp = np.max(logps)
        logsumexp = max_logp + np.log(np.sum(np.exp(logps - max_logp)))
        return logps - logsumexp

    def logps_sampling(self, prompts: List[str]) -> List[List[float]]:
        all_logprobs: List[List[float]] = []
        for start in range(0, len(prompts), self.prompt_chunk_size):
            end = start + self.prompt_chunk_size
            chunk_prompts = prompts[start:end]
            outputs = self.llm.generate(chunk_prompts, self.logps_params)
            for output in outputs:
                logps_dict = output.outputs[0].logprobs[0]
                logps = np.array([lp.logprob for _, lp in logps_dict.items()])
                normalized_logps = self._normalize_logprobs_stable(logps)
                all_logprobs.append(normalized_logps)
        return all_logprobs

    def rollout_sampling_chunked(self, prompts: List[str]) -> List[str]:
        all_completions: List[str] = []
        for start in range(0, len(prompts), self.prompt_chunk_size):
            end = start + self.prompt_chunk_size
            chunk_prompts = prompts[start:end]
            outputs = self.llm.generate(chunk_prompts, self.rollout_params)
            for output in outputs:
                for sample_output in output.outputs:
                    all_completions.append(sample_output.text.strip())
        return all_completions

    def evaluate_correctness(self, ground_truth: str, solutions: List[str]) -> float:
        correct_count = 0

        for solution in solutions:
            predicted_answer = extract_latex_answer(solution)
            if is_correct(predicted_answer, ground_truth):
                correct_count += 1

        return correct_count / max(1, self.num_rollouts)

    def build_prefix_prompts(self, problem: str, solution: str) -> List[str]:
        tokens = self.tokenizer.encode(solution)
        prompts: List[str] = []
        running_solution = ""
        for idx in range(len(tokens)):
            prompt = self.chat_template(
                problem=problem, partial_solution=running_solution
            )
            prompts.append(prompt)
            running_solution += self.tokenizer.decode(
                [tokens[idx]], skip_special_tokens=True
            )
        return prompts

    def process_single(self, trace_data: Dict) -> Dict:
        problem = trace_data["problem"]
        ground_truth = trace_data["true_answer"]
        solution = trace_data["solution"]
        sample_index = trace_data["sample_index"]
        parsed_answer = trace_data["parsed_answer"]

        prefix_prompts = self.build_prefix_prompts(problem, solution)

        entropy_scores: List[float] = []
        confidence_scores: List[float] = []
        correctness_scores: List[float] = []

        all_logprobs = self.logps_sampling(prefix_prompts)
        for logprobs in all_logprobs:
            probs = np.exp(logprobs)
            entropy = -np.sum(probs * logprobs)
            sorted_logprobs = np.sort(logprobs)[::-1]
            top_k_logprobs = sorted_logprobs[: self.top_k]
            avg_logprob = np.mean(top_k_logprobs)
            confidence = -avg_logprob
            entropy_scores.append(float(entropy))
            confidence_scores.append(float(confidence))

        all_completions = self.rollout_sampling_chunked(prefix_prompts)
        num_prefixes = len(prefix_prompts)
        for idx in range(num_prefixes):
            start_idx = idx * self.num_rollouts
            end_idx = start_idx + self.num_rollouts
            completions = all_completions[start_idx:end_idx]
            correctness_ratio = self.evaluate_correctness(ground_truth, completions)
            correctness_scores.append(correctness_ratio)

        result = {
            "sample_index": sample_index,
            "problem": problem,
            "true_answer": ground_truth,
            "solution": solution,
            "parsed_answer": parsed_answer,
        }

        result.update(
            {
                "entropy": entropy_scores,
                "confidence": confidence_scores,
            }
        )

        result.update(
            {
                "correctness": correctness_scores,
            }
        )

        return result

    def collect(self) -> List[Dict]:
        if self.resume:
            checkpoint = self.load_checkpoint()
            if checkpoint["completed"]:
                logger.info("All work is already completed. Loading existing results.")
                return self.load_existing_results()

            start_index = checkpoint.get("next_index", 0)
            results = self.load_existing_results()
            if len(results) > start_index:
                results = results[:start_index]
        else:
            results = []
            start_index = 0

        logger.info(
            f"Starting combined collection for {len(self.data)} traces (starting from {start_index})"
        )

        for idx in range(start_index, len(self.data)):
            trace_data = self.data[idx]
            logger.info(f"Processing trace {idx + 1}/{len(self.data)}")

            try:
                result = self.process_single(trace_data)
                results.append(result)

                if self.use_wandb:
                    wandb.log(
                        {
                            "overall_progress": len(results) / len(self.data),
                            "processed_examples": len(results),
                        }
                    )

                with open(self.output_file, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)

                checkpoint_data = {
                    "next_index": idx + 1,
                    "completed": (idx + 1) >= len(self.data),
                }
                self.save_checkpoint(checkpoint_data)

            except Exception as e:
                logger.error(f"Error processing trace {idx + 1}: {e}")
                checkpoint_data = {
                    "next_index": idx,
                    "completed": False,
                }
                self.save_checkpoint(checkpoint_data)
                with open(self.output_file, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                raise

        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        final_checkpoint = {
            "next_index": len(self.data),
            "completed": True,
        }
        self.save_checkpoint(final_checkpoint)

        return results

    def shutdown(self):
        try:
            stop_profile = getattr(self.llm, "stop_profile", None)
            if callable(stop_profile):
                stop_profile()
        except Exception:
            pass
        if self.use_wandb:
            wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = OmegaConf.load(args.config)

    collector = CombinedCollector(
        model_name=config["model"],
        data_path=config["data_path"],
        output_dir=config["output_dir"],
        temperature=config["temperature"],
        top_k=config["top_k"],
        max_logprobs=config["max_logprobs"],
        num_rollouts=config["num_rollouts"],
        max_new_tokens=config["max_new_tokens"],
        prompt_chunk_size=config["prompt_chunk_size"],
        use_wandb=config["use_wandb"],
        project_name=config["project_name"],
        wandb_api_key=config["wandb_api_key"],
        resume=config["resume"],
    )

    try:
        traces = collector.collect()
        logger.info(
            f"Combined collection completed: Total {len(traces)} traces processed."
        )
    except (Exception, KeyboardInterrupt) as e:
        logger.error(f"Error in combined collection: {e}")
    finally:
        collector.shutdown()
