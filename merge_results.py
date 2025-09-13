import json
import argparse
import logging
from typing import List, Dict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def merge_results(logprobs_file: str, rollouts_file: str, output_file: str):
    """
    Merge logprobs results and rollouts results into a single combined file.
    """
    
    # Load logprobs results
    try:
        with open(logprobs_file, "r", encoding="utf-8") as f:
            logprobs_results = json.load(f)
        logger.info(f"Loaded {len(logprobs_results)} logprobs results")
    except Exception as e:
        logger.error(f"Failed to load logprobs results: {e}")
        return

    # Load rollouts results
    try:
        with open(rollouts_file, "r", encoding="utf-8") as f:
            rollouts_results = json.load(f)
        logger.info(f"Loaded {len(rollouts_results)} rollouts results")
    except Exception as e:
        logger.error(f"Failed to load rollouts results: {e}")
        return

    # Create lookup for rollouts results by sample_index
    rollouts_lookup = {result["sample_index"]: result for result in rollouts_results}

    # Merge results
    combined_results = []
    for logprobs_result in logprobs_results:
        sample_index = logprobs_result["sample_index"]
        
        if sample_index in rollouts_lookup:
            rollouts_result = rollouts_lookup[sample_index]
            
            # Combine the results
            combined_result = {
                "sample_index": sample_index,
                "problem": logprobs_result["problem"],
                "true_answer": rollouts_result["true_answer"],
                "solution": logprobs_result["solution"],
                "parsed_answer": rollouts_result["parsed_answer"],
                "entropy": logprobs_result["entropy"],
                "confidence": logprobs_result["confidence"],
                "correctness": rollouts_result["correctness"],
            }
            combined_results.append(combined_result)
        else:
            logger.warning(f"No rollouts result found for sample_index {sample_index}")

    # Save combined results
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(combined_results, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(combined_results)} combined results to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save combined results: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge logprobs and rollouts results")
    parser.add_argument("--logprobs_file", type=str, required=True,
                       help="Path to logprobs results JSON file")
    parser.add_argument("--rollouts_file", type=str, required=True,
                       help="Path to rollouts results JSON file")
    parser.add_argument("--output_file", type=str, required=True,
                       help="Path to output combined results JSON file")
    
    args = parser.parse_args()
    
    merge_results(args.logprobs_file, args.rollouts_file, args.output_file)
