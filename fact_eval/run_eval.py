import argparse
import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import yaml
import torch
import re


def load_data_from_files(input_paths: List[str]) -> List[Dict[str, Any]]:
    """
    Load data from multiple input files.
    Args:
        input_paths: List of file paths to load data from
    Returns:
        List of data items
    """
    data_all = []
    if isinstance(input_paths, str):
        input_paths = [input_paths]
    for input_file in input_paths:
        if not os.path.exists(input_file):
            print(f"Warning: File {input_file} does not exist, skipping...")
            continue
        try:
            if input_file.endswith('.jsonl'):
                with open(input_file, "r") as f:
                    data = [json.loads(x) for x in f.readlines() if x.strip()]
                    data_all.extend(data)
            elif input_file.endswith('.json'):
                with open(input_file, "r") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    if "data" not in data:
                        raise ValueError(f"JSON file {input_file} must contain a 'data' key")
                    data = data["data"]
                data_all.extend(data)
            else:
                print(f"Warning: Unsupported file type {input_file}, skipping...")
        except Exception as e:
            print(f"Error loading file {input_file}: {e}")
            continue
    return data_all


def extract_final_answer(prediction: str) -> str:
    """
    Extract the substring between <answer> and </answer>.
    If no match is found, extract the substring after </think>.
    If neither condition matches, clean the prediction by removing the <|assistant|> tag.
    If none of the above applies, return the original string.

    Args:
        prediction (str): The input string.

    Returns:
        str: The extracted substring or the cleaned/original string.
    """
    answer_match = re.search(r"<answer>(.*?)</answer>", prediction, re.DOTALL)
    if answer_match:
        return answer_match.group(1).strip()

    think_match = re.search(r"</think>(.*)", prediction, re.DOTALL)
    if think_match:
        return think_match.group(1).strip()

    cleaned = re.sub(r"<\|assistant\|>", "", prediction)
    if cleaned != prediction:
        return cleaned.strip()

    return prediction


def main():
    """Main function to run factuality evaluation."""
    parser = argparse.ArgumentParser(description="Compute factuality scores for model responses")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    parser.add_argument("--config_kwargs", type=str, help="Config kwargs")

    parser.add_argument("--scorer", type=str, help="Scoring method to use")
    parser.add_argument("--data_name", type=str, help="Huggingface dataset name")
    parser.add_argument("--data_split", type=str, help="Data split for Huggingface dataset")
    parser.add_argument("--data_path", type=str, help="Data file path (json/jsonl)")
    parser.add_argument("--max_samples", type=int, help="Max samples to evaluate")

    parser.add_argument("--model_name_or_path", type=str, help="Model name or path")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max tokens for model to generate")
    parser.add_argument("--model_alias", type=str, help="Tag for saving intermediate/final results")
    parser.add_argument("--response_path", type=str, help="Response file path")
    
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--cache_dir", type=str, help="Output directory")
    parser.add_argument("--use_cache", action="store_true", help="Use cache")

    parser_args = parser.parse_args()
    if parser_args.cache_dir is None:
        parser_args.cache_dir = parser_args.output_dir

    # Load config and merge with CLI args
    with open(parser_args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    if parser_args.config_kwargs is not None:
        config_kwargs = json.loads(parser_args.config_kwargs)
    else:
        config_kwargs = {}
    
    args = {**config, **config_kwargs}
    args.update({k: v for k, v in vars(parser_args).items() if v is not None or args.get(k) is None})
    args = argparse.Namespace(**args)

    # Data loading
    if args.data_name:
        import datasets
        data_raw = datasets.load_dataset(args.data_name, split=args.data_split, streaming=True)
        data = [dict(item) for item in data_raw]
        print(f"Loaded {len(data)} samples from {args.data_name} {args.data_split}")
    elif args.data_path:
        # Accept comma-separated paths or a single path
        if "," in args.data_path:
            paths = [p.strip() for p in args.data_path.split(",")]
        else:
            paths = [args.data_path]
        data = load_data_from_files(paths)
        print(f"Loaded {len(data)} samples from {args.data_path}")
    else:
        raise ValueError("Either --data_name or --data_path must be provided.")
    # if "ground_truth" in data, change the name to "docs"
    for item in data:
        if "ground_truth" in item:
            item["docs"] = item["ground_truth"]
            del item["ground_truth"]
        if "question" in item:
            item["prompt"] = item["question"]
            del item["question"]
    # Attach responses if provided
    if args.response_path:
        # ifgnore args.max_samples
        if args.max_samples:
            print(f"Warning: --max_samples is ignored when --response_path is provided")
        
        if "," in args.response_path:
            resp_paths = [p.strip() for p in args.response_path.split(",")]
        else:
            resp_paths = [args.response_path]
        response_list = load_data_from_files(resp_paths)
        response_list = [{
            "prompt": item["prompt"] if "prompt" in item else item["input"],    # compatibility with factscore
            "response": item["response"] if "response" in item else item["output"],
        } for item in response_list]
        prompt_to_item = {item["prompt"]: item for item in data}
        data = [{
            "prompt": item["prompt"],
            "response": item["response"],
            "docs": prompt_to_item[item["prompt"]].get("docs", []),
            "model": args.model_name_or_path,
            "model_alias": args.model_alias,
        } for item in response_list]
    else:
        if args.max_samples:
            random.seed(42)
            random_indices = random.sample(range(len(data)), int(args.max_samples))
            data = [data[i] for i in random_indices]
        # If no response_path, generate responses (OpenAI/vLLM)
        if not args.model_name_or_path:
            raise ValueError("--model_name_or_path must be provided if --response_path is not used.")
        if args.model_name_or_path.startswith("openai::"):
            from fact_eval.utils.async_completion import batch_chat_complete
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            messages_list = [[{"role": "user", "content": item["prompt"]}] for item in data]
            outputs = batch_chat_complete(
                client=client,
                messages_list=messages_list,
                model=args.model_name_or_path.split("::")[-1],
                max_tokens=args.max_new_tokens,
            )
            response_list = [output.choices[0].message.content for output in outputs]
        else:
            import vllm
            model = vllm.LLM(model=args.model_name_or_path, tensor_parallel_size=1)
            messages_list = [[{"role": "user", "content": item["prompt"]}] for item in data]
            outputs = model.chat(messages=messages_list, sampling_params=vllm.SamplingParams(temperature=0.0, max_tokens=512))
            response_list = [output.outputs[0].text for output in outputs]
            # delete vllm model
            import gc
            del model
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.empty_cache()
        for item, response in zip(data, response_list):
            item["output"] = response
            item["response"] = extract_final_answer(response)
            item["model"] = args.model_name_or_path
            item["model_alias"] = args.model_alias
        response_file = os.path.join(args.output_dir, "response.jsonl")
        os.makedirs(args.output_dir, exist_ok=True)
        with open(response_file, "w") as f:
            for item in data:
                f.write(json.dumps({"prompt": item["prompt"], "response": item["response"], "output": item["output"], "model": item["model"]}) + "\n")

    data_clean = [{
        "prompt": item["prompt"],
        "response": item["response"],
        "docs": item["docs"] if "docs" in item else None,
    } for item in data]

    # Scoring
    if args.scorer == "veriscore":
        from fact_eval.scorer.veriscore import VeriScoreConfig, VeriScorer
        scorer_config = VeriScoreConfig(**{k: v for k, v in vars(args).items() if k in VeriScoreConfig.__dataclass_fields__})
        scorer = VeriScorer(scorer_config)
        aggregate_metrics, per_instance_results = scorer.get_score(data_clean, model_alias=args.model_alias)
    elif args.scorer == "factscore":
        from fact_eval.scorer.factscore import FactScoreConfig, FactScorer
        scorer_config = FactScoreConfig(**{k: v for k, v in vars(args).items() if k in FactScoreConfig.__dataclass_fields__})
        scorer = FactScorer(scorer_config)
        aggregate_metrics, per_instance_results = scorer.get_score(data_clean, model_alias=args.model_alias)
    else:
        raise ValueError(f"Invalid scorer: {args.scorer}")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(aggregate_metrics, f, indent=2)
    with open(os.path.join(args.output_dir, "results_per_instance.jsonl"), "w") as f:
        for result in per_instance_results:
            f.write(json.dumps(result) + "\n")


if __name__ == '__main__':
    main()
