#!/usr/bin/env python3
"""
Script to run expert curated patient prompts using Biomni with secure API.

This script:
1. Loads prompts from expert_curated_patient_prompts.jsonl
2. Runs each prompt through Biomni using the secure API
3. Saves results to a timestamped output file

Setup:
    export OPENAI_API_KEY="your-subscription-key"

Usage:
    python run_udn_biomni.py [--model MODEL] [--start START] [--end END] [--output OUTPUT]

Examples:
    # Run all prompts with Claude 3.7
    python run_udn_biomni.py --model claude37

    # Run specific range with GPT-4o
    python run_udn_biomni.py --model gpt4o --start 0 --end 10

    # Run with custom output file
    python run_udn_biomni.py --output results.jsonl
"""

import os
import json
import argparse
from datetime import datetime
from tqdm import tqdm
from pathlib import Path

from biomni.config import BiomniConfig
from biomni.agent.qa_llm import qa_llm

# ========== SECURE API CONFIGURATIONS ==========
MODEL_CONFIGS = {
    "gpt4o": {
        "url": "https://apim.stanfordhealthcare.org/openai24/deployments/gpt-4o/chat/completions?api-version=2023-05-15",
        "model_id": "gpt-4o"
    },
    "gpt41": {
        "url": "https://apim.stanfordhealthcare.org/openai-eastus2/deployments/gpt-4.1/chat/completions?api-version=2025-01-01-preview",
        "model_id": "gpt-4.1"
    },
    "claude35": {
        "url": "https://apim.stanfordhealthcare.org/Claude35Sonnetv2/awssig4fa",
        "model_id": "anthropic.claude-3-5-sonnet-20241022-v2:0"
    },
    "claude37": {
        "url": "https://apim.stanfordhealthcare.org/awssig4claude37/aswsig4claude37",
        "model_id": "arn:aws:bedrock:us-west-2:679683451337:inference-profile/us.anthropic.claude-3-7-sonnet-20250219-v1:0"
    },
    "deepseekr1": {
        "url": "https://apim.stanfordhealthcare.org/deepseekr1/v1/chat/completions",
        "model_id": "deepseek-chat"
    },
    "llama33": {
        "url": "https://apim.stanfordhealthcare.org/llama3370b/v1/chat/completions",
        "model_id": "Llama-3.3-70B-Instruct"
    },
    "llama4_maverick": {
        "url": "https://apim.stanfordhealthcare.org/llama4-maverick/v1/chat/completions",
        "model_id": "Llama-4-Maverick-17B-128E-Instruct-FP8"
    },
    "llama4_scout": {
        "url": "https://apim.stanfordhealthcare.org/llama4-scout/v1/chat/completions",
        "model_id": "Llama-4-Scout-17B-16E-Instruct"
    }
}

# Default paths
DEFAULT_PROMPTS_FILE = "/Users/fionacai/Library/CloudStorage/Box-Box/Fiona Cai's Externally Shareable Files/FC-Alsentzer/rare-project/prompts/expert_curated_patient_prompts.jsonl"
DEFAULT_DATA_PATH = "./data"


def load_prompts(prompts_file: str, start_idx: int = None, end_idx: int = None):
    """Load prompts from JSONL file."""
    with open(prompts_file, "r") as f:
        prompts = [json.loads(line.strip()) for line in f]

    if start_idx is not None or end_idx is not None:
        start = start_idx if start_idx is not None else 0
        end = end_idx if end_idx is not None else len(prompts)
        prompts = prompts[start:end]

    return prompts


def create_biomni_agent(model_name: str, api_key: str, data_path: str = DEFAULT_DATA_PATH):
    """Create a Biomni agent configured with the secure API."""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(MODEL_CONFIGS.keys())}")

    model_config = MODEL_CONFIGS[model_name]

    # Create config with secure API settings
    config = BiomniConfig(
        path=data_path,
        source="SecureAPI",
        secure_api_url=model_config["url"],
        secure_model_id=model_config["model_id"],
        api_key=api_key,
        temperature=0.0,  # Use temperature 0 for consistent results
    )

    # Create a Q&A agent (no tools needed for this task)
    agent = qa_llm(path=config.path, llm=config.llm, config=config)

    return agent


def run_prompts(prompts, agent, output_file: str, model_name: str):
    """Run all prompts through the agent and save results."""
    results = []
    errors = []

    print(f"\nRunning {len(prompts)} prompts with model: {model_name}")
    print(f"Results will be saved to: {output_file}")
    print("=" * 80)

    for i, prompt_data in enumerate(tqdm(prompts, desc="Processing prompts")):
        patient_id = prompt_data["patient_id"]
        prompt = prompt_data["prompt"]

        try:
            # Run the prompt through the agent
            trace, answer = agent.go(prompt)

            # Save result
            result = {
                "patient_id": patient_id,
                "model": model_name,
                "timestamp": datetime.now().isoformat(),
                "prompt": prompt,
                "response": answer,
                "trace_length": len(trace) if trace else 0,
                "success": True
            }
            results.append(result)

            # Write result immediately (in case of interruption)
            with open(output_file, "a") as f:
                f.write(json.dumps(result) + "\n")

        except Exception as e:
            error_msg = f"Error processing patient {patient_id}: {str(e)}"
            print(f"\n{error_msg}")
            errors.append({
                "patient_id": patient_id,
                "error": str(e)
            })

            # Save error result
            result = {
                "patient_id": patient_id,
                "model": model_name,
                "timestamp": datetime.now().isoformat(),
                "prompt": prompt,
                "response": None,
                "error": str(e),
                "success": False
            }
            with open(output_file, "a") as f:
                f.write(json.dumps(result) + "\n")

    # Print summary
    print("\n" + "=" * 80)
    print(f"SUMMARY:")
    print(f"  Total prompts: {len(prompts)}")
    print(f"  Successful: {len(results) - len(errors)}")
    print(f"  Errors: {len(errors)}")
    print(f"  Output file: {output_file}")

    if errors:
        print(f"\nErrors occurred for patients:")
        for error in errors:
            print(f"  - {error['patient_id']}: {error['error']}")

    return results, errors


def main():
    parser = argparse.ArgumentParser(
        description="Run patient prompts using Biomni with secure API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all prompts with Claude 3.7
  python run_udn_biomni.py --model claude37

  # Run specific range with GPT-4o
  python run_udn_biomni.py --model gpt4o --start 0 --end 10

  # Run with custom output file
  python run_udn_biomni.py --output results.jsonl

Available models: """ + ", ".join(MODEL_CONFIGS.keys())
    )

    parser.add_argument(
        "--model",
        type=str,
        default="claude37",
        choices=list(MODEL_CONFIGS.keys()),
        help="Model to use (default: claude37)"
    )
    parser.add_argument(
        "--prompts-file",
        type=str,
        default=DEFAULT_PROMPTS_FILE,
        help=f"Path to prompts JSONL file (default: {DEFAULT_PROMPTS_FILE})"
    )
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="Start index (inclusive, default: 0)"
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="End index (exclusive, default: all)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: results_MODEL_TIMESTAMP.jsonl)"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=DEFAULT_DATA_PATH,
        help=f"Path to Biomni data directory (default: {DEFAULT_DATA_PATH})"
    )

    args = parser.parse_args()

    # Check for API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please set it with your secure API subscription key:")
        print("export OPENAI_API_KEY='your-subscription-key'")
        return 1

    # Create output filename if not specified
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"results_{args.model}_{timestamp}.jsonl"

    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Load prompts
    print(f"Loading prompts from: {args.prompts_file}")
    prompts = load_prompts(args.prompts_file, args.start, args.end)
    print(f"Loaded {len(prompts)} prompts")

    # Create agent
    print(f"Creating Biomni agent with model: {args.model}")
    agent = create_biomni_agent(args.model, api_key, args.data_path)

    # Run prompts
    results, errors = run_prompts(prompts, agent, args.output, args.model)

    return 0 if not errors else 1


if __name__ == "__main__":
    exit(main())
