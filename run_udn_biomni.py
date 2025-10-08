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
    python run_udn_biomni.py [--start START] [--end END] [--output OUTPUT]

Examples:
    # Run all prompts
    python run_udn_biomni.py

    # Run specific range
    python run_udn_biomni.py --start 0 --end 10

    # Run with custom output file
    python run_udn_biomni.py --output results.jsonl
"""

import os
import json
import argparse
import re
import csv
from datetime import datetime
from tqdm import tqdm
from pathlib import Path
from biomni.agent import A1
from biomni.config import default_config
from biomni.agent.react import react

# ========== SECURE API CONFIGURATIONS ==========
GPT41_CONFIG = {
    "url": "https://apim.stanfordhealthcare.org/openai-eastus2/deployments/gpt-4.1/chat/completions?api-version=2025-01-01-preview",
    "model_id": "gpt-4.1"
}

# Default paths
DEFAULT_PROMPTS_FILE = "/share/pi/ema2016/users/fionacai/project/rare-kg/prompts/expert_curated_patient_prompts.jsonl"
DEFAULT_DATA_PATH = "./data"
DEFAULT_OUTPUT_DIR = "/share/pi/ema2016/users/fionacai/project/rare-kg/results/Biomni_UDN_expert"

def load_prompts(prompts_file: str, start_idx: int = None, end_idx: int = None):
    """Load prompts from JSONL file."""
    with open(prompts_file, "r") as f:
        prompts = [json.loads(line.strip()) for line in f]

    if start_idx is not None or end_idx is not None:
        start = start_idx if start_idx is not None else 0
        end = end_idx if end_idx is not None else len(prompts)
        prompts = prompts[start:end]

    return prompts

def create_biomni_agent(api_key: str, data_path: str = DEFAULT_DATA_PATH):
    """Create a Biomni agent configured with the secure API."""
    # Configure default_config with secure API settings

    default_config.source = "SecureAPI"
    default_config.secure_api_url = "https://apim.stanfordhealthcare.org/openai-eastus2/deployments/gpt-4.1/chat/completions?api-version=2025-01-01-preview"
    default_config.secure_model_id = "gpt-4.1"
    default_config.api_key = os.environ.get("OPENAI_API_KEY")
    agent = A1()

    return agent

def parse_genes_from_response(response: str):
    """Parse ranked genes from JSONL response."""
    genes = []
    if not response:
        return genes

    for line in response.strip().split('\n'):
        try:
            gene_data = json.loads(line)
            genes.append({
                "gene_name": gene_data.get("gene_name", ""),
                "rank": gene_data.get("rank", -1),
                "explanation": gene_data.get("explanation", "")
            })
        except json.JSONDecodeError:
            continue

    # Sort by rank
    genes.sort(key=lambda x: x.get("rank", 999))
    return genes


def run_prompts(prompts, agent, output_file: str):
    """Run all prompts through the agent and save results."""
    results = []
    errors = []
    csv_rows = []

    # Create output directory structure
    output_path = Path(output_file)
    output_dir = output_path.parent if output_path.parent != Path('.') else Path.cwd()
    traces_dir = output_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)

    csv_file = output_dir / "results_gpt41_summary.csv"

    print(f"\nRunning {len(prompts)} prompts with model: gpt-4.1")
    print(f"JSONL results: {output_file}")
    print(f"Traces directory: {traces_dir}")
    print(f"CSV summary: {csv_file}")
    print("=" * 80)

    for i, prompt_data in enumerate(tqdm(prompts, desc="Processing prompts")):
        patient_id = prompt_data["patient_id"]
        prompt = prompt_data["prompt"]
        true_genes = prompt_data.get("true_causal_gene")
        if not true_genes:
            print("no true gene inputted")
            continue

        try:
            # Run the prompt through the agent
            trace, answer = agent.go(prompt)

            # Parse genes from response
            parsed_genes = parse_genes_from_response(answer)
            top_gene = parsed_genes[0]["gene_name"] if parsed_genes else None

            # Find rank of true causal gene
            rank_of_true = None
            if true_genes:
                true_gene = true_genes[0]
                if top_gene == true_gene.upper():
                    rank_of_true = parsed_genes[0]["rank"]
                    break
            else:
                print("no true genes inputted")

            # Save result to JSONL
            result = {
                "patient_id": patient_id,
                "model": "gpt-4.1",
                "timestamp": datetime.now().isoformat(),
                "prompt": prompt,
                "response": answer,
                "trace_length": len(trace) if trace else 0,
                "parsed_genes": parsed_genes,
                "true_causal_gene": true_gene,
                "top_ranked_gene": top_gene,
                "rank_of_true_gene": rank_of_true,
                "success": True
            }
            results.append(result)

            # Write JSONL result immediately (in case of interruption)
            with open(output_file, "a") as f:
                f.write(json.dumps(result) + "\n")

            # Save trace to individual txt file
            trace_file = traces_dir / f"{patient_id}_trace.txt"
            with open(trace_file, "w") as f:
                f.write(f"Patient ID: {patient_id}\n")
                f.write(f"Model: gpt-4.1\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write("=" * 80 + "\n\n")
                if trace:
                    for step_num, step in enumerate(trace, 1):
                        f.write(f"STEP {step_num}:\n")
                        f.write(str(step) + "\n")
                        f.write("-" * 80 + "\n")
                else:
                    f.write("No trace available\n")

            # Add to CSV data
            csv_rows.append({
                "patient_id": patient_id,
                "top_ranked_gene": top_gene or "",
                "true_causal_gene": true_gene or "",
                "rank_of_true_gene": rank_of_true if rank_of_true is not None else ""
            })

        except Exception as e:
            error_msg = f"Error processing patient {patient_id}: {str(e)}"
            print(f"\n{error_msg}")
            errors.append({
                "patient_id": patient_id,
                "error": str(e)
            })

            # Save error result to JSONL
            result = {
                "patient_id": patient_id,
                "model": "gpt-4.1",
                "timestamp": datetime.now().isoformat(),
                "prompt": prompt,
                "response": None,
                "error": str(e),
                "success": False
            }
            with open(output_file, "a") as f:
                f.write(json.dumps(result) + "\n")

            # Add error to CSV
            csv_rows.append({
                "patient_id": patient_id,
                "top_ranked_gene": "ERROR",
                "true_causal_gene": true_gene or "",
                "rank_of_true_gene": "ERROR"
            })

    # Write CSV summary
    with open(csv_file, "w", newline="") as f:
        fieldnames = ["patient_id", "top_ranked_gene", "true_causal_gene", "rank_of_true_gene"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

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
  # Run all prompts
  python run_udn_biomni.py

  # Run specific range
  python run_udn_biomni.py --start 0 --end 10

  # Run with custom output directory
  python run_udn_biomni.py --output-dir ./my_results
"""
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
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for all results (default: {DEFAULT_OUTPUT_DIR})"
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

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(args.output_dir, f"results_gpt41_{timestamp}.jsonl")

    # Load prompts
    print(f"Loading prompts from: {args.prompts_file}")
    prompts = load_prompts(args.prompts_file, args.start, args.end)
    print(f"Loaded {len(prompts)} prompts")

    # Create agent
    print("Creating Biomni agent with model: gpt-4.1")
    agent = create_biomni_agent(api_key, args.data_path)

    # Run prompts
    results, errors = run_prompts(prompts, agent, output_file)

    return 0 if not errors else 1

if __name__ == "__main__":
    exit(main())