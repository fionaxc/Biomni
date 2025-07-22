#!/usr/bin/env python3
"""
Example: Using Biomni with a Stanford Secure API Endpoint 

"""

import os
import sys
sys.path.append("../")

from biomni.agent import A1
DEFAULT_API_KEY = os.environ.get("OPENAI_API_KEY")

def main():
    # Your secure API configuration
    gpt4o_url = "https://apim.stanfordhealthcare.org/openai24/deployments/gpt-4o/chat/completions?api-version=2023-05-15"
    gpt4o_modelid = "gpt-4o"
    api_key = DEFAULT_API_KEY
    
    print("Initializing Biomni agent with secure API...")
    
    # Initialize the agent with your secure API
    agent = A1(
        path="./data", 
        llm=gpt4o_modelid,
        base_url=gpt4o_url,
        api_key=api_key
    )
    
    print("Agent initialized successfully!")
    print(f"Using model: {gpt4o_modelid}")
    print(f"API endpoint: {gpt4o_url}")
    
    # Example task
    task = """
        You are a clinical genomics expert specializing in diagnosing rare genetic disorders.

        A patient presents with the following phenotypic features (encoded as Human Phenotype Ontology (HPO) terms):

        HPO Terms:
        - Epicanthus
        - Global developmental delay
        - Intellectual disability
        - Hypotonia
        - Speech apraxia
        - Obesity
        - Almond-shaped eyes
        - Hyperphagia
        - Narrow forehead
        - Downturned corners of mouth

        The following is a list of candidate genes identified from the patient’s exome sequencing data:

        Candidate Genes:
        - SNRPN
        - MAGEL2
        - UBE3A
        - MECP2
        - SHANK3

        Task:
        Given the phenotypic features and candidate genes, rank the genes in order of their likelihood of being the causal gene for this patient. For each gene, briefly explain your reasoning, citing known gene–phenotype associations, disease relevance, and any supporting biological knowledge.

        Output format:
        A JSON object list where each object includes:
        - gene_name: string
        - rank: integer
        - explanation: string

        Example output:
        [
        {"gene_name": "GENE1", "rank": 1, "explanation": "Explanation here"},
        {"gene_name": "GENE2", "rank": 2, "explanation": "Explanation here"}
        ]
        """
 
    
    print("\nExecuting task...")
    print("=" * 50)
    print(task)
    print("=" * 50)
    
    # Execute the task
    log, result = agent.go(task)
    
    print("\nTask completed!")
    print("Final result:", result)

if __name__ == "__main__":
    main() 