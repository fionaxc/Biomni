"""
Example: Using Biomni with Stanford Healthcare's Secure API

This example demonstrates how to configure and use Biomni with the secure
API gateway, similar to the run_llm_baseline.py workflow.

Setup:
1. Set your API key in environment variable: OPENAI_API_KEY
2. Configure the secure API URL and model ID
3. Use Biomni agents with the secure API

For more information about available models and endpoints, see:
/Users/fionacai/Library/CloudStorage/Box-Box/Fiona Cai's Externally Shareable Files/FC-Alsentzer/rare-project/rare-kg/run_llm_baseline.py
"""

import os
from biomni.config import BiomniConfig
from biomni.agent.react import react

# ========== SECURE API CONFIGURATION ==========
# Available model configurations (from run_llm_baseline.py)

# GPT-4o
gpt4o_url = "https://apim.stanfordhealthcare.org/openai24/deployments/gpt-4o/chat/completions?api-version=2023-05-15"
gpt4o_modelid = "gpt-4o"

# GPT-4.1
gpt41_url = "https://apim.stanfordhealthcare.org/openai-eastus2/deployments/gpt-4.1/chat/completions?api-version=2025-01-01-preview"
gpt41_modelid = "gpt-4.1"

# Claude 3.5 Sonnet v2
claude35_url = "https://apim.stanfordhealthcare.org/Claude35Sonnetv2/awssig4fa"
claude35_modelid = "anthropic.claude-3-5-sonnet-20241022-v2:0"

# Claude 3.7 Sonnet
claude37_url = "https://apim.stanfordhealthcare.org/awssig4claude37/aswsig4claude37"
claude37_modelid = "arn:aws:bedrock:us-west-2:679683451337:inference-profile/us.anthropic.claude-3-7-sonnet-20250219-v1:0"

# DeepSeek R1
deepseekr1_url = "https://apim.stanfordhealthcare.org/deepseekr1/v1/chat/completions"
deepseekr1_modelid = "deepseek-chat"

# Llama 3.3 70B
llama33_url = "https://apim.stanfordhealthcare.org/llama3370b/v1/chat/completions"
llama33_modelid = "Llama-3.3-70B-Instruct"

# Llama 4 Maverick
llama4_maverick_url = "https://apim.stanfordhealthcare.org/llama4-maverick/v1/chat/completions"
llama4_maverick_modelid = "Llama-4-Maverick-17B-128E-Instruct-FP8"

# Llama 4 Scout
llama4_scout_url = "https://apim.stanfordhealthcare.org/llama4-scout/v1/chat/completions"
llama4_scout_modelid = "Llama-4-Scout-17B-16E-Instruct"


# ========== EXAMPLE 1: Using Biomni with Secure API via Environment Variables ==========
def example_1_env_variables():
    """
    Example using environment variables to configure the secure API.

    Before running, set these environment variables:
    export OPENAI_API_KEY="your-subscription-key"
    export BIOMNI_SOURCE="SecureAPI"
    export BIOMNI_SECURE_API_URL="https://apim.stanfordhealthcare.org/Claude35Sonnetv2/awssig4fa"
    export BIOMNI_SECURE_MODEL_ID="anthropic.claude-3-5-sonnet-20241022-v2:0"
    """
    print("Example 1: Using environment variables")
    print("=" * 60)

    # Create config with just the data path
    # Other settings will be read from environment variables
    config = BiomniConfig(path="./data")

    # Create a react agent
    agent = react(
        path=config.path,
        llm=config.llm,
        use_tool_retriever=config.use_tool_retriever,
        timeout_seconds=config.timeout_seconds,
        config=config
    )

    # Use the agent
    question = "What are the main functions of mitochondria?"
    trace, answer = agent.go(question)

    print(f"\nQuestion: {question}")
    print(f"Answer: {answer}")
    print(f"\nTrace length: {len(trace)} steps")


# ========== EXAMPLE 2: Using Biomni with Secure API via BiomniConfig ==========
def example_2_config_object():
    """
    Example using BiomniConfig to programmatically configure the secure API.
    This approach doesn't rely on environment variables (except for API key).
    """
    print("\nExample 2: Using BiomniConfig object")
    print("=" * 60)

    # Get API key from environment
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        return

    # Create config with secure API settings - using GPT-4.1 which supports tool calling
    config = BiomniConfig(
        path="./data",
        source="SecureAPI",
        secure_api_url=gpt41_url,
        secure_model_id=gpt41_modelid,
        api_key=api_key,
        temperature=0.7,
    )

    # Create a react agent
    agent = react(
        path=config.path,
        llm=config.llm,
        use_tool_retriever=config.use_tool_retriever,
        timeout_seconds=config.timeout_seconds,
        config=config
    )

    # Use the agent
    question = """You are an expert in rare disease diagnosis.

    IMPORTANT: First, use the available database tools to research each candidate gene and their associations with the provided HPO terms.
    Query databases like ClinVar, OpenTargets, OMIM, Ensembl, and genetic variant databases to gather evidence about disease-gene associations.
    Research each gene thoroughly before ranking.

    After completing your research, generate a ranked list of all candidate genes based on their likelihood of causing the patient's symptoms.

    Patient HPO terms:
    [Male hypogonadism, Azoospermia, Increased circulating gonadotropin level, Obesity, Increased body weight, Primary testicular failure, Decreased testicular size, Sex reversal, Decreased serum testosterone level]

    Candidate genes to rank:
    [BNC2, LHCGR, AKR1C4, NR5A1, RSPO1]

    The output must be in JSON Lines (jsonl) format with NO markdown formatting. Each line must be a valid JSON object containing exactly these fields:
    - gene_name: string
    - rank: number
    - explanation: string

    Example format:
    {"gene_name":"GENE1","rank":1,"explanation":"Explanation here"}
    {"gene_name":"GENE2","rank":2,"explanation":"Explanation here"}

    Do not include any other text, markdown formatting, or code block markers in your response. Only output the JSON lines.
    """
    trace, answer = agent.go(question)

    print(f"\nQuestion: {question}")
    print(f"Answer: {answer}")
    print(f"\nTrace length: {len(trace)} steps")


# ========== EXAMPLE 3: Comparing Different Models ==========
def example_3_compare_models():
    """
    Example comparing responses from different models available through the secure API.
    """
    print("\nExample 3: Comparing different models")
    print("=" * 60)

    # Get API key from environment
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        return

    # List of models to compare
    models = [
        ("Claude 3.5 Sonnet", claude35_url, claude35_modelid),
        ("GPT-4o", gpt4o_url, gpt4o_modelid),
        ("Llama 3.3 70B", llama33_url, llama33_modelid),
    ]

    question = "What is CRISPR?"

    for model_name, url, model_id in models:
        print(f"\n--- Testing {model_name} ---")

        # Create config for this model
        config = BiomniConfig(
            path="./data",
            source="SecureAPI",
            secure_api_url=url,
            secure_model_id=model_id,
            api_key=api_key,
            temperature=0.7,
        )

        # Create agent
        agent = react(
            path=config.path,
            llm=config.llm,
            use_tool_retriever=config.use_tool_retriever,
            timeout_seconds=config.timeout_seconds,
            config=config
        )

        # Get answer
        try:
            trace, answer = agent.go(question)
            print(f"Answer: {answer[:200]}...")  # Print first 200 chars
        except Exception as e:
            print(f"Error: {e}")


# ========== EXAMPLE 4: Using Secure API for Simple Q&A (without tools) ==========
def example_4_simple_qa():
    """
    Example using the secure API for simple question-answering without tools.
    Similar to the qa_llm agent in Biomni.
    """
    print("\nExample 4: Simple Q&A without tools")
    print("=" * 60)

    from biomni.agent.qa_llm import qa_llm

    # Get API key from environment
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        return

    # Create config with secure API settings
    config = BiomniConfig(
        path="./data",
        source="SecureAPI",
        secure_api_url=claude35_url,
        secure_model_id=claude35_modelid,
        api_key=api_key,
        temperature=0.7,
    )

    # Create a simple Q&A agent
    agent = qa_llm(path=config.path, llm=config.llm, config=config)

    # Use the agent
    question = "What is the central dogma of molecular biology?"
    trace, answer = agent.go(question)

    print(f"\nQuestion: {question}")
    print(f"Answer: {answer}")


# ========== MAIN ==========
if __name__ == "__main__":
    print("Stanford Healthcare Secure API with Biomni - Examples")
    print("=" * 60)

    # Check if API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        print("\nError: OPENAI_API_KEY environment variable not set")
        print("Please set it with your secure API subscription key:")
        print("export OPENAI_API_KEY='your-subscription-key'")
        exit(1)

    # Run examples
    # Uncomment the example you want to run:

    # example_1_env_variables()  # Requires additional env vars
    example_2_config_object()
    # example_3_compare_models()
    # example_4_simple_qa()

    print("\n" + "=" * 60)
    print("Examples completed!")
