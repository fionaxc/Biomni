# Using Biomni with Stanford Healthcare Secure API

This guide explains how to use Biomni with the Stanford Healthcare secure API gateway to access various LLMs (Claude, GPT, Llama, etc.) through a secure endpoint.

## Overview

The secure API integration allows you to use Biomni with models hosted on Stanford Healthcare's secure API gateway. This is particularly useful for working with sensitive data or in environments requiring additional security controls.

## Prerequisites

1. A valid subscription key for the Stanford Healthcare API gateway
2. Access to the secure API endpoints
3. Biomni installed with required dependencies

## Available Models

The following models are available through the secure API:

| Model | Model ID | API URL |
|-------|----------|---------|
| GPT-4o | `gpt-4o` | `https://apim.stanfordhealthcare.org/openai24/deployments/gpt-4o/chat/completions?api-version=2023-05-15` |
| GPT-4.1 | `gpt-4.1` | `https://apim.stanfordhealthcare.org/openai-eastus2/deployments/gpt-4.1/chat/completions?api-version=2025-01-01-preview` |
| Claude 3.5 Sonnet v2 | `anthropic.claude-3-5-sonnet-20241022-v2:0` | `https://apim.stanfordhealthcare.org/Claude35Sonnetv2/awssig4fa` |
| Claude 3.7 Sonnet | `arn:aws:bedrock:us-west-2:679683451337:inference-profile/us.anthropic.claude-3-7-sonnet-20250219-v1:0` | `https://apim.stanfordhealthcare.org/awssig4claude37/aswsig4claude37` |
| DeepSeek R1 | `deepseek-chat` | `https://apim.stanfordhealthcare.org/deepseekr1/v1/chat/completions` |
| Llama 3.3 70B | `Llama-3.3-70B-Instruct` | `https://apim.stanfordhealthcare.org/llama3370b/v1/chat/completions` |
| Llama 4 Maverick | `Llama-4-Maverick-17B-128E-Instruct-FP8` | `https://apim.stanfordhealthcare.org/llama4-maverick/v1/chat/completions` |
| Llama 4 Scout | `Llama-4-Scout-17B-16E-Instruct` | `https://apim.stanfordhealthcare.org/llama4-scout/v1/chat/completions` |

## Setup

### Method 1: Using Environment Variables

Set the following environment variables:

```bash
export OPENAI_API_KEY="your-subscription-key"
export BIOMNI_SOURCE="SecureAPI"
export BIOMNI_SECURE_API_URL="https://apim.stanfordhealthcare.org/Claude35Sonnetv2/awssig4fa"
export BIOMNI_SECURE_MODEL_ID="anthropic.claude-3-5-sonnet-20241022-v2:0"
```

Then use Biomni normally:

```python
from biomni.agent.react import react

agent = react()
trace, answer = agent.go("What are mitochondria?")
print(answer)
```

### Method 2: Using BiomniConfig (Recommended)

This method gives you more control and doesn't require setting environment variables (except for the API key):

```python
import os
from biomni.config import BiomniConfig
from biomni.agent.react import react

# Configure the secure API
config = BiomniConfig(
    path="./data",
    source="SecureAPI",
    secure_api_url="https://apim.stanfordhealthcare.org/Claude35Sonnetv2/awssig4fa",
    secure_model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
    api_key=os.environ["OPENAI_API_KEY"],
    temperature=0.7,
)

# Create agent with config
agent = react(
    path=config.path,
    llm=config.llm,
    use_tool_retriever=config.use_tool_retriever,
    timeout_seconds=config.timeout_seconds
)

# Use the agent
trace, answer = agent.go("What is DNA?")
print(answer)
```

### Method 3: Direct LLM Usage

You can also use the secure API directly with the `get_llm` function:

```python
import os
from biomni.llm import get_llm
from langchain_core.messages import HumanMessage

# Create LLM instance
llm = get_llm(
    source="SecureAPI",
    secure_api_url="https://apim.stanfordhealthcare.org/Claude35Sonnetv2/awssig4fa",
    secure_model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
    api_key=os.environ["OPENAI_API_KEY"],
    temperature=0.7,
)

# Use the LLM
response = llm.invoke([HumanMessage(content="What is photosynthesis?")])
print(response.content)
```

## Configuration Parameters

When using the secure API, you need to specify:

- **source**: Must be set to `"SecureAPI"`
- **secure_api_url**: The API endpoint URL for the specific model
- **secure_model_id**: The model identifier (varies by model)
- **api_key**: Your subscription key (typically from `OPENAI_API_KEY` environment variable)
- **temperature**: (Optional) Sampling temperature, default is 0.7

## Switching Between Models

To switch between different models, simply change the `secure_api_url` and `secure_model_id`:

```python
from biomni.config import BiomniConfig
from biomni.agent.react import react
import os

# Use Claude 3.5 Sonnet
config_claude = BiomniConfig(
    source="SecureAPI",
    secure_api_url="https://apim.stanfordhealthcare.org/Claude35Sonnetv2/awssig4fa",
    secure_model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
    api_key=os.environ["OPENAI_API_KEY"],
)

# Use GPT-4o
config_gpt4 = BiomniConfig(
    source="SecureAPI",
    secure_api_url="https://apim.stanfordhealthcare.org/openai24/deployments/gpt-4o/chat/completions?api-version=2023-05-15",
    secure_model_id="gpt-4o",
    api_key=os.environ["OPENAI_API_KEY"],
)

# Create agents with different models
agent_claude = react(path=config_claude.path, llm=config_claude.llm)
agent_gpt = react(path=config_gpt4.path, llm=config_gpt4.llm)
```

## Example: Replicating run_llm_baseline.py Workflow

To replicate the workflow from `run_llm_baseline.py` using Biomni:

```python
import os
import json
from biomni.config import BiomniConfig
from biomni.agent.react import react

# Model configuration (same as run_llm_baseline.py)
claude35_url = "https://apim.stanfordhealthcare.org/Claude35Sonnetv2/awssig4fa"
claude35_modelid = "anthropic.claude-3-5-sonnet-20241022-v2:0"

# Get API key
api_key = os.environ.get("OPENAI_API_KEY")

# Create Biomni config
config = BiomniConfig(
    path="./data",
    source="SecureAPI",
    secure_api_url=claude35_url,
    secure_model_id=claude35_modelid,
    api_key=api_key,
    temperature=0.7,
)

# Create agent
agent = react(
    path=config.path,
    llm=config.llm,
    use_tool_retriever=config.use_tool_retriever,
    timeout_seconds=config.timeout_seconds
)

# Load patient data (example)
with open("patient_file.txt", "r") as f:
    patient_data = [json.loads(line.strip()) for line in f]

# Process patients
for patient in patient_data:
    # Build your prompt based on patient data
    prompt = f"Analyze this patient: {patient}"

    # Get response from agent
    trace, answer = agent.go(prompt)

    # Process the answer
    print(f"Patient {patient.get('patient_id')}: {answer}")
```

## Troubleshooting

### API Key Not Found

If you see an error about missing API key:
- Ensure `OPENAI_API_KEY` environment variable is set
- Or pass `api_key` parameter directly to `BiomniConfig`

### Invalid Source Error

If you see "Invalid source" error:
- Ensure you've set `source="SecureAPI"`
- Check that `secure_api_url` and `secure_model_id` are provided

### Connection Errors

If you get connection errors:
- Verify you have access to the secure API endpoint
- Check that your subscription key is valid
- Ensure you're on the Stanford network or VPN if required

## Examples

See `examples/use_secure_api.py` for complete working examples including:
- Basic usage with environment variables
- Programmatic configuration
- Comparing different models
- Simple Q&A without tools

## Security Notes

- Never commit your API key to version control
- Use environment variables or secure configuration management
- The subscription key provides access to the secure API gateway
- Different models may have different usage limits and access controls

## Related Files

- `biomni/secure_llm.py` - Custom LangChain wrapper for the secure API
- `biomni/llm.py` - Main LLM configuration logic
- `biomni/config.py` - Configuration management
- `examples/use_secure_api.py` - Example scripts
