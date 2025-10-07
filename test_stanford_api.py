"""
Test script for configuring Biomni with Stanford Healthcare Azure OpenAI endpoint.
"""
import os
import json
from biomni.config import default_config
from biomni.agent import A1

# ============================================================================
# Configuration Options - Try them in order until one works
# ============================================================================

def test_option_1_custom_full_url():
    """Option 1: Custom source with full URL"""
    print("\n" + "="*80)
    print("TESTING OPTION 1: Custom source with full URL")
    print("="*80)

    default_config.source = "Custom"
    default_config.base_url = "https://apim.stanfordhealthcare.org/openai-eastus2/deployments/gpt-4.1/chat/completions?api-version=2025-01-01-preview"
    default_config.api_key = os.environ.get("OPENAI_API_KEY")
    default_config.llm = "gpt-4.1"

    try:
        agent = A1()
        print("✓ Agent created successfully")

        # Test with a simple query
        result = agent.go("What is 2+2?")
        print(f"✓ Test query successful!")
        print(f"Response: {result}")
        return True
    except Exception as e:
        print(f"✗ Failed with error: {e}")
        return False


def test_option_2_custom_base_url():
    """Option 2: Custom source with base URL only"""
    print("\n" + "="*80)
    print("TESTING OPTION 2: Custom source with base URL (no deployment path)")
    print("="*80)

    default_config.source = "Custom"
    default_config.base_url = "https://apim.stanfordhealthcare.org/openai-eastus2"
    default_config.api_key = os.environ.get("OPENAI_API_KEY")
    default_config.llm = "gpt-4.1"

    try:
        agent = A1()
        print("✓ Agent created successfully")

        result = agent.go("What is 2+2?")
        print(f"✓ Test query successful!")
        print(f"Response: {result}")
        return True
    except Exception as e:
        print(f"✗ Failed with error: {e}")
        return False


def test_option_3_azure_endpoint():
    """Option 3: Azure source with endpoint environment variable"""
    print("\n" + "="*80)
    print("TESTING OPTION 3: Azure source with OPENAI_ENDPOINT")
    print("="*80)

    os.environ["OPENAI_ENDPOINT"] = "https://apim.stanfordhealthcare.org/openai-eastus2"
    os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")

    default_config.source = "Azure"
    default_config.llm = "azure-gpt-4.1"

    try:
        agent = A1()
        print("✓ Agent created successfully")

        result = agent.go("What is 2+2?")
        print(f"✓ Test query successful!")
        print(f"Response: {result}")
        return True
    except Exception as e:
        print(f"✗ Failed with error: {e}")
        return False


def test_direct_api_call():
    """Test direct API call to verify credentials work"""
    print("\n" + "="*80)
    print("TESTING: Direct API call (to verify your credentials)")
    print("="*80)

    import requests

    url = "https://apim.stanfordhealthcare.org/openai-eastus2/deployments/gpt-4.1/chat/completions?api-version=2025-01-01-preview"

    # Common Headers (Used for all models)
    headers = {'Ocp-Apim-Subscription-Key': os.environ.get("OPENAI_API_KEY"), 'Content-Type': 'application/json'}
    

    payload = json.dumps({
        "model": "gpt-4.1",
        "messages": [{"role": "user", "content": "What is 2+2?"}]
    })

    try:
        response = requests.post(url, headers=headers, data=payload)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")

        if response.status_code == 200:
            print("✓ Direct API call successful! Your credentials work.")
            return True
        else:
            print("✗ Direct API call failed. Check your credentials and URL.")
            return False
    except Exception as e:
        print(f"✗ Failed with error: {e}")
        return False


# ============================================================================
# Main Test Runner
# ============================================================================

def main():
    print("="*80)
    print("Stanford Healthcare Azure OpenAI - Biomni Configuration Test")
    print("="*80)

    # Check if API key is set
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        print("\n⚠️  WARNING: STANFORD_API_KEY environment variable not set!")
        print("Please set it with: export OPENAI_API_KEY='your_actual_key'")
        print("Or modify this script to include your API key directly")
        print("\nContinuing with placeholder key for testing configuration...")
    else:
        print(f"\n✓ Found STANFORD_API_KEY: {api_key[:10]}...")

    # Test direct API call first to verify credentials
    print("\n" + "="*80)
    print("STEP 1: Testing direct API access")
    print("="*80)
    direct_works = test_direct_api_call()

    if not direct_works:
        print("\n⚠️  Direct API call failed. Please fix credentials before testing Biomni.")
        print("\nThings to check:")
        print("1. Is your API key correct?")
        print("2. What header does your API expect? (api-key, Authorization, etc.)")
        print("3. Is the URL exactly correct?")
        return

    # Test Biomni configurations
    print("\n" + "="*80)
    print("STEP 2: Testing Biomni configurations")
    print("="*80)

    options = [
        ("Option 1: Full URL", test_option_1_custom_full_url),
        ("Option 2: Base URL", test_option_2_custom_base_url),
        ("Option 3: Azure with env vars", test_option_3_azure_endpoint),
    ]

    for name, test_func in options:
        success = test_func()
        if success:
            print(f"\n🎉 SUCCESS! {name} works!")
            print("\nYou can use this configuration in your scripts.")
            return

    print("\n" + "="*80)
    print("All options failed. Additional debugging needed.")
    print("="*80)
    print("\nNext steps:")
    print("1. Check biomni/llm.py to see how it constructs API calls")
    print("2. Verify what headers your Stanford API expects")
    print("3. You may need to modify Biomni's get_llm() function")


if __name__ == "__main__":
    main()
