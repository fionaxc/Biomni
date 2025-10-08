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

prompt = "You are an expert in rare disease diagnosis. I will provide a list of Human Phenotype Ontology (HPO) terms describing a patient's symptoms, \n along with a list of candidate genes. Using your knowledge of genetics, known disease-gene associations, and variant interpretation, \n generate a ranked list of all of the candidate genes based on their likelihood of causing the patient's symptoms. \n Only output a valid jsonl file with no spaces between each json. Rank every candidate gene according to its association with the HPO terms, \n known gene-disease relationships, and functional impact. Make sure to rank all candidate genes in the list.\n\n The output must be in JSON Lines (jsonl) format with NO markdown formatting. Each line must be a valid JSON object containing exactly these fields:\n - gene_name: string\n - rank: number\n - explanation: string\n\n Example format:\n {\"gene_name\":\"GENE1\",\"rank\":1,\"explanation\":\"Explanation here\"}\n {\"gene_name\":\"GENE2\",\"rank\":2,\"explanation\":\"Explanation here\"}\n\n Do not include any other text, markdown formatting, or code block markers in your response. Only output the JSON lines.\n\n Patient HPO terms:\n [Hypoplasia of the maxilla, Abnormal cochlea morphology, Blue sclerae, Decreased response to growth hormone stimuation test, Prominent scalp veins, Global developmental delay, Growth delay, Talipes equinovarus, Bilateral talipes equinovarus, Apraxia, Small foramen magnum, Abnormal sella turcica morphology, Abnormality of the skull base, Short stature, Dilatated internal auditory canal, Relative macrocephaly, Dilated third ventricle, Congenital bilateral ptosis, Lamellar cataract, Cervical spinal canal stenosis, Bilateral sensorineural hearing impairment, Aplasia/Hypoplasia of the mandible, Aplasia/Hypoplasia of the middle phalanges of the hand, Short distal phalanx of finger, Speech apraxia, Prominent forehead, Dilated vestibule of the inner ear, Enlarged semicircular canal]\n\n Candidate genes to rank:\n [UNC13C, ERCC6, MAN2B1, LAMC1, EP300, LAMC3, MYO15A, WFS1, KRTAP10-3, ATM, TSPEAR, LRSAM1, BRWD3, BOD1, STX3, RBMX, COG4, CELSR1, COL11A1, KRTAP10-7, RYR3, ESRRB, POLG, ITIH6, LAMB1, KIF1A, UNC5C, R3HCC1L, FYCO1, MC1R, ADGRV1, DNAH2, DHCR7, TTI1, HSPG2, MFF, APOD, LRRK1, ALDOA, PEX6]\n"

def test_option_1_custom_full_url():
    """Option 1: Custom source with full URL"""
    print("\n" + "="*80)
    print("TESTING OPTION 1: Custom source with full URL")
    print("="*80)

    default_config.source = "Custom"
    default_config.base_url = "https://apim.stanfordhealthcare.org/openai-eastus2/deployments/gpt-5/chat/completions?api-version=2024-12-01-preview"
    default_config.api_key = os.environ.get("OPENAI_API_KEY")
    default_config.llm = "gpt-5"

    try:
        agent = A1()
        print("✓ Agent created successfully")

        # Test with a simple query
        result = agent.go(prompt)
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
    default_config.llm = "gpt-5"

    try:
        agent = A1()
        print("✓ Agent created successfully")

        result = agent.go(prompt)
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

    default_config.source = "AzureOpenAI"
    default_config.llm = "gpt-5"

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


def test_option_4_secure_api():
    """Option 4: SecureAPI source (designed for Ocp-Apim-Subscription-Key)"""
    print("\n" + "="*80)
    print("TESTING OPTION 4: SecureAPI source")
    print("="*80)

    default_config.source = "SecureAPI"
    default_config.secure_api_url = "https://apim.stanfordhealthcare.org/openai-eastus2/deployments/gpt-5/chat/completions?api-version=2024-12-01-preview"
    default_config.secure_model_id = "gpt-5"
    default_config.api_key = os.environ.get("OPENAI_API_KEY")

    try:
        agent = A1()
        print("✓ Agent created successfully")

        result = agent.go(prompt)
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

    url = "https://apim.stanfordhealthcare.org/openai-eastus2/deployments/gpt-5/chat/completions?api-version=2024-12-01-preview"

    # Common Headers (Used for all models)
    headers = {'Ocp-Apim-Subscription-Key': os.environ.get("OPENAI_API_KEY"), 'Content-Type': 'application/json'}


    payload = json.dumps({
        "model": "gpt-5",
        "messages": [{"role": "user", "content": prompt}]
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
        print("found key")
#     # Test direct API call first to verify credentials
#     print("\n" + "="*80)
#     print("STEP 1: Testing direct API access")
#     print("="*80)
#     direct_works = test_direct_api_call()

#     if not direct_works:
#         print("\n⚠️  Direct API call failed. Please fix credentials before testing Biomni.")
#         print("\nThings to check:")
#         print("1. Is your API key correct?")
#         print("2. What header does your API expect? (api-key, Authorization, etc.)")
#         print("3. Is the URL exactly correct?")
#         return

    # Test Biomni configurations
    print("\n" + "="*80)
    print("STEP 2: Testing Biomni configurations")
    print("="*80)

    options = [
        ("Option 3: AzureOpenAI with env vars", test_option_3_azure_endpoint),
        ("Option 2: Custom Base URL", test_option_2_custom_base_url),
        ("Option 1: Custom Full URL", test_option_1_custom_full_url),
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
