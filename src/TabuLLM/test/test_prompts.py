import sys
import os
import pandas as pd

# Add the project root to sys.path
sys.path.append(os.getcwd())

from services.prompt_manager import prompt_manager
from services.llm_client import LocalOpenAIProvider

def test_prompts():
    print("Testing PromptManager...")
    df = pd.DataFrame({'A': [1, 2], 'B': ['x', 'y']})
    
    # Test dataset description prompt
    prompt = prompt_manager.get_prompt(
        "dataset_description",
        provided_metadata="Test metadata",
        df_sample_str=df.to_string(index=False)
    )
    print("\nDataset Description Prompt Snippet:")
    print(prompt[:200] + "...")
    
    # Test target detection prompt
    prompt = prompt_manager.get_prompt(
        "target_detection",
        df_sample_str=df.to_string(index=False)
    )
    print("\nTarget Detection Prompt Snippet:")
    print(prompt[:200] + "...")
    
    # Test provider initialization
    print("\nTesting Provider Initialization...")
    try:
        provider = LocalOpenAIProvider()
        print("LocalOpenAIProvider initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize provider: {e}")

if __name__ == "__main__":
    test_prompts()
