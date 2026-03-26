import sys
import os
import json

# Add src/TabuLLM to path
current_dir = os.getcwd()
if current_dir.endswith("TabuLLM") and "src" in current_dir:
    sys.path.append(current_dir)
elif "TabuLLM" in current_dir:
    sys.path.append(os.path.join(current_dir, "src", "TabuLLM"))
else:
    sys.path.append(os.path.join(current_dir, "src", "TabuLLM"))

print(f"Adding to path: {sys.path[-1]}")

from services.llm_client import LocalOpenAIProvider

class MockLocalProvider(LocalOpenAIProvider):
    def __init__(self):
        super().__init__()
        self.call_count = 0
        
    def _call_local_llm(self, messages, temperature=0.7, json_mode=False):
        self.call_count += 1
        print(f"\n[Mock] Call {self.call_count} with {len(messages)} messages.")
        
        # Determine the subset size based on last message content
        # Payload was json.dumps(subset)
        last_item = messages[-1]["content"]
        
        try:
             # Find index where JSON starts (usually immediately after '):\n')
             split_str = "):\n"
             if split_str in last_item:
                 json_str = last_item.split(split_str)[1]
             else:
                 json_str = last_item # fallback if it was num_empty only or full instructions
             
             subset = json.loads(json_str)
             count = len(subset) if isinstance(subset, list) else 1
        except Exception as e:
             # If instructions or num_empty only
             count = 1 

        print(f"[Mock] Target subset size received: {count}")
        
        # Return exact list size matching the subset to maintain length consistency
        data_list = [{"Age": 25}] * count
        response_content = json.dumps({"data": data_list})
        
        return {
            "choices": [{"message": {"content": response_content}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        }

def run_test():
    print("--- STARTING CHUNKING LOGIC REGRESSION TEST ---")
    provider = MockLocalProvider()
    
    # 35 rows payload -> with chunk_size=15 -> should loop EXACTLY 3 times
    # 15 + 15 + 5 = 35
    rows_payload = [{"Name": f"User_{i}"} for i in range(35)]
    context = {"metadata": "Test Metadata", "manual_ctx": "[]", "global_ctx": []}
    all_columns = ["Name", "Age"]
    algorithm = "Standard"
    instructions = "Complete Age Column."
    
    print("Executing generate_bulk_data (35 rows with chunk_size=15)...")
    result = provider.generate_bulk_data(
        rows_payload=rows_payload,
        context=context,
        all_columns=all_columns,
        algorithm=algorithm,
        instructions=instructions,
        temperature=0.7,
        num_empty=0,
        chunk_size=15
    )
    
    if not result:
        print("\n❌ FAILED: generate_bulk_data returned None.")
        sys.exit(1)

    print("\n--- TEST RESULTS ---")
    print(f"Total Call Count: {provider.call_count} (Expected: 3)")
    print(f"Total Data Rows Returned: {len(result['data'])} (Expected: 35)")
    
    if provider.call_count == 3 and len(result['data']) == 35:
        print("\n[SUCCESS] Chunking is active and works perfectly!")
        sys.exit(0)
    else:
        print("\n❌ FAILED: Unexpected iteration split sizes.")
        sys.exit(1)

if __name__ == "__main__":
    run_test()
