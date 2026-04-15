from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
from tqdm import tqdm

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
# Path to the fine-tuned weights to be tested

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Load JSON file
with open('test.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Set termination tokens
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

# Store all results
results = []

# Process each sample sequentially
for idx, item in enumerate(tqdm(data, desc="Processing samples")):
    # Build messages format
    messages = [
        {"role": "system", "content": item["instruction"]},
        {"role": "user", "content": item["input"]}
    ]
    
    # Apply chat template
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    
    # Generate inference results
    outputs = model.generate(
        input_ids,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=False,  # Use greedy decoding for classification tasks
        temperature=0.1,   # Lower temperature for more deterministic results
        top_p=0.95,
    )
    
    # Decode response
    response = outputs[0][input_ids.shape[-1]:]
    predicted_output = tokenizer.decode(response, skip_special_tokens=True)
    
    # Save results
    result = {
        "index": idx,
        "instruction": item["instruction"],
        "input": item["input"],
        "ground_truth": item.get("output", ""),  # Include ground truth if available
        "predicted": predicted_output
    }
    results.append(result)
    
    # Optional: Print progress
    print(f"\n--- Sample {idx} ---")
    print(f"Input: {item['input'][:100]}...")
    print(f"Predicted: {predicted_output}")

# Save all prediction results to file
with open('predictions.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print(f"\nDone! Processed {len(results)} samples. Results saved to predictions.json")