import os
import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

try:
    from qwen.tokenization_qwen import QWenTokenizer
except ImportError:
    pass

# ============================================
# CONFIGURATION
# ============================================
BASE_MODEL = "Qwen/Qwen-7B"

LORA_PATHS = {
    "2-bit": "./outputs/qwen7b_lora_2bit",
    "4-bit": "./outputs/qwen7b_lora_4bit",
    "8-bit": "./outputs/qwen7b_lora_8bit",
}

# ============================================
# HELPER FUNCTIONS
# ============================================
def load_lora_model(lora_choice):
    """Load base model + LoRA weights safely."""
    if lora_choice not in LORA_PATHS:
        print(f"❌ Invalid choice: {lora_choice}")
        print(f"Available options: {list(LORA_PATHS.keys())}")
        return None, None, None

    print(f"\n🔹 Loading base model '{BASE_MODEL}' ...")
    bit_width = int(lora_choice.split("-")[0])

    # Use BitsAndBytesConfig for quantized loading
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=(bit_width == 4),
        load_in_8bit=(bit_width == 8),
        llm_int8_threshold=6.0,
        bnb_4bit_compute_dtype=torch.float16,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🧠 Using device: {device.upper()}")

    # Load model
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config if bit_width in [4, 8] else None,
        torch_dtype=torch.float16,
        device_map=None if device == "cpu" else "auto",
        trust_remote_code=True,
    )

    print(f"🔹 Applying LoRA weights from: {LORA_PATHS[lora_choice]}")
    model = PeftModel.from_pretrained(base_model, LORA_PATHS[lora_choice], is_trainable=False)
    model = model.merge_and_unload()  # merge LoRA into base model
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    return model, tokenizer, device


def chat_with_model(model, tokenizer, device):
    """Interactive chat loop."""
    print("\n💬 Chat mode activated! Type 'exit' to quit.\n")

    while True:
        question = input("🧠 You: ").strip()
        if question.lower() in {"exit", "quit"}:
            print("👋 Exiting chat...")
            break

        start_time = time.time()
        inputs = tokenizer(question, return_tensors="pt").to(device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(output[0], skip_special_tokens=True)
        elapsed = time.time() - start_time

        print(f"\n🤖 Qwen ({elapsed:.2f}s): {response}\n")


# ============================================
# MAIN EXECUTION
# ============================================
def main():
    print("\n==============================")
    print(" Qwen-7B LoRA Chat Interface ")
    print("==============================\n")

    print("Available fine-tuned models:")
    for key in LORA_PATHS:
        print(f"  - {key}")

    choice = input("\nSelect LoRA model (2-bit / 4-bit / 8-bit): ").strip()
    model, tokenizer, device = load_lora_model(choice)

    if model is None:
        return

    print(f"\n✅ Loaded {choice} fine-tuned Qwen-7B successfully on {device.upper()}!")
    chat_with_model(model, tokenizer, device)


if __name__ == "__main__":
    main()
