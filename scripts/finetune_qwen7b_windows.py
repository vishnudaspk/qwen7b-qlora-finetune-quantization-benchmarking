# scripts/finetune_qwen7b_windows_bits.py
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# -------------------------------
# CONFIGURATION
# -------------------------------
BIT_WIDTH = 8  # Choose 2, 4, or 8
MODEL_NAME = "Qwen/Qwen-7B"
OUTPUT_DIR = f"./outputs/qwen7b_lora_{BIT_WIDTH}bit"
BATCH_SIZE = 1
GRAD_ACCUM = 32
MAX_SAMPLES = 1000
MAX_LENGTH = 128
EPOCHS = 2
LEARNING_RATE = 2e-4

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*60)
print(f"Qwen-7B Fine-tuning (Windows {BIT_WIDTH}-bit QLoRA Mode)")
print("="*60)
print(f"\nTraining Configuration:")
print(f"  - Samples: {MAX_SAMPLES}")
print(f"  - Sequence Length: {MAX_LENGTH}")
print(f"  - Batch Size: {BATCH_SIZE}")
print(f"  - Gradient Accumulation: {GRAD_ACCUM}")
print(f"  - Effective Batch Size: {BATCH_SIZE * GRAD_ACCUM}")
print(f"  - Epochs: {EPOCHS}")

# -------------------------------
# ENVIRONMENT / GPU CHECK
# -------------------------------
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if not torch.cuda.is_available():
    print("ERROR: CUDA not available!")
    exit(1)

print(f"\nCUDA Version: {torch.version.cuda}")
print(f"PyTorch Version: {torch.__version__}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# -------------------------------
# 1. Load Dataset
# -------------------------------
print("\n" + "-"*60)
print("Loading Dataset")
print("-"*60)
ds = load_dataset("databricks/databricks-dolly-15k", split="train")
train_ds = ds.select(range(min(MAX_SAMPLES, len(ds))))
print(f"Loaded {len(train_ds)} samples")

# -------------------------------
# 2. Load Tokenizer
# -------------------------------
print("\n" + "-"*60)
print("Loading Tokenizer")
print("-"*60)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '<|endoftext|>'})
    tokenizer.pad_token = '<|endoftext|>'
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('<|endoftext|>')
tokenizer.padding_side = "right"
print(f"Vocab size: {len(tokenizer)}")
print(f"Pad token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")

# -------------------------------
# 3. Tokenize Dataset
# -------------------------------
print("\n" + "-"*60)
print("Tokenizing Dataset")
print("-"*60)
def format_prompt(example):
    instruction = example.get("instruction", "").strip()
    context = example.get("context", "").strip()
    response = example.get("response", "").strip()
    if context:
        return f"### Instruction:\n{instruction}\n\n### Context:\n{context}\n\n### Response:\n{response}"
    else:
        return f"### Instruction:\n{instruction}\n\n### Response:\n{response}"

def tokenize_function(examples):
    prompts = [format_prompt(ex) for ex in [
        {"instruction": i, "context": c, "response": r}
        for i, c, r in zip(examples["instruction"], examples["context"], examples["response"])
    ]]
    outputs = tokenizer(prompts, truncation=True, max_length=MAX_LENGTH, padding="max_length", return_tensors=None)
    outputs["labels"] = outputs["input_ids"].copy()
    return outputs

tokenized_dataset = train_ds.map(tokenize_function, batched=True, remove_columns=train_ds.column_names, desc="Tokenizing")
print(f"Tokenization complete - {len(tokenized_dataset)} samples")

# -------------------------------
# 4. Load Model (Quantization)
# -------------------------------
print("\n" + "-"*60)
print(f"Loading Model ({BIT_WIDTH}-bit QLoRA)")
print("-"*60)
torch.cuda.empty_cache()
free_memory = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1024**3
print(f"Free GPU Memory: {free_memory:.2f} GB")

# BitsAndBytes configuration
if BIT_WIDTH == 2:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=False,
        llm_int8_threshold=0.0,
    )
elif BIT_WIDTH == 4:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=False,
    )
elif BIT_WIDTH == 8:
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_use_double_quant=True,
        bnb_8bit_compute_dtype=torch.bfloat16,
    )
else:
    raise ValueError("BIT_WIDTH must be 2, 4, or 8")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map={"": 0},
    trust_remote_code=True,
)
model.config.use_cache = False
model.config.pad_token_id = tokenizer.pad_token_id
print(f"SUCCESS: Model loaded in {BIT_WIDTH}-bit quantization")

# -------------------------------
# 5. Apply LoRA
# -------------------------------
print("\n" + "-"*60)
print("Applying LoRA")
print("-"*60)
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["c_attn", "c_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
))
model.print_trainable_parameters()
model.train()

# -------------------------------
# 6. Data Collator
# -------------------------------
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# -------------------------------
# 7. Training Arguments
# -------------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    warmup_steps=20,
    num_train_epochs=EPOCHS,
    fp16=True,
    gradient_checkpointing=True,
    optim="adamw_torch",
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=1,
    dataloader_num_workers=0,
    report_to="none",
    disable_tqdm=False,
    ddp_find_unused_parameters=False,
)
print("Training arguments configured")

# -------------------------------
# 8. Trainer
# -------------------------------
trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_dataset, tokenizer=tokenizer, data_collator=data_collator)
print("Trainer initialized successfully")

# -------------------------------
# 9. Train
# -------------------------------
print("\n" + "="*60)
print("STARTING TRAINING")
print("="*60)
torch.cuda.empty_cache()
used_memory = torch.cuda.memory_allocated(0) / 1024**3
print(f"GPU Memory before training: {used_memory:.2f} GB")
try:
    trainer.train()
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("="*60)
except KeyboardInterrupt:
    print("\n\nTraining interrupted by user")
except Exception as e:
    print(f"\n\nTraining failed with error: {str(e)}")
    import traceback
    traceback.print_exc()
    raise

# -------------------------------
# 10. Save Model
# -------------------------------
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Model saved to: {OUTPUT_DIR}")
if torch.cuda.is_available():
    peak_memory = torch.cuda.max_memory_allocated(0) / 1024**3
    print(f"\nPeak GPU Memory: {peak_memory:.2f} GB")
print("\n" + "="*60)
print("FINE-TUNING COMPLETE!")
print("="*60)
