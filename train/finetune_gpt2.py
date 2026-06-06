import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BASE_DIR, DATA_DIR, MODELS_DIR

def main():
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))
    
    # Freeze base transformer blocks, train only LM head
    for param in model.transformer.parameters():
        param.requires_grad = False
        
    model.gradient_checkpointing_enable()
    
    data_path = os.path.join(DATA_DIR, 'gpt2_finetune_data.jsonl')
    dataset = load_dataset('json', data_files=data_path, split='train')
    
    def tokenize_function(examples):
        texts = [f"{p} {c}{tokenizer.eos_token}" for p, c in zip(examples["prompt"], examples["completion"])]
        encodings = tokenizer(texts, truncation=True, padding="max_length", max_length=128)
        encodings["labels"] = encodings["input_ids"].copy()
        return encodings
        
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    out_dir = os.path.join(MODELS_DIR, 'gpt2_finetuned')
    
    training_args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        save_steps=100,
        save_total_limit=2,
        logging_steps=10,
        learning_rate=5e-5,
        fp16=torch.cuda.is_available(),
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )
    
    print("[Phase 3] Fine-tuning GPT-2...")
    trainer.train()
    
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"Saved fine-tuned GPT-2 to {out_dir}")

if __name__ == "__main__":
    main()
