from transformers import MarianTokenizer, MarianMTModel, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model
from datasets import Dataset
import pandas as pd
import torch
from pathlib import Path

def trainModel(train_dataset_path: str, output_dir: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = 'Helsinki-NLP/opus-mt-hi-en'
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name).to(device)
    print("Model initialized...")

    # Load and prepare dataset
    df = pd.read_csv(train_dataset_path)
    df = df[['Chhattisgarhi', 'English']].dropna()
    df = df.rename(columns={'Chhattisgarhi': 'translation_source', 'English': 'translation_target'})
    hf_dataset = Dataset.from_pandas(df)

    # Tokenization
    def preprocess_function(batch):
        inputs = tokenizer(batch["translation_source"], padding="max_length", truncation=True, max_length=128)
        targets = tokenizer(batch["translation_target"], padding="max_length", truncation=True, max_length=128)

        inputs["labels"] = targets["input_ids"]
        return inputs

    hf_dataset = hf_dataset.map(preprocess_function, batched=True)
    hf_dataset = hf_dataset.train_test_split(test_size=0.2)
    train_data = hf_dataset["train"]
    eval_data = hf_dataset["test"]

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        predict_with_generate=True,
        num_train_epochs=5,
        logging_steps=10,
        save_steps=100,
        eval_steps=100,  # Use eval_steps instead of evaluation_strategy
        learning_rate=2e-5,
        save_total_limit=2,
        report_to="none"
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer)
    )

    trainer.train()

    # Save the model & tokenizer
    model.save_pretrained(output_dir / "cg_eng_small")
    tokenizer.save_pretrained(output_dir / "cg_eng_small")
    print("Model saved at:", output_dir / "mocg_eng_small")
