# File: backend/scripts/train_model.py

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer, DefaultDataCollator

# --- 1. CONFIGURATION ---
# All project constants are defined in this section.
# This makes the script easy to adapt for different models or datasets.

MODEL_NAME = "dmis-lab/biobert-base-cased-v1.1"
DATASET_NAME = "pubmed_qa"

# CORRECTED: Use the full, valid configuration name as discovered from the error message.
DATASET_CONFIG = "pqa_labeled" 

# Define the output path for the final model artifact.
# '..' navigates up one level from 'scripts' to the 'backend' root directory.
BEST_MODEL_DIR = "../trained_models/biobert"


# --- 2. DATA PREPROCESSING ---

# Initialize the tokenizer globally. It will be used in the preprocessing
# function and passed to the Trainer.
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess_function(examples):
    """
    This function tokenizes the question and context, and then identifies the
    start and end token positions of the answer within the tokenized input.
    """
    # Extract the text fields from the dataset, handling the nested structure of the context.
    questions = [q.strip() for q in examples["question"]]
    contexts = [c['contexts'][0] for c in examples["context"]]
    
    # The 'answers' format requires the text and the character start position.
    answers = [
        {"text": ans, "answer_start": contexts[i].find(ans)}
        for i, ans in enumerate(examples["long_answer"])
    ]

    # Tokenize the question and context together.
    # `return_offsets_mapping` is essential for mapping token indices back to character indices.
    inputs = tokenizer(
        questions, contexts, max_length=512, truncation="only_second",
        return_offsets_mapping=True, padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    start_positions, end_positions = [], []

    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
        start_char = answer["answer_start"]
        end_char = start_char + len(answer["text"])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context within the full tokenized sequence.
        # The context corresponds to sequence_id 1.
        context_start = sequence_ids.index(1)
        context_end = len(sequence_ids) - 1 - sequence_ids[::-1].index(1)

        # If the answer is not fully contained within the context span, label it as (0, 0).
        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise, find the token indices that correspond to the answer's start and end characters.
            token_start_index = context_start
            while token_start_index <= context_end and offset[token_start_index][0] < start_char:
                token_start_index += 1
            start_positions.append(token_start_index)

            token_end_index = context_end
            while token_end_index >= token_start_index and offset[token_end_index][1] > end_char:
                token_end_index -= 1
            end_positions.append(token_end_index)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs


# --- 3. MAIN TRAINING LOGIC ---

def main():
    # Step 3.1: Load and Process the Dataset
    print("Step 1/4: Loading and preparing the dataset...")
    # Load the dataset using the corrected configuration name.
    raw_dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split='train').shuffle(seed=42).select(range(5000))
    train_test_split = raw_dataset.train_test_split(test_size=0.2)
    
    # The .map() function efficiently applies our preprocessing logic to the entire dataset.
    tokenized_datasets = train_test_split.map(preprocess_function, batched=True, remove_columns=raw_dataset.column_names)

    # Step 3.2: Load the Pre-trained Model
    print("Step 2/4: Loading the pre-trained BioBERT model...")
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)

    # Step 3.3: Configure and Run Training
    print("Step 3/4: Configuring and starting the training process...")
    # `TrainingArguments` is a class that contains all the customizable hyperparameters.
    training_args = TrainingArguments(
        output_dir="./results",              # Directory to save training checkpoints
        evaluation_strategy="epoch",         # Run evaluation at the end of each epoch
        learning_rate=2e-5,
        per_device_train_batch_size=8,       # Batch size per GPU
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        save_strategy="epoch",
        load_best_model_at_end=True,         # Automatically load the best model when training is done
    )

    # The `Trainer` object abstracts away the complexity of the PyTorch training loop.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        data_collator=DefaultDataCollator(), # Handles creating batches of data
    )
    
    print("Beginning model training...")
    trainer.train()

    # Step 3.4: Save the Best Performing Model
    print(f"Step 4/4: Saving the best model to {BEST_MODEL_DIR}...")
    trainer.save_model(BEST_MODEL_DIR)
    print("Training complete and model successfully saved!")

if __name__ == "__main__":
    main()


