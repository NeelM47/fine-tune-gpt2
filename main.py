from datasets import load_dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModelForCausalLM, pipeline
import torch

# Load dataset
dataset = load_dataset("text", data_files="data/dataset.txt")

# Load tokenizer and set padding token
model_name = "gpt2"  # Base model
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Set padding token to EOS token

# Load and configure model
model = AutoModelForCausalLM.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))  # Resize token embeddings

# Tokenization function for dataset preparation
def tokenize_function(examples):
    tokenized = tokenizer(examples["text"], padding="max_length", truncation=True)
    return {
        'input_ids': tokenized["input_ids"],
        'labels': tokenized["input_ids"],  # Set 'labels' for language modeling
    }

# Apply tokenization
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Split into training and validation datasets
train_test_split = tokenized_dataset['train'].train_test_split(test_size=0.1)  # 90% train, 10% validation
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",               # Output directory
    eval_strategy="epoch",                # Evaluate every epoch
    learning_rate=2e-5,                   # Learning rate
    per_device_train_batch_size=1,        # Training batch size
    per_device_eval_batch_size=1,         # Evaluation batch size
    num_train_epochs=3,                   # Number of epochs
    weight_decay=0.01,                    # Weight decay for regularization
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Train and evaluate the model
torch.cuda.empty_cache()  # Clear cache to manage memory
trainer.train()
trainer.evaluate()

# Save the fine-tuned model and tokenizer
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

# Load the fine-tuned model for text generation
generator = pipeline('text-generation', model="./fine_tuned_model")  # Set device parameter if needed

# Generate text from a prompt
prompt = "What do you think should be my name?"
output = generator(prompt, max_length=100)
print(output)

