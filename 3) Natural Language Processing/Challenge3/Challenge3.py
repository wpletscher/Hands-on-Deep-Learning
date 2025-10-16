from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np


MODEL_NAME = 'distilbert-base-uncased'
MODEL_NAME = 'bert-base-uncased'

training_args = TrainingArguments(
  output_dir="./results",
  learning_rate=2e-5,
  per_device_train_batch_size=16,
  num_train_epochs=3,
  weight_decay=0.05,
  # logging_steps=100,
  # logging_dir="./logs"
)



def compute_metrics(eval_pred):
  logits, labels = eval_pred
  predictions = np.argmax(logits, axis=-1)
  accuracy = np.mean(predictions == labels)
  return {"accuracy": accuracy}



# Do not change function signature
def preprocess_function(examples):
  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
  result = tokenizer(
    examples["text"], 
    truncation=True, 
    #padding=True, 
    #max_length=512
  )
  return result



# Do not change function signature
def init_model():
  model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
  return model



# Do not change function signature
def train_model(model, train_dataset):
  trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    compute_metrics=compute_metrics
  )
  trainer.train()
  return model
