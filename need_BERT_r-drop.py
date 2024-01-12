import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
torch.manual_seed(42)
from transformers import set_seed, BertForSequenceClassification, BertTokenizer, TrainingArguments, Trainer
set_seed(42)
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, ndcg_score
from sklearn.model_selection import train_test_split
import os
os.chdir('/xxxx/xxxx/xxxx')

df = pd.read_excel('novel_generate_v2.xlsx', keep_default_na=False)
text_values = df.text.values.tolist()
label_values = df.label.values.tolist()
# Define pretrained tokenizer and model
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
# Define Trainer parameters
BATCH_SIZE = 16
max_len = 512
LR = 5e-5
EPOCH = 3
alpha = 2
random_state = 111

# ----- 1. Preprocess data -----#
# Preprocess data
X_train_ori, X_test, y_train_ori, y_test = train_test_split(text_values, label_values, shuffle=True, test_size=0.4, random_state=random_state)
X_train, X_val, y_train, y_val = train_test_split(X_train_ori, y_train_ori, shuffle=True, test_size=1/3, random_state=random_state)
X_train_tokenized = tokenizer(X_train, padding='max_length', truncation=True, max_length=max_len)
X_test_tokenized = tokenizer(X_test, padding='max_length', truncation=True, max_length=max_len)

# Create torch dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.encodings["input_ids"])

train_dataset = Dataset(X_train_tokenized, y_train)
val_dataset = Dataset(X_test_tokenized, y_test)

# ----- 2. Fine-tune pretrained model -----#
def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=-1)
    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    return {"accuracy": accuracy}
# Define Trainer
class CustomTrainer(Trainer):
    def compute_kl_loss(self, p, q):
        p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
        q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')
        p_loss = p_loss.sum()
        q_loss = q_loss.sum()
        loss = (p_loss + q_loss) / 2
        return loss
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs_1 = model(**inputs)
        outputs_2 = model(**inputs)      
        logits_1 = outputs_1.get('logits')
        logits_2 = outputs_2.get('logits')
        loss_fct = nn.CrossEntropyLoss() 
        ce_loss = 0.5 * (loss_fct(logits_1, labels) + loss_fct(logits_2, labels))
        kl_loss = self.compute_kl_loss(logits_1, logits_2)
        loss = ce_loss + kl_loss * alpha
        return (loss, outputs_1) if return_outputs else loss

args = TrainingArguments(
    output_dir="output",
    evaluation_strategy="epoch",
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LR,
    num_train_epochs=EPOCH,
    save_strategy='epoch',
    fp16=True,
	weight_decay=0.01
)
trainer = CustomTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)
# Train pre-trained model
trainer.train()

# ----- 3. Predict -----#
# Load test data
X_test_tokenized = tokenizer(X_test, padding='max_length', truncation=True, max_length=max_len)

# Create torch dataset
test_dataset = Dataset(X_test_tokenized)

# Make prediction
raw_pred, _, _ = trainer.predict(test_dataset)

# Preprocess raw predictions
y_pred = np.argmax(raw_pred, axis=1)
print(classification_report(y_test, y_pred, target_names=["Common","Unique"], digits=4))
print(confusion_matrix(y_test, y_pred))
df_rank = pd.DataFrame({'y_pred':raw_pred[:,1], 'y_test':y_test})
df_rank.sort_values(by=['y_pred'], ascending=False, inplace=True)
P = df_rank.value_counts(['y_test'])[1]
for top_k in [5,10,15,25,50,75,100,150,200,250,300]:
    if df_rank[:top_k].y_test.sum()==0: 
        print('k =', top_k, '  %.4f'%0, '  %.4f'%0, '  %.4f'%0, '  %.4f'%0)
        continue
    TP = df_rank[:top_k].value_counts(['y_test'])[1]
    precision = TP / top_k
    recall = TP / P
    f1 = 2/(1/precision+1/recall)
    ndcg = ndcg_score([df_rank.y_test.values.tolist()], [df_rank.y_pred.values.tolist()], k=top_k)
    print('k =', top_k, '  %.4f'%precision, '  %.4f'%recall, '  %.4f'%f1, '  %.4f'%ndcg)
