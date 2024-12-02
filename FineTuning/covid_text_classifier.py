import torch
import pandas as pd
import transformers
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("training_data.csv")

x_train, x_eval, y_train, y_eval = train_test_split(dataset["text"], dataset["target"], train_size=0.8)

tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")

class TokenDataSet(Dataset):
    def __init__(self, data, labels, tokens):
        self.text_data = data
        self.tokens = tokens
        self.labels = list(labels)

    def __len__(self):
        return len(self.text_data)

    def __getitem__(self, idx):
        sample = {}
        for k, v in self.tokens.items():
            sample[k] = torch.tensor(v[idx])
        sample['labels'] = torch.tensor(self.labels[idx])
        return sample

batch_size = 32

train_tokens = tokenizer(list(x_train), padding=True, truncation=True)
train_dataset = TokenDataSet(x_train, y_train, train_tokens)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

eval_tokens = tokenizer(list(x_eval), padding=True, truncation=True)
eval_dataset = TokenDataSet(x_eval, y_eval, eval_tokens)
eval_loader = DataLoader(eval_dataset, shuffle=False, batch_size=batch_size)

model = transformers.BertForSequenceClassification.from_pretrained("bert-base-cased")
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
loss_function = torch.nn.CrossEntropyLoss()

epochs = 1
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def train():
    model.train()

    for batch_index, batch in enumerate(train_loader):
        device_batch = {key : tensor.to(device) for key, tensor in batch.items()}
        optimizer.zero_grad()

        outputs = model(input_ids=device_batch["input_ids"], attention_mask=device_batch["attention_mask"])
        loss = loss_function(outputs.logits, device_batch["labels"])
        loss.backward()
        optimizer.step()

        print(f"Training Batch {batch_index + 1} / {len(train_loader)}: Loss of {loss.item() / len(device_batch)}")

def eval():
    model.eval()

    num_correct = 0
    total_logits = 0
    for batch_index, batch in enumerate(eval_loader):
        device_batch = {key : tensor.to(device) for key, tensor in batch.items()}

        with torch.no_grad():
            outputs = model(input_ids=device_batch["input_ids"], attention_mask=device_batch["attention_mask"])

        batch_correct = (outputs.logits.argmax(1) == device_batch["labels"]).sum().item()
        num_correct += batch_correct
        total_logits += len(outputs.logits)

        print(f"Evaluation Batch {batch_index + 1} / {len(eval_loader)}: Accuracy of {batch_correct / len(outputs.logits)}")

    print(f"Final Accuracy of {num_correct / total_logits}")

# do the training !!
for epoch in range(epochs):
    print(f"Training Epoch {epoch + 1}:")
    train()

    print(f"Evaluating Epoch {epoch + 1}:")
    eval()

print("Finished Training!\n")

def is_covid_sentence(text):
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    device_tokens = {key : tensor.to(device) for key, tensor in tokens.items()}

    model.eval()
    with torch.no_grad():
        outputs = model(**device_tokens)

    return outputs.logits.argmax(dim=1).item() == 1

while True:
    text = input("Enter a sentence to be classified as covid-related or not:\n")

    print("This is covid related." if is_covid_sentence(text) else "This is not covid related.")