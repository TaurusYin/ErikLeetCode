import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments


def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return data


def json_to_sql(json_sql):
    sel = json_sql["sel"]
    agg = json_sql["agg"]
    conds = json_sql["conds"]

    select_clause = f"SELECT {agg_mapper[agg]}(col{sel})"
    where_clause = "WHERE " + " AND ".join([f"col{col} {op_mapper[op]} '{val}'" for col, op, val in conds])

    return select_clause + " " + where_clause


agg_mapper = {
    0: "",
    1: "AVG",
    2: "COUNT",
    3: "SUM",
    4: "MIN",
    5: "MAX",
}

op_mapper = {
    0: "=",
    1: ">",
    2: "<",
    3: "!=",
}


class NL2SQLDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item['question']
        sql = json_to_sql(item['sql'])

        encoding = self.tokenizer(question, sql, padding='max_length', truncation=True, max_length=self.max_length,
                                  return_tensors='pt', return_token_type_ids=False)
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        target_ids = encoding['input_ids'].squeeze(0)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': target_ids,
        }


# 载入预训练的T5模型和分词器
tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')

# 加载WikiSQL数据集
train_data = load_data('/Users/yineric/PycharmProjects/LeetCode/machine_learning/nlp/data/test.jsonl')
val_data = load_data('/Users/yineric/PycharmProjects/LeetCode/machine_learning/nlp/data/train.jsonl')

# 创建数据集和数据加载器
train_dataset = NL2SQLDataset(train_data, tokenizer)
val_dataset = NL2SQLDataset(val_data, tokenizer)
train_dataloader = DataLoader(train_dataset, batch_size=8)
val_dataloader = DataLoader(val_dataset, batch_size=8)

# 设置训练参数
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_steps=10,
    logging_dir='./logs',
    evaluation_strategy="epoch",
)

# 创建Trainer并开始训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()
print()

# 示例查询列表
queries = [
    "Find employees who are software engineers.",
    "How many employees are older than 30?",
    "List all products with a price greater than 100.",
    "Show me the total revenue for each region.",
]

# 使用训练过的模型和分词器
model.eval()

# 为每个查询生成相应的 SQL 语句
for query in queries:
    generated_sql = generate_sql(query, model, tokenizer)
    print(f"查询: {query}\n生成的 SQL 语句: {generated_sql}\n")
