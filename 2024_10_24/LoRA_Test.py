
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from peft import get_peft_config, LoraConfig, get_peft_model

# 加载公开数据集 - 这里使用 IMDb 电影评论分类数据集
# Load public dataset - using the IMDb movie review dataset
dataset = load_dataset("imdb")

# 加载小模型 - DistilBERT
# Load small model - DistilBERT
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# 配置LoRA设置，设定低秩矩阵的秩 r 和微调参数 alpha
# Configure LoRA settings - setting rank r and alpha for fine-tuning
peft_config = LoraConfig(
    r=8,  # 低秩矩阵的秩 - Rank of the low-rank matrices
    lora_alpha=16,  # LoRA的缩放参数 - Scaling parameter for LoRA
    target_modules=["query", "value"],  # 目标模块 - Which layers to fine-tune
    lora_dropout=0.1,  # Dropout 用于正则化 - Dropout for regularization
)

# 将LoRA集成到模型中
# Integrate LoRA into the model
model = get_peft_model(model, peft_config)

# 配置训练参数
# Configure training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
)

# 使用Hugging Face的Trainer API进行微调
# Fine-tune the model using Hugging Face's Trainer API
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'].select(range(1000)),  # 只选取一部分数据进行演示 - Select part of dataset for demo
    eval_dataset=dataset['test'].select(range(1000))
)

# 开始训练
# Start training
trainer.train()

# LoRA 微调后的模型保存
# Save the LoRA fine-tuned model
model.save_pretrained("./lora-finetuned-distilbert")
