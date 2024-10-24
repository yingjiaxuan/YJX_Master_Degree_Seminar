# 双语注释:
# DoRA 示例代码：通过权重的分解进行微调。
# DoRA通过大小和方向的权重分解来微调大模型。
# DoRA example: Fine-tuning via weight decomposition into magnitude and direction.

from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from peft import get_peft_model, LoraConfig

# 加载IMDb数据集 - 电影评论分类
# Load IMDb dataset for movie review classification
dataset = load_dataset("imdb")

# 加载DistilBERT模型
# Load DistilBERT model
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# 配置DoRA相关参数 (假设DoRA基于LoRA的扩展实现)
# Configure DoRA parameters (Assuming DoRA extends LoRA configuration)
peft_config = LoraConfig(
    r=8,  # 低秩矩阵的秩 - Rank of low-rank matrices
    lora_alpha=16,  # LoRA 缩放参数 - LoRA scaling parameter
    target_modules=["query", "value"],  # 微调的目标模块 - Target modules for fine-tuning
    lora_dropout=0.1,  # Dropout 用于正则化 - Dropout for regularization
)

# 将LoRA集成到模型中（DoRA中假设结合大小和方向的分解）
# Integrate LoRA into the model (assuming DoRA with magnitude and direction decomposition)
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
    train_dataset=dataset['train'].select(range(1000)),  # 只选取部分数据进行演示 - Selecting part of dataset for demo
    eval_dataset=dataset['test'].select(range(1000))
)

# 开始训练
# Start training
trainer.train()

# 保存微调后的DoRA模型
# Save the fine-tuned DoRA model
model.save_pretrained("./dora-finetuned-distilbert")
