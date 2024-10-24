from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from peft import get_peft_model, LoraConfig

# 使用Hugging Face的bitsandbytes进行8-bit量化
# Using bitsandbytes from Hugging Face for 8-bit quantization
from transformers import BitsAndBytesConfig

# 加载IMDb数据集 - 电影评论分类
# Load IMDb dataset
dataset = load_dataset("imdb")

# 配置量化设置
# Configure quantization settings
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,  # 8-bit量化 - 8-bit quantization
    bnb_8bit_use_double_quant=True,  # 双重量化 - Double quantization
    bnb_8bit_quant_type="nf4",  # 使用NF4的量化类型 - Quantization type NF4
)

# 加载8-bit量化的DistilBERT模型
# Load DistilBERT model with 8-bit quantization
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2, quantization_config=bnb_config)

# 配置LoRA参数
# Configure LoRA parameters
peft_config = LoraConfig(
    r=8,  # 低秩矩阵秩 - Low-rank matrix rank
    lora_alpha=16,  # LoRA缩放参数 - LoRA scaling factor
    target_modules=["query", "value"],  # 微调的模块 - Target modules for fine-tuning
    lora_dropout=0.1,  # Dropout 用于正则化 - Dropout for regularization
)

# 将LoRA与8-bit量化模型集成
# Integrate LoRA with the 8-bit quantized model
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

# 保存微调后的QLoRA模型
# Save the fine-tuned QLoRA model
model.save_pretrained("./qlora-finetuned-distilbert")
