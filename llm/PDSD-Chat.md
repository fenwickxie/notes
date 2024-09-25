# PDSD-Chat

---

---

---

## 模型训练
### 基座模型
+ 开源模型：Qwen2.5-0.5B-Instruct

---

### 微调软硬件要求
#### 硬件平台
+ 资源估算

|方法|精度|0.5B|7B|13B|30B|70B|110B|
|---|---|---|---|---|---|---|---|
|Full|AMP|9GB|120GB|240GB|600GB|1200GB|2000GB|
|Full|16|5GB|	60GB|120GB|300GB|600GB|	900GB|
|Freeze|16|2GB|20GB|40GB|80GB|200GB|360GB|
|LoRA/GaLore/BAdam|16|1.5GB|16GB|32GB|64GB|160GB|240GB|
|QLoRA|8|1GB|10GB|20GB|40GB|80GB|140GB|
|QLoRA|4|0.5GB|6GB|12GB|24GB|48GB|72GB|
|QLoRA|2|0.5GB|4GB|8GB	|16GB|24GB|48GB|

+ 
#### 软件平台
+ 微调工具：LLaMA-Factory
  + 安装 LLaMA Factory
  ```bash
  git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
  cd LLaMA-Factory
  pip install -e ".[torch,metrics]"
  ```
  可选的额外依赖项：torch、torch-npu、metrics、deepspeed、liger-kernel、bitsandbytes、hqq、eetq、gptq、awq、aqlm、vllm、galore、badam、adam-mini、qwen、modelscope、quality
  + 在 Windows 平台上开启量化 LoRA（QLoRA），需要安装预编译的 bitsandbytes 库,支持 CUDA 11.1 到 12.2
  ```bash
  pip install https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.38.1-py3-none-win_amd64.whl
  ```

  + 在 Windows 平台上开启 FlashAttention-2，需要安装预编译的 flash-attn 库,支持 CUDA 12.1 到 12.2
+ 语言环境：Python 3.9.20
+ 依赖库
  + [CUDA_11.8](https://developer.nvidia.com/cuda-toolkit-archive)
  + [cuDNN_9.0.1](https://developer.nvidia.com/rdp/cudnn-archive)
  + [pytorch_2.4.0](https://pytorch.org/get-started/locally/)
  + [bitsandbytes](https://github.com/jllllll/bitsandbytes-windows-webui/releases/tag/wheels)
  + [flash-attn](https://github.com/bdashore3/flash-attention/releases)

---

### 数据获取
#### 自定义数据
将数据以 `json` 格式进行组织，并将数据放入 `data` 文件夹中。LLaMA-Factory 支持以 `alpaca` 或 `sharegpt` 格式的数据集
+ alpaca 格式的数据集应遵循以下格式
```json
[
  {
    "instruction": "user instruction (required)",
    "input": "user input (optional)",
    "output": "model response (required)",
    "system": "system prompt (optional)",
    "history": [
      ["user instruction in the first round (optional)", "model response in the first round (optional)"],
      ["user instruction in the second round (optional)", "model response in the second round (optional)"]
    ]
  }
]
```
+ sharegpt 格式的数据集应遵循以下格式
```json
[
  {
    "conversations": [
      {
        "from": "human",
        "value": "user instruction"
      },
      {
        "from": "gpt",
        "value": "model response"
      }
    ],
    "system": "system prompt (optional)",
    "tools": "tool description (optional)"
  }
]
```

#### 定义数据集
在 data/dataset_info.json 文件中提供数据集定义

+ 对于 alpaca 格式的数据集，其 dataset_info.json 文件中的列应为
```
"dataset_name": {
  "file_name": "dataset_name.json",
  "columns": {
    "prompt": "instruction",
    "query": "input",
    "response": "output",
    "system": "system",
    "history": "history"
  }
}
```
+ 对于 sharegpt 格式的数据集，dataset_info.json 文件中的列应该包括

```
"dataset_name": {
    "file_name": "dataset_name.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "conversations",
      "system": "system",
      "tools": "tools"
    },
    "tags": {
      "role_tag": "from",
      "content_tag": "value",
      "user_tag": "user",
      "assistant_tag": "assistant"
    }
  }
```

---

### 模型微调
执行下列命令:
```bash
DISTRIBUTED_ARGS="
    --nproc_per_node $NPROC_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
  "

torchrun $DISTRIBUTED_ARGS src/train.py \
    --deepspeed $DS_CONFIG_PATH \
    --stage sft \
    --do_train \
    --use_fast_tokenizer \
    --flash_attn \
    --model_name_or_path $MODEL_PATH \
    --dataset your_dataset \
    --template qwen \
    --finetuning_type lora \
    --lora_target q_proj,v_proj\
    --output_dir $OUTPUT_PATH \
    --overwrite_cache \
    --overwrite_output_dir \
    --warmup_steps 100 \
    --weight_decay 0.1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --ddp_timeout 9000 \
    --learning_rate 5e-6 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --cutoff_len 4096 \
    --save_steps 1000 \
    --plot_loss \
    --num_train_epochs 3 \
    --bf16
```
+ `cutoff_len`:代表训练数据的最大长度。通过控制这个参数，可以避免出现OOM（内存溢出）错误

---

### 合并LoRA

如果使用 LoRA 训练模型，可能需要将adapter参数合并到主分支中。运行以下命令以执行 LoRA adapter 的合并操作
```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli export \
    --model_name_or_path path_to_base_model \
    --adapter_name_or_path path_to_adapter \
    --template qwen \
    --finetuning_type lora \
    --export_dir path_to_export \
    --export_size 2 \
    --export_legacy_format False
```

---

---

## 应用搭建

### LangChain-Chatchat
利用[langchain](https://github.com/langchain-ai/langchain)思想实现的基于本地知识库的问答应用
