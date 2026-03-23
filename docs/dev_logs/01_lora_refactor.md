# LoRA 代码重构日志

## 2026-03-22

### 问题1: Epoch Time 计算错误
- **问题**: `epoch_time` 一直是0
- **原因**: ETA计算公式有误：`spend_time / (step + 1) * iters // 60 - spend_time // 60`
- **修复**: 改为 `avg_time_per_step * remaining_steps / 60`
- **文件**: `LoRA/train_lora.py`

### 问题2: 数据格式兼容
- **问题**: 训练数据使用 `user_input` + `answer_r1` 格式，但代码期望 `conversations` 格式
- **修复**: `LoRADataset` 增加格式自动检测和兼容处理
- **文件**: `dataset/lora_dataset.py`

### 问题3: LoRA 加载失败
- **问题**: 推理时提示 `Missing key: A.weight, B.weight`
- **原因**: `eval_llm.py` 使用旧的 `model/model_lora.py`，与训练用的 `LoRA/model_LoRA.py` 不一致
- **修复**: 
  - 统一使用 `LoRA/model_LoRA.py`
  - 修改导入路径为 `from LoRA.model_LoRA import ...`
- **文件**: `eval_llm.py`, `scripts/serve_openai_api.py`

### 问题4: Tokenizer 输入类型错误
- **问题**: `TypeError: TextEncodeInput must be Union[TextInputSequence, Tuple[InputSequence, InputSequence]]`
- **修复**: 确保 `apply_chat_template` 返回值一定是字符串
- **文件**: `eval_llm.py`

### 重构: 清理重复文件
- **操作**: 删除重复的代码文件，统一使用 `LoRA/` 目录
- **删除**: `model/model_lora.py`, `trainer/train_lora.py`
- **保留**: `LoRA/model_LoRA.py`, `LoRA/train_lora.py`

### 新增功能: Loss 曲线绘图
- **功能**: 训练结束后自动保存 loss 曲线图
- **输出**: `out/lora/loss_history.json`, `out/lora/loss_curve.png`
- **文件**: `LoRA/train_lora.py`
