# 02 项目结构重构日志

## 2026-03-24

### 背景
`LoRA/trainer_utils.py` 包含通用训练工具函数（`Logger`、`get_lr`、`lm_checkpoint`、`init_model` 等），但被所有训练脚本共用，放在 LoRA 目录下职责不清晰。

### 变更内容

#### 1. 文件移动
- **操作**：`LoRA/trainer_utils.py` → `trainer/trainer_utils.py`（物理复制，旧文件保留以兼容）
- **原因**：`trainer_utils` 是全局训练工具，与 LoRA 无关

#### 2. 修复 4 处导入路径
| 文件 | 旧导入 | 新导入 |
|------|--------|--------|
| `trainer/train_pretrain.py` | `from LoRA.trainer_utils import ...` | `from trainer.trainer_utils import ...` |
| `trainer/train_full_sft.py` | `from LoRA.trainer_utils import ...` | `from trainer.trainer_utils import ...` |
| `eval_llm.py` | `from LoRA.trainer_utils import ...` | `from trainer.trainer_utils import ...` |
| `LoRA/train_lora.py` | `from trainer_utils import ...` | `from trainer.trainer_utils import ...` |

#### 3. 消除重复的 checkpoint 保存逻辑
- **文件**：`trainer/train_pretrain.py`、`trainer/train_full_sft.py`
- **问题**：`train_epoch()` 内部同时有手写 `torch.save()` + 调用 `lm_checkpoint()`，二者功能重叠
- **修复**：删除手写 `torch.save` 块，统一由 `lm_checkpoint()` 负责保存

### 注意事项
- `sys.path.append` hack 保留：项目无 `setup.py`，脚本需从根目录运行，这是当前唯一合理方式
- Pyre2 IDE lint 报 "Could not find import of torch" 等为静态分析器未配置项目路径导致，不影响实际运行
