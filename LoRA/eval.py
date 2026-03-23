
import os
import sys
import json
import time
import torch
import re
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from LoRA.model_LoRA import load_lora
from transformers import AutoTokenizer


def load_model(weight_name, lora_name=None, hidden_size=768, num_layers=16):
    """加载模型"""
    config = MiniMindConfig(hidden_size=hidden_size, num_hidden_layers=num_layers)
    model = MiniMindForCausalLM(config)
    
    base_path = f"../out/{weight_name}_{hidden_size}.pth"
    model.load_state_dict(torch.load(base_path, map_location="cpu"), strict=True)
    
    if lora_name:
        lora_path = f"../out/lora/{lora_name}_{hidden_size}.pth"
        load_lora(model, lora_path)
    
    tokenizer = AutoTokenizer.from_pretrained("../model")
    return model.eval().to("cuda"), tokenizer


def normalize_text(text):
    """简单文本规范化：去除多余空白，转小写"""
    text = re.sub(r'\s+', ' ', text.strip())
    return text.lower()


def compute_f1(pred, target):
    """计算F1分数（基于词重叠）"""
    pred_tokens = set(normalize_text(pred).split())
    target_tokens = set(normalize_text(target).split())
    
    if len(pred_tokens) == 0 or len(target_tokens) == 0:
        return 0
    
    common = pred_tokens & target_tokens
    if len(common) == 0:
        return 0
    
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(target_tokens)
    
    if precision + recall == 0:
        return 0
    return 2 * precision * recall / (precision + recall)


def compute_exact_match(pred, target):
    """计算Exact Match"""
    return 1.0 if normalize_text(pred) == normalize_text(target) else 0.0


def generate_answer(model, tokenizer, prompt, max_new_tokens=256):
    """生成模型答案"""
    bos = tokenizer.bos_token if tokenizer.bos_token else ''
    inputs = tokenizer(bos + prompt, return_tensors="pt", truncation=True, max_length=512).to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    generated = outputs[0][len(inputs["input_ids"][0]):]
    answer = tokenizer.decode(generated, skip_special_tokens=True)
    return answer.strip()


def evaluate(model, tokenizer, test_data, max_samples=50):
    """评测模型，返回F1、EM、Accuracy"""
    f1_scores = []
    em_scores = []
    
    print(f"  评测 {min(max_samples, len(test_data))} 条数据...")
    
    for i, item in enumerate(test_data[:max_samples]):
        prompt = item.get("user_input", "")
        reference = item.get("answer_r1", item.get("answer", ""))
        
        # 生成模型答案
        pred = generate_answer(model, tokenizer, prompt)
        
        # 计算指标
        f1 = compute_f1(pred, reference)
        em = compute_exact_match(pred, reference)
        
        f1_scores.append(f1)
        em_scores.append(em)
        
        if i < 3:  # 显示前3条样例
            print(f"\n--- 样例 {i+1} ---")
            print(f"问题: {prompt[:80]}...")
            print(f"参考答案: {reference[:80]}...")
            print(f"模型答案: {pred[:80]}...")
            print(f"F1: {f1:.4f}, EM: {em:.4f}")
    
    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
    avg_em = sum(em_scores) / len(em_scores) if em_scores else 0
    
    # Accuracy: EM >= 0.9 就算匹配（宽松标准）
    accuracy = sum(1 for em in em_scores if em >= 0.9) / len(em_scores) if em_scores else 0
    
    return {
        "f1": avg_f1,
        "em": avg_em,
        "accuracy": accuracy
    }


def main():
    # 测试数据
    test_file = "../dataset/lora_dataset/Finance_R1-Distill_data_0.jsonl"
    test_data = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 10:  # 取前10条
                break
            test_data.append(json.loads(line))
    
    print(f"测试集: {len(test_data)} 条")
    
    # 要评测的模型列表
    models = [
        ("full_sft", None),      # 基座模型
        ("full_sft", "lora_8"),  # LoRA 8层
        ("full_sft", "lora_16"), # LoRA 16层
    ]
    
    results = []
    print("\n========== 开始评测 ==========\n")
    
    for weight, lora in models:
        try:
            model_name = weight + (f"+{lora}" if lora else "")
            print(f"评测: {model_name}")
            model, tokenizer = load_model(weight, lora)
            
            start = time.time()
            metrics = evaluate(model, tokenizer, test_data, max_samples=10)
            cost = time.time() - start
            
            results.append({
                "model": model_name,
                "f1": metrics["f1"],
                "em": metrics["em"],
                "accuracy": metrics["accuracy"],
                "time": cost
            })
            
            print(f"  F1: {metrics['f1']:.4f}, EM: {metrics['em']:.4f}, Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Time: {cost:.2f}s\n")
            
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  加载失败: {e}\n")
    
    # 输出表格
    print("\n" + "="*65)
    print("| Model              | F1     | EM     | Acc    | Time(s) |")
    print("|" + "-"*19 + "|" + "-"*8 + "|" + "-"*8 + "|" + "-"*8 + "|" + "-"*10 + "|")
    for r in results:
        print(f"| {r['model']:<18} | {r['f1']:.4f} | {r['em']:.4f} | {r['accuracy']:.4f} | {r['time']:.2f}   |")
    print("="*65)
    
    # 保存结果
    with open("../out/lora/eval_results.txt", "w", encoding='utf-8') as f:
        f.write("| Model              | F1     | EM     | Acc    | Time(s) |\n")
        f.write("|" + "-"*19 + "|" + "-"*8 + "|" + "-"*8 + "|" + "-"*8 + "|" + "-"*10 + "|\n")
        for r in results:
            f.write(f"| {r['model']:<18} | {r['f1']:.4f} | {r['em']:.4f} | {r['accuracy']:.4f} | {r['time']:.2f}   |\n")
    print("\n结果已保存: out/lora/eval_results.txt")


if __name__ == "__main__":
    main()
