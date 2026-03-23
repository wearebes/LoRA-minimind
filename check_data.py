import json

# 读取数据集示例
with open('./dataset/lora_dataset/Finance_R1-Distill_data_0.jsonl', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 2:
            break
        data = json.loads(line.strip())
        print(f"=== 第 {i+1} 条 ===")
        print(json.dumps(data, ensure_ascii=False, indent=2)[:800])
        print()
