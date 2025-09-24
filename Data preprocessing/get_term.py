import json

def parse_go_obo(file_path):
    results = []
    current_entry = None  # 追踪当前条目
    last_entry = None  # 追踪上一个条目

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line == "[Term]":
                if last_entry:  # 只存入倒数第二个条目，跳过最后一个
                    results.append(last_entry)
                last_entry = current_entry  # 记录当前条目
                current_entry = {}  # 重新初始化
            elif current_entry is not None:  # 确保在 [Term] 之后才处理
                if line.startswith("id: "):
                    current_entry["id"] = line.split("id: ")[1]
                elif line.startswith("name: "):
                    current_entry["name"] = line.split("name: ")[1]
                elif line.startswith("def: "):
                    current_entry["def"] = line.split("def: ")[1].strip('"')

    return results  # 由于最后一个未存入 last_entry，它不会被返回


# 使用示例
go_obo_file = "../data/go_2025/go2025.obo"  # 替换为实际文件路径
data = parse_go_obo(go_obo_file)

# 转换为 JSON 格式并保存
output_file = "../data/go_2025/go_terms.json"
with open(output_file, "w", encoding="utf-8") as json_file:
    json.dump(data, json_file, indent=4, ensure_ascii=False)

print(f"提取完成，结果已保存到 {output_file}")
