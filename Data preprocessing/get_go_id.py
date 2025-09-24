import json


def parse_go_obo(file_path):
    results = []
    current_entry = {}

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line == "[Term]":  # Start of a new term
                if current_entry:
                    results.append(current_entry)
                current_entry = {}
            elif line.startswith("id: "):
                current_entry["id"] = line.split("id: ")[1]
            elif line.startswith("name: "):
                current_entry["name"] = line.split("name: ")[1]
            elif line.startswith("def: "):
                current_entry["def"] = line.split("def: ")[1].strip('"')  # Remove leading/trailing quotes
        # Append the last entry
        if current_entry:
            results.append(current_entry)

    # 为每个 GO term 添加编号
    go_id_to_index = {term["id"]: idx for idx, term in enumerate(results)}

    return go_id_to_index


# 使用示例
go_obo_file = "../data/go_2025/go2025.obo"  # 替换为实际文件路径
go_id_to_index = parse_go_obo(go_obo_file)

# 转换为 JSON 格式并保存
output_file = "../data/go_2025/id_to_index.json"
with open(output_file, "w", encoding="utf-8") as json_file:
    json.dump(go_id_to_index, json_file, indent=4, ensure_ascii=False)

print(f"提取并编号完成，结果已保存到 {output_file}")
