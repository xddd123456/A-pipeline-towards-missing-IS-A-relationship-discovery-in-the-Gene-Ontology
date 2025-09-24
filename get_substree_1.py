import networkx as nx
import csv
import pandas as pd
import random
import json

def get_min_depth(go_id):
    """计算某个 GO 术语到最近根节点的最短路径（层级数）。"""
    if go_id not in G_directed:
        return None
    min_depth = float("inf")
    for root in roots:
        try:
            depth = nx.shortest_path_length(G_directed, source=root, target=go_id)
            min_depth = min(min_depth, depth)
        except nx.NetworkXNoPath:
            continue
    return min_depth if min_depth != float("inf") else None


def find_leaf_nodes(graph, valid_depths={4, 5, 6}):
    """找到所有符合深度要求的叶子节点（出度为 0 且深度符合条件）。"""
    leaf_nodes = []
    for node in graph.nodes():
        if graph.out_degree(node) == 0:  # 叶子节点
            depth = get_min_depth(node)
            if depth in valid_depths:
                leaf_nodes.append(node)
    return leaf_nodes


def find_grandparents(graph, leaf_nodes):
    """找到所有叶子节点的祖父节点（无叶子数量或距离限制）。"""
    grandparents = set()
    for leaf in leaf_nodes:
        parents = list(graph.predecessors(leaf))
        for parent in parents:
            grandparents.update(graph.predecessors(parent))  # 获取祖父节点
    return list(grandparents)


def extract_subtree(graph, grandparent, min_nodes=30, max_nodes=100):
    """从指定祖父节点构建子树，并限制节点数在 min_nodes 和 max_nodes 之间。"""
    subtree_nodes = list(nx.dfs_preorder_nodes(graph, source=grandparent))
    if min_nodes <= len(subtree_nodes) <= max_nodes:
        return subtree_nodes
    return []

with open("data/go_2022/go_terms.json", "r", encoding="utf-8") as f:
    go_terms = {entry["id"]: entry["name"] for entry in json.load(f)}

def get_word_count(go_id):
    """获取 GO 术语的名称并计算单词数"""
    return len(go_terms.get(go_id, "").split())

def find_ndr_pairs(graph, subtrees):
    """
    查找 NDR 节点对：
    - 共享相同子树的节点对。
    - 叔侄节点对（具有共同祖父但不是父子）。
    """
    ndr_pairs = set()

    for subtree in subtrees:
        for i in range(len(subtree)):
            for j in range(i + 1, len(subtree)):
                node1, node2 = subtree[i], subtree[j]

                parents1 = set(graph.predecessors(node1))
                parents2 = set(graph.predecessors(node2))

                node1_word_count = get_word_count(node1)
                node2_word_count = get_word_count(node2)
                # 共享相同父节点
                if parents1 & parents2:
                    if node1_word_count >= node2_word_count:
                        ndr_pairs.add((node1, node2))
                    else:
                        ndr_pairs.add((node2, node1))

                # 叔侄关系：共同祖父但不是直接父子
                common_grandparents = set()
                for parent1 in parents1:
                    common_grandparents.update(graph.successors(parent1))  # 查找父亲的“父亲”（即祖父）
                for parent2 in parents2:
                    if parent2 in common_grandparents and node1 not in parents2 and node2 not in parents1:
                        if node1_word_count >= node2_word_count:
                            ndr_pairs.add((node1, node2))

    return sorted(ndr_pairs)


def save_ndr_pairs_to_csv(ndr_pairs, output_file):
    """将 NDR 节点对保存到 CSV 文件。"""
    with open(output_file, mode='w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(["Node1", "Node2"])
        writer.writerows(ndr_pairs)


def load_go_graph(is_a_file):
    """加载 GO 图。"""
    go_data = pd.read_csv(is_a_file, sep='\t')

    go_graph = nx.DiGraph()
    for _, row in go_data.iterrows():
        child, parent = row['id'], row['related_id']
        go_graph.add_edge(parent, child)

    return go_graph


# ------------- 主流程 --------------
is_a_file = "data/go_2022/is_a_relations.csv"  # 替换为你的 is_a 文件路径
go_graph = load_go_graph(is_a_file)

# 找到所有根节点（入度为 0）
roots = [node for node in go_graph.nodes() if go_graph.in_degree(node) == 0]
G_directed = go_graph  # 全局变量，供 get_min_depth() 使用

# 只从深度为 4、5、6 的叶子节点开始构建子树
leaf_nodes = find_leaf_nodes(go_graph)
print(f"找到 {len(leaf_nodes)} 个深度在 4-6 的叶子节点")

# 查找所有祖父节点（无叶子数量和距离限制）
grandparents = find_grandparents(go_graph, leaf_nodes)
print(f"找到 {len(grandparents)} 个祖父节点")

# 进行 10 轮抽样
for iteration in range(1, 1001):
    print(f"开始第 {iteration} 次随机抽取...")

    # 生成符合条件的子树
    subtrees = []
    for grandparent in grandparents:
        subtree = extract_subtree(go_graph, grandparent)
        if subtree:
            subtrees.append(subtree)

    # 如果符合条件的子树超过 100 棵，随机选择 100 棵
    if len(subtrees) > 100:
        subtrees = random.sample(subtrees, 100)

    print(f"第 {iteration} 轮：最终保留 {len(subtrees)} 组符合条件的祖父子树")

    # 查找 NDR 节点对
    ndr_pairs = find_ndr_pairs(go_graph, subtrees)
    print(f"第 {iteration} 轮：找到 {len(ndr_pairs)} 对 NDR 节点对")

    # 保存 NDR 节点对到 CSV 文件
    output_file = f"model_prediction/prediction_data/go_2022/fillter/example/ndr_pairs_with_uncle_nephew_iter{iteration}.csv"
    save_ndr_pairs_to_csv(ndr_pairs, output_file)

    print(f"第 {iteration} 轮：NDR 节点对已保存到 {output_file}")

print(f"所有 {iteration} 轮随机抽取完成！")
