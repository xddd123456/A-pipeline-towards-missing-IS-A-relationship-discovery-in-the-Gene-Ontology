import networkx as nx
import random
import csv


def calculate_node_depths(graph):
    """
    计算每个节点的深度（从根节点开始到该节点的最长路径长度）。
    :param graph: GO 图 (NetworkX DiGraph)。
    :return: 字典 {节点: 深度}。
    """
    root_nodes = [node for node in graph.nodes() if graph.in_degree(node) == 0]
    depths = {}

    for root in root_nodes:
        for node, depth in nx.single_source_shortest_path_length(graph.reverse(), root).items():
            depths[node] = depth

    return depths


def find_leaf_nodes(graph):
    """
    找到图中的所有叶子节点（出度为 0 的节点）。
    :param graph: GO 图 (NetworkX DiGraph)。
    :return: 叶子节点列表。
    """
    return [node for node in graph.nodes() if graph.out_degree(node) == 0]


def extract_subtrees_from_leaves(graph, depths, min_nodes=30, max_nodes=100, num_subtrees=100):
    """
    从叶子节点向上追溯五层构建子树。
    :param graph: GO 图 (NetworkX DiGraph)。
    :param depths: 节点深度字典。
    :param min_nodes: 子树的最小节点数。
    :param max_nodes: 子树的最大节点数。
    :param num_subtrees: 需要提取的子树数量。
    :return: 提取的子树列表。
    """
    subtrees = []
    visited_nodes = set()
    leaf_nodes = find_leaf_nodes(graph)

    for leaf in leaf_nodes:
        # 向上追溯四层找根节点
        current_node = leaf
        for _ in range(4):
            predecessors = list(graph.predecessors(current_node))
            if predecessors:
                current_node = predecessors[0]
            else:
                break

        # 从根节点向下提取子树
        subtree_nodes = list(nx.dfs_preorder_nodes(graph, source=current_node))

        # 子树必须满足节点范围要求
        if min_nodes <= len(subtree_nodes) <= max_nodes and current_node not in visited_nodes:
            subtrees.append(subtree_nodes)
            visited_nodes.update(subtree_nodes)

        # 如果达到了需要的子树数量，则退出
        if len(subtrees) >= num_subtrees:
            break

    return subtrees


def find_ndr_pairs(graph, subtrees):
    """
    查找 NDR 节点对：共享相同子树或父节点的节点对，以及最近的叔侄关系。
    :param graph: GO 图 (DiGraph)。
    :param subtrees: 提取的子树列表，每个子树是节点的集合。
    :return: NDR 节点对列表。
    """
    ndr_pairs = []

    for subtree in subtrees:
        # 遍历子树中的所有节点对
        for i in range(len(subtree)):
            for j in range(i + 1, len(subtree)):
                node1, node2 = subtree[i], subtree[j]
                # 检查是否有共同父节点或共同子节点
                parents1 = set(graph.predecessors(node1))
                parents2 = set(graph.predecessors(node2))
                children1 = set(graph.successors(node1))
                children2 = set(graph.successors(node2))

                if parents1 & parents2 or children1 & children2:
                    ndr_pairs.append((node1, node2))
                    ndr_pairs.append((node2, node1))
                else:
                    # 检查最近的叔侄关系
                    if node1 in children2 or node2 in children1:
                        ndr_pairs.append((node1, node2))
                        ndr_pairs.append((node2, node1))

    return ndr_pairs

# def find_ndr_pairs(graph, subtrees):
#     """
#     查找 NDR 节点对：共享相同子树或父节点的节点对，以及最近的叔侄关系。
#     :param graph: GO 图 (DiGraph)。
#     :param subtrees: 提取的子树列表，每个子树是节点的集合。
#     :return: NDR 节点对列表。
#     """
#     ndr_pairs = set()  # 用 set 避免重复
#
#     for subtree in subtrees:
#         # 遍历子树中的所有节点对
#         for i in range(len(subtree)):
#             for j in range(i + 1, len(subtree)):
#                 node1, node2 = subtree[i], subtree[j]
#
#                 if node1 not in graph or node2 not in graph:
#                     continue  # 避免访问不存在的节点
#
#                 # 获取父节点和子节点
#                 parents1 = set(graph.predecessors(node1))
#                 parents2 = set(graph.predecessors(node2))
#                 children1 = set(graph.successors(node1))
#                 children2 = set(graph.successors(node2))
#
#                 # **条件 1：检查是否有共同父节点（兄弟关系）**
#                 if parents1 & parents2:
#                     ndr_pairs.add((node1, node2))
#                     ndr_pairs.add((node2, node1))  # 加入双向关系
#                     continue  # 找到兄弟关系后，不用再检查下面的情况
#
#                 # **条件 2：检查是否有共同子节点**
#                 if children1 & children2:
#                     ndr_pairs.add((node1, node2))
#                     ndr_pairs.add((node2, node1))
#                     continue
#
#                 # **条件 3：检查叔侄关系**
#                 for parent1 in parents1:
#                     for parent2 in parents2:
#                         if parent1 == parent2:
#                             continue  # 如果是同一个父亲，那是兄弟，不是叔侄
#
#                         grandparent1 = set(graph.predecessors(parent1))
#                         grandparent2 = set(graph.predecessors(parent2))
#
#                         if grandparent1 & grandparent2:  # 说明 parent1 和 parent2 是兄弟
#                             ndr_pairs.add((node1, node2))
#                             break
#
#     return ndr_pairs


def save_ndr_pairs_to_csv(ndr_pairs, output_file):
    """
    将 NDR 节点对保存到 CSV 文件。
    :param ndr_pairs: NDR 节点对列表。
    :param output_file: 输出文件路径。
    """
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(["Node1", "Node2"])  # 写入表头
        writer.writerows(ndr_pairs)  # 写入数据


def load_go_graph(is_a_file):
    """
    从 is_a 关系文件加载 GO 图。
    :param is_a_file: 包含 GO is_a 关系的文件路径，CSV 格式，列为 'child' 和 'parent'。
    :return: 构建的有向图 (DiGraph)。
    """
    import pandas as pd
    go_data = pd.read_csv(is_a_file, sep='\t')

    # 构建 GO 图
    go_graph = nx.DiGraph()
    for _, row in go_data.iterrows():
        child, parent = row['id'], row['related_id']
        go_graph.add_edge(parent, child)

    return go_graph


# 主流程
is_a_file = "data/go_2022/is_a_relations.csv"  # 替换为你的 is_a 文件路径
output_file = "model_prediction/prediction_data/go_2022/ndr_pairs_100_4.csv"

# 加载 GO 图
go_graph = load_go_graph(is_a_file)

# 计算节点深度
node_depths = calculate_node_depths(go_graph)

# 提取子树
subtrees = extract_subtrees_from_leaves(go_graph, node_depths, min_nodes=30, max_nodes=100, num_subtrees=100)
print(f"提取了 {len(subtrees)} 个子树")

# 查找 NDR 节点对
ndr_pairs = find_ndr_pairs(go_graph, subtrees)
print(f"找到 {len(ndr_pairs)} 对 NDR 节点对")

# 保存 NDR 节点对到 CSV 文件
save_ndr_pairs_to_csv(ndr_pairs, output_file)
print(f"NDR 节点对已保存到 {output_file}")
