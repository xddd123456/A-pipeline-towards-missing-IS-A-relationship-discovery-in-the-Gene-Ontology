import os
import pandas as pd
import networkx as nx


def load_go_graph(is_a_file):
    """
    从 is_a 关系文件加载 GO 图。
    :param is_a_file: 包含 GO is_a 关系的文件路径，CSV 格式，列为 'id' 和 'related_id'。
    :return: 构建的有向图 (DiGraph)。
    """
    go_data = pd.read_csv(is_a_file, sep='\t')

    # 构建 GO 图
    go_graph = nx.DiGraph()
    for _, row in go_data.iterrows():
        child, parent = row['id'], row['related_id']
        go_graph.add_edge(parent, child)

    return go_graph


def load_node_pairs_from_csv(file_path):
    """
    从 CSV 文件中加载目标节点对。
    :param file_path: 文件路径，CSV 文件，包含两列：'id' 和 'related_id'。
    :return: 节点对列表 [(node1, node2), ...]。
    """
    pairs_data = pd.read_csv(file_path, sep='\t')
    return list(zip(pairs_data['id'], pairs_data['related_id']))


def check_sibling_in_new_pair(graph, new_pairs):
    """
    条件1：检查 new_pair 中的每个节点对，在 2022 图中是否共享父节点
    """
    sibling_pairs = []
    for (a, b) in new_pairs:
        # 检查节点是否在图中
        if not (a in graph.nodes and b in graph.nodes):
            print(f"跳过不存在于 2022 图中的节点对：({a}, {b})")
            continue

        # 获取 a 和 b 在 2022 图中的父节点集合
        parents_a = set(graph.predecessors(a))
        parents_b = set(graph.predecessors(b))
        if parents_a and parents_b and (parents_a & parents_b):
            sibling_pairs.append((a, b))
    return sibling_pairs


def check_grandparent_in_new_pair(graph, new_pairs):
    """
    条件2：检查 new_pair 中的每个节点对，在 2022 图中是否存在爷孙关系
    """
    grandparent_pairs = []
    for (a, b) in new_pairs:
        # 检查节点是否在图中
        if not (a in graph.nodes and b in graph.nodes):
            print(f"跳过不存在于 2022 图中的节点对：({a}, {b})")
            continue

        is_grand = False
        # 检查 a 是否为 b 的祖父
        for parent in graph.predecessors(b):
            if a in graph.predecessors(parent):
                is_grand = True
                break
        # 检查 b 是否为 a 的祖父
        if not is_grand:
            for parent in graph.predecessors(a):
                if b in graph.predecessors(parent):
                    is_grand = True
                    break
        if is_grand:
            grandparent_pairs.append((a, b))
    return grandparent_pairs


def check_common_child_in_new_pair(graph, new_pairs):
    """
    条件3：检查 new_pair 中的每个节点对，在 2022 图中是否具有共同子节点
    """
    common_child_pairs = []
    for (a, b) in new_pairs:
        # 检查节点是否在图中
        if not (a in graph.nodes and b in graph.nodes):
            print(f"跳过不存在于 2022 图中的节点对：({a}, {b})")
            continue

        children_a = set(graph.successors(a))
        children_b = set(graph.successors(b))
        if children_a and children_b and (children_a & children_b):
            common_child_pairs.append((a, b))
    return common_child_pairs


def save_results_to_file(results, filename):
    """
    将结果保存为 CSV 文件。
    :param results: 节点对列表 [(node1, node2), ...]。
    :param filename: 保存的文件名。
    """
    if not os.path.exists('output'):
        os.makedirs('output')

    df = pd.DataFrame(results, columns=['Node1', 'Node2'])
    file_path = os.path.join('output', filename)
    df.to_csv(file_path, sep='\t', index=False)
    print(f"结果已保存至：{file_path}")


def main():
    # 文件路径（请根据实际情况修改）
    is_a_file_2022 = '../data/go_2022/is_a_relations.csv'  # 2022 版本的 is_a 关系文件
    new_pair_file = 'new_go_pairs_2023.csv'  # new_pair 中的 IS_A 关系文件

    # 加载 2022 版本的 GO 图
    go_graph_2022 = load_go_graph(is_a_file_2022)
    # 加载 new_pair 中的关系
    new_pairs = load_node_pairs_from_csv(new_pair_file)

    # 条件 1：兄弟关系在 2022 中，但在 new_pair 中呈现 IS_A 关系
    sibling_results = check_sibling_in_new_pair(go_graph_2022, new_pairs)
    print("条件1 - 兄弟关系变 IS_A 的情况：")
    print(f"数量：{len(sibling_results)}")
    save_results_to_file(sibling_results, "sibling_results.csv")

    # 条件 2：爷孙关系在 2022 中，但在 new_pair 中呈现 IS_A 关系
    grandparent_results = check_grandparent_in_new_pair(go_graph_2022, new_pairs)
    print("\n条件2 - 爷孙关系变 IS_A 的情况：")
    print(f"数量：{len(grandparent_results)}")
    save_results_to_file(grandparent_results, "grandparent_results.csv")

    # 条件 3：new_pair 中的 IS_A 关系在 2022 中具有同一个子节点
    common_child_results = check_common_child_in_new_pair(go_graph_2022, new_pairs)
    print("\n条件3 - 同子 IS_A 的情况：")
    print(f"数量：{len(common_child_results)}")
    save_results_to_file(common_child_results, "common_child_results.csv")

    # 统计跳过的节点对数量
    print("\n统计：")
    print(f"new_pair 中节点对总数：{len(new_pairs)}")
    total_skipped = len(new_pairs) - len(sibling_results) - len(grandparent_results) - len(common_child_results)
    print(f"被跳过的节点对数量：{total_skipped}")


if __name__ == "__main__":
    main()
