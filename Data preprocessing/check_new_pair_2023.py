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
    go_graph = nx.DiGraph()
    for _, row in go_data.iterrows():
        child, parent = row['id'], row['related_id']
        go_graph.add_edge(parent, child)
    return go_graph


def load_pairs_from_file(file_path):
    """
    从 CSV 文件中加载节点对。
    :param file_path: CSV 文件路径。
    :return: 节点对列表 [(node1, node2), ...]。
    """
    pairs_data = pd.read_csv(file_path, sep='\t')
    return list(zip(pairs_data['Node1'], pairs_data['Node2']))


def check_sibling_in_2023(graph, pairs):
    """
    检查 2023 图中是否保持兄弟关系（具有共同父节点）
    """
    still_sibling = []
    no_longer_sibling = []
    for (a, b) in pairs:
        if not (a in graph.nodes and b in graph.nodes):
            continue

        parents_a = set(graph.predecessors(a))
        parents_b = set(graph.predecessors(b))
        if parents_a and parents_b and (parents_a & parents_b):
            still_sibling.append((a, b))
        else:
            no_longer_sibling.append((a, b))
    return still_sibling, no_longer_sibling


def check_grandparent_in_2023(graph, pairs):
    """
    检查 2023 图中是否保持爷孙关系
    """
    still_grandparent = []
    no_longer_grandparent = []
    for (a, b) in pairs:
        if not (a in graph.nodes and b in graph.nodes):
            continue

        is_grand = False
        for parent in graph.predecessors(b):
            if a in graph.predecessors(parent):
                is_grand = True
                break
        if not is_grand:
            for parent in graph.predecessors(a):
                if b in graph.predecessors(parent):
                    is_grand = True
                    break

        if is_grand:
            still_grandparent.append((a, b))
        else:
            no_longer_grandparent.append((a, b))
    return still_grandparent, no_longer_grandparent


def check_common_child_in_2023(graph, pairs):
    """
    检查 2023 图中是否保持共同子节点关系
    """
    still_common_child = []
    no_longer_common_child = []
    for (a, b) in pairs:
        if not (a in graph.nodes and b in graph.nodes):
            continue

        children_a = set(graph.successors(a))
        children_b = set(graph.successors(b))
        if children_a and children_b and (children_a & children_b):
            still_common_child.append((a, b))
        else:
            no_longer_common_child.append((a, b))
    return still_common_child, no_longer_common_child


def save_results_to_file(results, filename):
    """
    保存结果为 CSV 文件
    """
    if not os.path.exists('output_2023_check'):
        os.makedirs('output_2023_check')

    df = pd.DataFrame(results, columns=['Node1', 'Node2'])
    file_path = os.path.join('output_2023_check', filename)
    df.to_csv(file_path, sep='\t', index=False)
    print(f"结果已保存至：{file_path}")


def main():
    # 2023 版本的 is_a 关系文件
    is_a_file_2023 = '../data/go_2023/is_a_relations.csv'

    # 加载 2023 版本的 GO 图
    go_graph_2023 = load_go_graph(is_a_file_2023)

    # 加载已保存的三种类型的节点对
    sibling_pairs = load_pairs_from_file('output/sibling_results.csv')
    grandparent_pairs = load_pairs_from_file('output/grandparent_results.csv')
    common_child_pairs = load_pairs_from_file('output/common_child_results.csv')

    # 检查并保存兄弟关系
    sibling_still, sibling_no_longer = check_sibling_in_2023(go_graph_2023, sibling_pairs)
    save_results_to_file(sibling_still, "sibling_still.csv")
    save_results_to_file(sibling_no_longer, "sibling_no_longer.csv")

    # 检查并保存爷孙关系
    grandparent_still, grandparent_no_longer = check_grandparent_in_2023(go_graph_2023, grandparent_pairs)
    save_results_to_file(grandparent_still, "grandparent_still.csv")
    save_results_to_file(grandparent_no_longer, "grandparent_no_longer.csv")

    # 检查并保存共同子节点关系
    common_child_still, common_child_no_longer = check_common_child_in_2023(go_graph_2023, common_child_pairs)
    save_results_to_file(common_child_still, "common_child_still.csv")
    save_results_to_file(common_child_no_longer, "common_child_no_longer.csv")


if __name__ == "__main__":
    main()
