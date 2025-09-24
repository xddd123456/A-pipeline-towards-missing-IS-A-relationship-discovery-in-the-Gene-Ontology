import networkx as nx
import pandas as pd


def load_go_graph(file_path):
    """
    加载 GO 图，文件格式为 child -> parent
    """
    go_graph = nx.DiGraph()
    df = pd.read_csv(file_path, sep='\t', header=None, names=['child', 'parent', 'relation'])
    go_graph.add_edges_from(zip(df['child'], df['parent']))  # 调整为 child -> parent
    return go_graph


def is_cycle(graph, child, parent):
    """
    检查添加 child -> parent 边是否会形成环
    """
    if child in nx.ancestors(graph, parent):  # 如果 child 是 parent 的祖先，则形成环
        return True
    return False


def remove_redundant_edges(graph):
    """
    移除冗余边
    """
    redundant_edges = []

    # 遍历每个节点的所有父节点
    for node in list(graph.nodes):
        predecessors = list(graph.predecessors(node))
        if len(predecessors) > 1:  # 可能存在冗余
            for i, parent1 in enumerate(predecessors):
                for parent2 in predecessors[i + 1:]:
                    if parent2 in nx.ancestors(graph, parent1):
                        redundant_edges.append((parent1, node))
                    elif parent1 in nx.ancestors(graph, parent2):
                        redundant_edges.append((parent2, node))

    # 移除冗余边
    for edge in redundant_edges:
        if graph.has_edge(*edge):
            graph.remove_edge(*edge)


def main():
    # 加载数据
    go_graph = load_go_graph("../data/go_2022/is_a_relations.csv")
    new_relations = pd.read_csv("prediction_data/go_2022/fillter/combined_top_pre.csv", sep='\t', header=None,
                                names=['child', 'parent'])

    successful_relations = []

    # 添加新关系并检查环
    for _, row in new_relations.iterrows():
        child, parent = row['child'], row['parent']
        if parent in go_graph.nodes and child in go_graph.nodes:
            if not is_cycle(go_graph, child, parent):  # 检查是否成环
                go_graph.add_edge(child, parent)  # 添加 child -> parent 边
                successful_relations.append((child, parent))  # 记录成功添加的关系

    # 保存成功添加的 IS_A 关系
    successful_relations_df = pd.DataFrame(successful_relations, columns=['child', 'parent'])
    successful_relations_df.to_csv("successful_is_a_relations.csv", sep='\t', index=False, header=False)

    # 移除冗余边
    remove_redundant_edges(go_graph)

    # 输出更新后的 GO 图
    updated_relations = pd.DataFrame(list(go_graph.edges()), columns=['child', 'parent'])
    updated_relations.to_csv("updated_go_relations.csv", sep='\t', index=False, header=False)

    print(f"更新后的 GO 图已保存到 'updated_go_relations.csv'")
    print(f"成功添加的 IS_A 关系已保存到 'successful_is_a_relations.csv'")


if __name__ == "__main__":
    main()