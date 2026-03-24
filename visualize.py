import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import pandas as pd
import json
import glob
import textwrap

def load_kg_to_networkx(file_path):
    G = nx.Graph()
    if not os.path.exists(file_path):
        print(f"❌ エラー: 知識グラフのファイルが見つかりません -> {file_path}")
        return G
        
    with open(file_path, 'r') as f:
        for line in f:
            h, r, t = map(int, line.strip().split())
            G.add_edge(h, t, relation=r)
    return G

def build_label_mapping(dataset_name):
    print("\n--- 言語ラベルのマッピングを作成中 ---")
    dataset_dir = f"data/{dataset_name}"
    remap2org = {}
    
    for list_file in ["item_list.txt", "entity_list.txt"]:
        path = os.path.join(dataset_dir, list_file)
        if os.path.exists(path):
            count = 0
            with open(path, "r", encoding="utf-8") as f:
                next(f)
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        remap2org[int(parts[1])] = parts[0]
                        count += 1
            print(f"✅ {list_file} から {count} 件のIDを読み込みました。")
        else:
            print(f"⚠️ 警告: {path} が見つかりません。")

    if dataset_name != "mind":
        print(f"ℹ️ MIND以外のデータセットのため、元のIDをラベルとして使用します。\n")
        return remap2org

    org2text = {}
    news_cols = ['NewsID', 'Category', 'SubCategory', 'Title', 'Abstract', 'URL', 'TitleEntities', 'AbstractEntities']
    
    print("🔍 news.tsv を探索中...")
    news_files = glob.glob("**/news.tsv", recursive=True)
    
    if not news_files:
        print("❌ 警告: 'news.tsv' が現在のフォルダ以下のどこにも見つかりません！")
    else:
        for p in news_files:
            print(f"✅ 【抽出中】{p} を発見しました！")
            try:
                df = pd.read_csv(p, sep='\t', names=news_cols)
                for _, row in df.iterrows():
                    nid = row['NewsID']
                    title = str(row['Title'])
                    title = textwrap.fill(title, width=15)
                    org2text[nid] = title
                    
                    for col in ['TitleEntities', 'AbstractEntities']:
                        if pd.notna(row[col]):
                            try:
                                ents = json.loads(row[col])
                                for ent in ents:
                                    w_id = ent.get('WikidataId')
                                    label = ent.get('Label')
                                    if w_id and label:
                                        org2text[w_id] = textwrap.fill(label, width=15)
                            except:
                                continue
            except Exception as e:
                print(f"⚠️ {p} の読み込み中にエラーが発生しました: {e}")
        
    remap2text = {}
    for remap_id, org_id in remap2org.items():
        remap2text[remap_id] = org2text.get(org_id, org_id)
        
    print(f"--- マッピング完了: 合計 {len(remap2text)} 件のテキストラベルを準備しました ---\n")
    return remap2text

def plot_subgraph_comparison(org_kg_path, pruned_kg_path, target_node, labels_dict, hop=2, max_nodes=30):
    G_org = load_kg_to_networkx(org_kg_path)
    G_pruned = load_kg_to_networkx(pruned_kg_path)
    
    if G_org.number_of_nodes() == 0 or G_pruned.number_of_nodes() == 0:
        return

    if target_node not in G_org:
        print(f"❌ 指定されたノードID '{target_node}' は元のグラフに存在しません。")
        return

    sub_G_org = nx.ego_graph(G_org, target_node, radius=hop)
    
    if sub_G_org.number_of_nodes() > max_nodes:
        print(f"ℹ️ ノード数が多すぎるため、中心から近い上位{max_nodes}ノードに絞ります。")
        # ★ 修正ポイント: ターゲットノードを確実に保持しつつ上限までノードを追加する ★
        nodes_to_keep = [target_node] + [n for n in sub_G_org.nodes() if n != target_node][:max_nodes - 1]
        sub_G_org = sub_G_org.subgraph(nodes_to_keep)

    # 左側と同じ丸を右側にも用意する
    sub_G_pruned = nx.Graph()
    sub_G_pruned.add_nodes_from(sub_G_org.nodes())
    for u, v in sub_G_org.edges():
        if G_pruned.has_edge(u, v):
            sub_G_pruned.add_edge(u, v)

    plt.figure(figsize=(24, 12))
    pos = nx.spring_layout(sub_G_org, seed=42, k=0.5)

    node_labels = {node: labels_dict.get(node, str(node)) for node in sub_G_org.nodes()}

    # --- 左側：プルーニング前 ---
    plt.subplot(1, 2, 1)
    plt.title("Original Knowledge Graph", fontsize=18)
    nx.draw_networkx_nodes(sub_G_org, pos, node_size=400, node_color='lightblue')
    nx.draw_networkx_edges(sub_G_org, pos, alpha=0.3, edge_color='gray')
    nx.draw_networkx_labels(sub_G_org, pos, labels=node_labels, font_size=9, font_weight='bold')
    nx.draw_networkx_nodes(sub_G_org, pos, nodelist=[target_node], node_size=800, node_color='red')

    # --- 右側：プルーニング後 ---
    plt.subplot(1, 2, 2)
    plt.title(f"Pruned Knowledge Graph (Ratio: {args.ratio}%)", fontsize=18)
    
    connected_nodes = [n for n in sub_G_pruned.nodes() if sub_G_pruned.degree(n) > 0 and n != target_node]
    isolated_nodes = [n for n in sub_G_pruned.nodes() if sub_G_pruned.degree(n) == 0 and n != target_node]
    
    nx.draw_networkx_nodes(sub_G_pruned, pos, nodelist=connected_nodes, node_size=400, node_color='lightgreen')
    nx.draw_networkx_nodes(sub_G_pruned, pos, nodelist=isolated_nodes, node_size=400, node_color='lightgray') # 灰色で描画
    nx.draw_networkx_edges(sub_G_pruned, pos, alpha=0.7, edge_color='black', width=1.5)
    nx.draw_networkx_labels(sub_G_pruned, pos, labels=node_labels, font_size=9, font_weight='bold')
    nx.draw_networkx_nodes(sub_G_pruned, pos, nodelist=[target_node], node_size=800, node_color='red')

    plt.tight_layout()
    output_filename = f"kg_comparison_text_node{target_node}.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"🎉 グラフを画像として保存しました -> {output_filename}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mind", help="データセット名")
    parser.add_argument("--ratio", type=int, default=70, help="比較する枝刈りのパーセンタイル")
    parser.add_argument("--node", type=int, default=0, help="グラフの中心に配置するノードID")
    parser.add_argument("--hop", type=int, default=2, help="中心ノードから何ホップ先まで表示するか")
    parser.add_argument("--max_nodes", type=int, default=30, help="描画する最大ノード数")
    args = parser.parse_args()

    ORG_KG_FILE = f"data/{args.dataset}/kg_final.txt"
    PRUNED_KG_FILE = f"data/{args.dataset}/{args.dataset}_kgpr_{args.ratio}.txt"

    labels_mapping = build_label_mapping(args.dataset)

    plot_subgraph_comparison(
        org_kg_path=ORG_KG_FILE,
        pruned_kg_path=PRUNED_KG_FILE,
        target_node=args.node,
        labels_dict=labels_mapping,
        hop=args.hop,
        max_nodes=args.max_nodes
    )