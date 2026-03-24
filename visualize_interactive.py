import networkx as nx
import os
import argparse
import pandas as pd
import json
import glob
import textwrap
import requests
import time
from pyvis.network import Network

def load_kg_to_networkx(file_path):
    G = nx.Graph()
    if not os.path.exists(file_path):
        print(f"❌ エラー: ファイルが見つかりません -> {file_path}")
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
            with open(path, "r", encoding="utf-8") as f:
                next(f)
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        remap2org[int(parts[1])] = parts[0]

    if dataset_name == "mind":
        org2text = {}
        news_cols = ['NewsID', 'Category', 'SubCategory', 'Title', 'Abstract', 'URL', 'TitleEntities', 'AbstractEntities']
        news_files = glob.glob("**/news.tsv", recursive=True)
        
        for p in news_files:
            try:
                df = pd.read_csv(p, sep='\t', names=news_cols)
                for _, row in df.iterrows():
                    nid = row['NewsID']
                    org2text[nid] = str(row['Title'])
                    for col in ['TitleEntities', 'AbstractEntities']:
                        if pd.notna(row[col]):
                            try:
                                ents = json.loads(row[col])
                                for ent in ents:
                                    if ent.get('WikidataId') and ent.get('Label'):
                                        org2text[ent['WikidataId']] = ent['Label']
                            except:
                                continue
            except:
                continue
        
        remap2text = {}
        for remap_id, org_id in remap2org.items():
            remap2text[remap_id] = org2text.get(org_id, org_id)
        print(f"--- マッピング完了: 合計 {len(remap2text)} 件 ---\n")
        return remap2text
        
    print(f"--- マッピング完了: 合計 {len(remap2org)} 件のIDを取得 ---\n")
    return remap2org

def fetch_freebase_names(freebase_ids):
    """Wikidata APIを使って、Freebase ID (m.xxx) を自然言語に変換 (50件ずつ分割処理)"""
    if not freebase_ids:
        return {}

    url = "https://query.wikidata.org/sparql"
    headers = {
        "User-Agent": "KGPR_Visualizer/2.0",
        "Accept": "application/json"
    }
    
    fb2name = {}
    chunk_size = 50 # 50件ずつに分割してリクエスト
    total_chunks = (len(freebase_ids) // chunk_size) + 1
    
    print(f"🌐 合計 {len(freebase_ids)} 件のIDを翻訳します (全 {total_chunks} 回の通信)...")
    
    for i in range(0, len(freebase_ids), chunk_size):
        chunk = freebase_ids[i:i+chunk_size]
        values_str = " ".join([f'"{fb.replace("m.", "/m/").replace("g.", "/g/")}"' for fb in chunk])
        
        # 英語優先で、なければ日本語や他言語を自動で持ってくる賢いクエリ
        query = f"""
        SELECT ?fb ?itemLabel WHERE {{
          VALUES ?fb {{ {values_str} }}
          ?item wdt:P646 ?fb .
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en,ja,es,fr,de". }}
        }}
        """
        try:
            response = requests.get(url, params={'query': query}, headers=headers, timeout=15)
            if response.status_code == 200:
                data = response.json()
                for item in data['results']['bindings']:
                    fb_val = item['fb']['value']
                    label_val = item['itemLabel']['value']
                    orig_fb = fb_val.replace("/m/", "m.").replace("/g/", "g.")
                    fb2name[orig_fb] = label_val
            
            # APIサーバーへの負荷を減らすため少し待機
            time.sleep(0.5)
            print(f"   ... 通信 {i//chunk_size + 1}/{total_chunks} 完了")
            
        except Exception as e:
            print(f"⚠️ 一部の翻訳通信でエラーが発生しました: {e}")
            
    return fb2name

def generate_interactive_html(org_kg_path, pruned_kg_path, target_node, labels_dict, hop=3, max_nodes=300):
    G_org = load_kg_to_networkx(org_kg_path)
    G_pruned = load_kg_to_networkx(pruned_kg_path)
    
    if target_node not in G_org:
        print(f"❌ 指定されたノードID '{target_node}' は元のグラフに存在しません。")
        return

    sub_G_org = nx.ego_graph(G_org, target_node, radius=hop)
    
    if sub_G_org.number_of_nodes() > max_nodes:
        print(f"ℹ️ ノードを上位{max_nodes}個に絞ります...")
        nodes_to_keep = [target_node] + [n for n in sub_G_org.nodes() if n != target_node][:max_nodes - 1]
        sub_G_org = sub_G_org.subgraph(nodes_to_keep).copy()

    fb_ids_to_query = []
    for n in sub_G_org.nodes():
        org_id = labels_dict.get(n, str(n))
        if isinstance(org_id, str) and (org_id.startswith("m.") or org_id.startswith("g.")):
            fb_ids_to_query.append(org_id)
            
    fb_names = {}
    if fb_ids_to_query:
        fb_names = fetch_freebase_names(fb_ids_to_query)
        print(f"✅ {len(fb_names)} 件の翻訳に成功しました！")

    net = Network(height="900px", width="100%", bgcolor="#ffffff", font_color="black", directed=False)
    
    for n in sub_G_org.nodes():
        org_id = labels_dict.get(n, str(n))
        display_name = fb_names.get(org_id, org_id)
        
        display_name = "\n".join(textwrap.wrap(display_name, width=25))
        is_pruned_away = not G_pruned.has_node(n) or G_pruned.degree(n) == 0
        
        if n == target_node:
            net.add_node(int(n), label="", title=f"【中心】\n{display_name}", color="red", size=25)
        elif is_pruned_away:
            net.add_node(int(n), label="", title=f"【切捨】\n{display_name}", color="#e0e0e0", size=10)
        else:
            net.add_node(int(n), label="", title=f"【重要】\n{display_name}", color="#32CD32", size=18)

    for u, v in sub_G_org.edges():
        if G_pruned.has_edge(u, v):
            net.add_edge(int(u), int(v), color="#32CD32", width=3, title="残った関係性")
        else:
            net.add_edge(int(u), int(v), color="#f0f0f0", width=1, title="切り捨てられた関係性")

    net.repulsion(node_distance=100, central_gravity=0.05, spring_length=100, spring_strength=0.05, damping=0.09)
    
    output_html = f"interactive_kg_node{target_node}.html"
    net.write_html(output_html)
    print(f"🎉 グラフを作成しました！ブラウザで開いてください -> {output_html}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mind", help="データセット名")
    parser.add_argument("--ratio", type=int, default=70, help="枝刈りのパーセンタイル")
    parser.add_argument("--node", type=int, default=0, help="中心のノードID")
    parser.add_argument("--hop", type=int, default=3, help="何ホップ先まで表示するか")
    parser.add_argument("--max_nodes", type=int, default=300, help="最大ノード数")
    args = parser.parse_args()

    ORG_KG_FILE = f"data/{args.dataset}/kg_final.txt"
    PRUNED_KG_FILE = f"data/{args.dataset}/{args.dataset}_kgpr_{args.ratio}.txt"

    labels_mapping = build_label_mapping(args.dataset)

    generate_interactive_html(
        org_kg_path=ORG_KG_FILE,
        pruned_kg_path=PRUNED_KG_FILE,
        target_node=args.node,
        labels_dict=labels_mapping,
        hop=args.hop,
        max_nodes=args.max_nodes
    )