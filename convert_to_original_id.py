# convert_to_original_id.py
import os

DATA_DIR = 'data/mind-news'
PRUNED_KG_FILE = 'saved_kg/mind-news_kgpr_70.txt'  # ここは評価したいファイル名に変えてください
OUTPUT_FILE = 'saved_kg/pruned_original_ids.txt'

# 1. リマップ辞書の読み込み（remap_id -> org_id）
remap2news = {}
with open(os.path.join(DATA_DIR, 'item_list.txt'), 'r') as f:
    next(f) # ヘッダーをスキップ
    for line in f:
        org_id, remap_id = line.strip().split()
        remap2news[remap_id] = org_id

remap2ent = {}
with open(os.path.join(DATA_DIR, 'entity_list.txt'), 'r') as f:
    next(f) # ヘッダーをスキップ
    for line in f:
        org_id, remap_id = line.strip().split()
        remap2ent[remap_id] = org_id

# 2. プルーニング済みKGの読み込みと変換
valid_pairs = set()
with open(PRUNED_KG_FILE, 'r') as f:
    for line in f:
        h, r, t = line.strip().split()
        
        # News(h) -> Entity(t) のエッジのみを抽出（Entity -> Entity はDivHGNNの内部で処理されるため一旦無視でOK）
        if h in remap2news and t in remap2ent:
            news_id = remap2news[h]
            ent_id = remap2ent[t]
            valid_pairs.add(f"{news_id}\t{ent_id}")

# 3. 結果の保存
with open(OUTPUT_FILE, 'w') as f:
    for pair in valid_pairs:
        f.write(pair + "\n")

print(f"変換完了！ {len(valid_pairs)} 件の News-Entity エッジを {OUTPUT_FILE} に保存しました。")