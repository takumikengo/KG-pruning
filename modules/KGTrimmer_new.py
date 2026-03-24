'''
Created on July 1, 2020
PyTorch Implementation of KGIN
@author: Tinglin Huang (tinglin.huang@zju.edu.cn)
'''
__author__ = "huangtinglin"

import random
import numpy as np
import math
import torch
import os
import pandas as pd
import datetime
from tqdm import tqdm
from time import time
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_add, scatter_sum, scatter_max, scatter_min
from torch_scatter.utils import broadcast
from torch_scatter.composite import scatter_softmax
from collections import defaultdict

def torch_isin(a, b, device=None):
        _, a_counts = a.unique(return_counts=True)
        a_cat_b, combined_counts = torch.cat([a, b]).unique(return_counts=True)
        return (combined_counts - a_counts).gt(0)[a]
def torch_np_isin(a, b, device=None):
        a = a.detach().cpu().numpy()
        b = b.detach().cpu().numpy()
        return torch.BoolTensor(np.isin(a, b)).to(device)
class Aggregator(nn.Module):
    """
    Relational Path-aware Convolution Network
    1ホップ分の集約を行う
    """
    def __init__(self, n_users):
        super(Aggregator, self).__init__()
        self.linear = nn.Linear(64, 64)
        self.gate1 = nn.Linear(64, 64, bias=False)
        self.gate2 = nn.Linear(64, 64, bias=False)
        self.n_users = n_users

    def forward(self, entity_emb, user_emb, entity_2nd_emb, ent_weight_emb,
                edge_index, edge_type, interact_mat, weight, unmask):

        n_entities = entity_emb.shape[0]
        channel = entity_emb.shape[1]
        n_users = self.n_users

        """KG aggregate"""
        ## all triples
        head, tail = edge_index  #エッジからヘッドとテイルのインデックスを取り出し、そのつながりの関係性のベクトルを取得
        edge_relation_emb = weight[edge_type - 1]  # exclude interact, remap [1, n_relations) to [0, n_relations-1)
        # corresponding tail entity * relation 
        # 隣のノードから伝わってくるメッセージ=テイル側のエンティティのベクトル*リレーションベクトル*エッジマスク(重要度スコア)、これによってスコアで剪定
        neigh_relation_emb = (entity_emb)[tail] * (unmask.unsqueeze(-1))  * edge_relation_emb   # + entity_emb[head] # [-1, channel]
        
        #  GCN、scatter_sumすることで同じヘッドに向かってくるメッセージを集約し、新たなエンティティ表現を作る
        entity_agg = scatter_sum(src=neigh_relation_emb, index=head, dim_size=n_entities, dim=0)

        """user aggregate"""
        """
        interact_matはユーザとアイテムのインタラクションを表す疎行列であり、過去にインタラクションした
        全アイテムのベクトルを足し合わせたものをユーザの新たな表現とする
        """
        user_agg = torch.sparse.mm(interact_mat, entity_emb)

        return entity_agg, user_agg


class GraphConv(nn.Module):
    """
    Graph Convolutional Network
    """
    def __init__(self, channel, n_hops, n_users,n_entities, n_relations, interact_mat,
                node_dropout_rate=0.5, mess_dropout_rate=0.1):
        super(GraphConv, self).__init__()

        self.convs = nn.ModuleList()
        self.interact_mat = interact_mat
        self.n_relations = n_relations
        self.n_users = n_users
        self.node_dropout_rate = node_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate
        self.ent_gru = nn.GRU(channel, channel)

        self.ent_linear = nn.Linear(128, 64)
        self.user_linear = nn.Linear(128, 64)
        initializer = nn.init.xavier_uniform_
        

        for i in range(n_hops):
            self.convs.append(Aggregator(n_users=n_users))

        self.dropout = nn.Dropout(p=mess_dropout_rate)  # mess dropout


    

    def forward(self, user_emb, entity_emb, entity_2nd_emb, user_2nd_emb, edge_index, edge_type,
                interact_mat, weight, triplet_mask,  mess_dropout=True, node_dropout=False, training_epoch=-1):

        entity_res_emb = entity_emb  # [n_entity, channel]
        entity_2nd_res_emb = entity_2nd_emb
        user_res_emb = user_emb   # [n_users, channel]
        user_2nd_res_emb = user_emb
        entity_layer_embed = []
        user_layer_embed = []
        
        unmask = triplet_mask
        for i in range(len(self.convs)):
            entity_emb, user_emb = self.convs[i](entity_emb, user_emb, entity_2nd_emb, user_2nd_emb,
                                                 edge_index, edge_type, interact_mat, weight, unmask)

            """message dropout"""
            if mess_dropout:
                # entity_2nd_emb = self.dropout(entity_2nd_emb)
                entity_emb = self.dropout(entity_emb)
                user_emb = self.dropout(user_emb)
                # user_2nd_emb = self.dropout(user_2nd_emb)
            
            entity_layer_embed.append(entity_emb)
            user_layer_embed.append(user_emb)

            entity_emb = F.normalize(entity_emb)
            user_emb = F.normalize(user_emb)
            
            # weight = (len(self.convs) - i) / len(self.convs)
            """result emb"""
            entity_res_emb = torch.add(entity_res_emb, entity_emb )
            user_res_emb = torch.add(user_res_emb, user_emb)

        if training_epoch >= 0:
            return entity_res_emb, user_res_emb
        else:
            return entity_res_emb, user_res_emb, unmask


class ScoreEstimator(nn.Module):

    def __init__(self, dim):
        super(ScoreEstimator, self).__init__()
        self.dim = dim
        self.layer1 = nn.Linear(self.dim, self.dim ,bias=False)
        self.layer2 = nn.Linear(self.dim , 1, bias=False)
        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, ent_embeddings, head, tail, dim_size):
        
        x = F.relu(self.layer1(ent_embeddings))
        
        ent_scores = F.sigmoid(self.layer2(x))

        return ent_scores.squeeze(1)

class Recommender(nn.Module):
    def __init__(self, data_config, args_config, graph, adj_mat):
        super(Recommender, self).__init__()

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_relations = data_config['n_relations']
        self.n_entities = data_config['n_entities']  # include items
        self.n_nodes = data_config['n_nodes']  # n_users + n_entities
        self.epoch = args_config.epoch
        self.decay = args_config.l2
        self.save_dir = args_config.save_dir

        self.args_config = args_config
        self.sim_decay = args_config.sim_regularity
        self.emb_size = args_config.dim
        self.context_hops = args_config.context_hops
        self.node_dropout = args_config.node_dropout
        self.node_dropout_rate = args_config.node_dropout_rate
        self.mess_dropout = args_config.mess_dropout
        self.mess_dropout_rate = args_config.mess_dropout_rate
        self.device = torch.device("cuda:" + str(args_config.gpu_id)) if args_config.cuda \
                                                                      else torch.device("cpu")
        self.gamma = args_config.gamma

        self.adj_mat = adj_mat
        self.graph = graph
        self.edge_index, self.edge_type,  = self._get_edges(graph)
        self.need_sampled_entity_idx = None

        self.edge_cnt = self.edge_index.shape[1]
        
        self.triplet_mask = torch.zeros(self.n_entities) + 0.5


        self.triplet_mask = nn.Parameter(self.triplet_mask)


        ### tail to edge_index map, in order to mapping the user specific mask
        self.final_masked_edges = torch.LongTensor(range(self.edge_index.shape[1]))
        self.saved_ent_scores = np.empty((0, self.n_entities))
        self.q_mask = torch.zeros(self.edge_cnt).to(self.device)
        # # self.q_mask = nn.Parameter(self.q_mask)
        self.q_mask.requires_grad = False
        # prioritize the no mask
        
        self._init_weight()
        i = self.interact_mat._indices()
        v = torch.FloatTensor([1.0] * i.shape[1]).to(self.device)
        self.item_interact_mat = torch.sparse.FloatTensor(i, v, torch.Size((self.n_users, self.n_items))).to(self.device)
        self._init_user2ent()

        self.all_embed = nn.Parameter(self.all_embed)
        self.weight = nn.Parameter(self.weight)  
        self.gcn = self._init_model()
        self.score_estimator = ScoreEstimator(self.emb_size)

    def _init_user2ent(self):
        ## create user-edge sparse matrix
        # if not os.path.exists("user2ent"):
        #     os.makedirs("user2ent")
        user_edge_path = os.path.join("data", self.args_config.dataset, f"{self.args_config.dataset}_{self.args_config.num_sample_user2ent}.pt")
        
        sample_user2ent_num = self.args_config.num_sample_user2ent
        if not os.path.exists(user_edge_path):
            all_head, all_tail = self.edge_index.detach().cpu().numpy()
            ent2user = [np.array([], dtype=int) for _ in range(self.n_entities)]
            interact_idx = self.item_interact_mat._indices().detach().cpu().numpy()
            for t in interact_idx.T: 
                # ent2user[t[1]].add(t[0])
                ent2user[t[1]] = np.append(ent2user[t[1]], np.array(t[0], dtype=int))
            head = np.arange(self.n_items)
            n_users = self.n_users
            def propagate(head, n_users, ent2user, depth=0):
                if depth==3:
                    return ent2user
                threshold = n_users // 3
                # edge_idxs = np.where((np.isin(all_head, head)))[0]
                uq_tail = np.unique(all_tail[np.isin(all_head, head)])
                next_head = []
                for t in tqdm(uq_tail, total=len(uq_tail)):
                    if len(ent2user[t]) > 0:
                        continue
                    next_head.append(t)
                    ent2user[t] = np.unique(np.concatenate([ent2user[h] for h in np.unique(all_head[all_tail == t])]))

                next_head = np.unique(np.array(next_head))
                return propagate(next_head, n_users, ent2user, depth= depth+1)
            ent2user = propagate(head, n_users, ent2user)
            row, col = [], []
            for i, users in tqdm(enumerate(ent2user), total=len(ent2user)):
                if len(users) <= sample_user2ent_num:
                    row += list(users)
                    col += [i] * len(users)
                else:
                    
                    row += list(np.random.choice(np.array(list(users)), size=sample_user2ent_num, replace=False))
                    col += [i] * sample_user2ent_num
            
            row = torch.LongTensor(row)
            col = torch.LongTensor(col)
            idx = torch.stack([row, col], dim=0)
            torch.save(idx, user_edge_path)
        else:
            idx = torch.load(user_edge_path, map_location=self.device)
        
        log_path = os.path.join("data", self.args_config.dataset, "user_item_time_log.tsv")
        decay_dict = {} # (user, item) -> weight
        
        if os.path.exists(log_path):
            print("時間ログを読み込み、減衰スコアを計算します...")
            df_log = pd.read_csv(log_path, sep='\t')
            # "unknown" を除外して最新の時刻を基準にするための処理
            df_valid = df_log[df_log['time'] != 'unknown'].copy()
            df_valid['time'] = pd.to_datetime(df_valid['time'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
            
            if not df_valid.empty:
                max_time = df_valid['time'].max()
                beta = 0.1 # 減衰係数（1日で約10%減衰）
                
                for _, row in df_log.iterrows():
                    u = int(row['user_id'])
                    i = int(row['item_id'])
                    if row['time'] == 'unknown':
                        decay_dict[(u, i)] = 0.1 # 過去のHistoryは低い重みで固定
                    else:
                        t = pd.to_datetime(row['time'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
                        if pd.notna(t):
                            diff_days = (max_time - t).total_seconds() / (3600 * 24)
                            weight = np.exp(-beta * diff_days)
                            decay_dict[(u, i)] = max(weight, 0.1) # 最小でも0.1は残す
                        else:
                            decay_dict[(u, i)] = 0.1
            else:
                print("有効な時間データがありませんでした。")
        else:
            print(f"警告: {log_path} が見つかりません。時間減衰は一律1.0になります。")

        # =======================================================

        # user-item (直接のインタラクション) のエッジを取得
        interact_idx = self.interact_mat._indices().detach().cpu()
        
        # 全結合 (サンプリングされた user-entity と、直接の user-item)
        idx_combined = torch.cat([idx, interact_idx], dim=-1).to(self.device)
        
        # ========== 【追加】Values(重み)配列の生成 ==========
        v_list = []
        for i in range(idx_combined.shape[1]):
            u = int(idx_combined[0, i])
            ent_or_item = int(idx_combined[1, i])
            
            # アイテムに対する直接のつながりの場合は時間減衰スコアを適用
            if (u, ent_or_item) in decay_dict:
                v_list.append(decay_dict[(u, ent_or_item)])
            else:
                # user-entityのマルチホップパスの場合は、対象ユーザーの平均重みなどを適用(簡易的に0.5とする)
                v_list.append(0.5) 
                
        v = torch.FloatTensor(v_list).to(self.device)
        # ====================================================
        self.user2ent_mat = torch.sparse.FloatTensor(idx_combined, v, torch.Size([self.n_users, self.n_entities])).to(self.device)

    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        self.all_embed = initializer(torch.empty(self.n_nodes, self.emb_size))
        self.user_2nd_emb = initializer(torch.empty(self.n_users, self.emb_size))
        self.ent_weight_emb = initializer(torch.empty(self.n_entities, self.emb_size))
        self.entity_2nd_emb = initializer(torch.empty(self.n_entities, self.emb_size))
        self.weight = initializer(torch.empty(self.n_relations - 1, self.emb_size))  # not include interact # [n_relations - 1, in_channel]
        # [n_users, n_entities]
        self.interact_mat = self._convert_sp_mat_to_sp_tensor(self.adj_mat).to(self.device)

    def _init_model(self):
        return GraphConv(channel=self.emb_size,
                         n_hops=self.context_hops,
                         n_users=self.n_users,
                         n_entities=self.n_entities,
                         n_relations=self.n_relations,
                         interact_mat=self.interact_mat,
                         node_dropout_rate=self.node_dropout_rate,
                         mess_dropout_rate=self.mess_dropout_rate,
                         )

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def _get_indices(self, X):
        coo = X.tocoo()
        return torch.LongTensor([coo.row, coo.col]).t()  # [-1, 2]

    def _get_edges(self, graph):
        graph_tensor = torch.tensor(list(graph.edges))  # [-1, 3]

        
        index = graph_tensor[:, :-1]  # [-1, 2]
        type = graph_tensor[:, -1]  # [-1, 1]
        return index.t().long().to(self.device), type.long().to(self.device)

    def _edge_sampling(self, edge_index, edge_type, triplet_mask, rate=0.5):
        n_edges = edge_index.shape[1]
        random_indices = np.random.choice(n_edges, size=int(n_edges * rate), replace=False)
        # rebuild graph to only keep the undropped edges
        return edge_index[:, random_indices], edge_type[random_indices], triplet_mask[random_indices],  random_indices

    def _sparse_dropout(self, x, rate=0.5):
        noise_shape = x._nnz()

        random_tensor = rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))

    def generate_user_specific_mask(self, batch_u2e_mat, user_ids, edge_index, edge_type, random_indices=None, training_epoch=-1):
        head, tail = edge_index.detach()
        user_embed = self.all_embed[user_ids].detach()
        normalized_user_embed = user_embed / torch.norm(user_embed, dim=1).unsqueeze(1)
        entity_emb = self.all_embed[self.n_users:].detach()

        edge_relation_emb = self.weight[edge_type - 1] 
        edge_relation_emb = edge_relation_emb.detach()
        neigh_relation_emb = entity_emb[tail] * edge_relation_emb
        ent_agg = scatter_mean(src=neigh_relation_emb, index=tail, dim_size=self.n_entities, dim=0)
        
        batch_idx = batch_u2e_mat._indices()
        
        # ========== 【追加1】時間減衰重みの適用 ==========
        # 前処理で渡された batch_u2e_mat の values() に時間減衰スコアが入っている前提
        batch_values = batch_u2e_mat._values()
        
        values = F.relu(torch.cosine_similarity(user_embed[batch_idx[0]], ent_agg[batch_idx[1]])) + 1e-8
        
        # 単純な平均ではなく、時間減衰の重み付き平均(Weighted Mean)でスコアを計算
        weighted_values = values * batch_values
        sum_weights = scatter_sum(src=batch_values, index=batch_idx[1], dim_size=self.n_entities) + 1e-8
        ent_score = scatter_sum(src=weighted_values, index=batch_idx[1], dim_size=self.n_entities) / sum_weights
        # ===============================================

        # ========== 【追加2】知識グラフ上のスコア伝播 (コールドスタート対策) ==========
        alpha = 0.3  # 伝播の強さ（ハイパーパラメータ：0.1〜0.5あたりで調整）
        # knowledge graphのエッジ(head -> tail)を使って、隣接ノードのスコアを集約
        neigh_scores = ent_score[head]
        propagated_score = scatter_mean(src=neigh_scores, index=tail, dim_size=self.n_entities, dim=0)
        propagated_score = torch.nan_to_num(propagated_score, nan=0.0) # NaN防止
        
        # 自身のスコアと伝播スコアをブレンド
        ent_score = (1 - alpha) * ent_score + alpha * propagated_score
        # ====================================================================

        idx = ent_score == 0
        
        # 注意: 元コードでは triplet_mask の値が大きい＝マスクされる(スコア0) という反転構造
        triplet_mask = 1 - self.triplet_mask
        triplet_mask = torch.clamp(triplet_mask,0, 1)
        gamma = self.gamma
        
        # 伝播済みのスコアを使用
        final_ent_score = ent_score.clone()

        final_ent_score = final_ent_score * gamma + triplet_mask * (1-gamma)
        final_ent_score = torch.clamp(final_ent_score, 0, 1)

        ## ignore the entities which are unseen in 3-hops
        if training_epoch == -1:
            final_ent_score[idx] = 0

        edge_score = final_ent_score[tail]
        
        return edge_score, final_ent_score
    

    def forward(self, batch=None, training_epoch=-1):
        user = batch['users']
        pos_item = batch['pos_items']
        neg_item = batch['neg_items']
        
        user_emb = self.all_embed[:self.n_users, :]
        item_emb = self.all_embed[self.n_users:, :]

        
        # select the users in batch, (tips: sparse matrix do not support index operation)
        # remove the duplicated users
        start_time = time()
        u = np.unique(user.detach().cpu().numpy())
        idx = np.array([[i for i in range(u.shape[0])], u])
        idx = torch.LongTensor(idx).to(self.device)
        v = torch.FloatTensor([1.0] * idx.shape[1]).to(self.device)
        u2batch = torch.sparse.FloatTensor(idx, v, torch.Size((idx.shape[1], self.n_users)))
        # ## shape: (unique_users, edge)
        batch_u2e_mat = torch.sparse.mm(u2batch, self.user2ent_mat)
        edge_index, edge_type, triplet_mask= self.edge_index, self.edge_type, self.triplet_mask
        
        
        triplet_mask, _ = self.generate_user_specific_mask(batch_u2e_mat, idx[1], edge_index, edge_type, random_indices=None, training_epoch=training_epoch)
        create_mask_time = time() - start_time
        removed_edges = torch.where(triplet_mask > 0)[0]
        edge_index, edge_type, triplet_mask = edge_index[:, removed_edges], edge_type[removed_edges], triplet_mask[removed_edges]
        if self.node_dropout:
            edge_index, edge_type, triplet_mask, random_indices = self._edge_sampling(edge_index, edge_type, triplet_mask, self.node_dropout_rate)
            interact_mat = self._sparse_dropout(self.interact_mat, self.node_dropout_rate)

        # entity_gcn_emb: [n_entity, channel]
        # user_gcn_emb: [n_users, channel]
        entity_gcn_emb, user_gcn_emb = self.gcn(user_emb,
                                                     item_emb,
                                                     self.entity_2nd_emb,
                                                     self.user_2nd_emb,
                                                     edge_index,
                                                     edge_type,
                                                     interact_mat,
                                                     self.weight,
                                                     triplet_mask,
                                                     mess_dropout=self.mess_dropout,
                                                     node_dropout=self.node_dropout,
                                                     training_epoch=training_epoch)

        u_e = user_gcn_emb[user]
        pos_e, neg_e = entity_gcn_emb[pos_item], entity_gcn_emb[neg_item.view(-1)]
        total_loss , mf_loss, emb_loss = self.create_bpr_loss(u_e, pos_e, neg_e) 
        return total_loss, mf_loss, emb_loss, create_mask_time
    def generate(self, eval_epoch=-1):
        user_emb = self.all_embed[:self.n_users, :]
        item_emb = self.all_embed[self.n_users:, :]
        
        edge_index, edge_type, triplet_mask = self.edge_index, self.edge_type, self.triplet_mask
        triplet_mask, _ = self.getmask()

        return self.gcn(user_emb,
                        item_emb,
                        self.entity_2nd_emb,
                        self.user_2nd_emb,
                        self.edge_index,
                        self.edge_type,
                        self.interact_mat,
                        self.weight,
                        triplet_mask,
                        mess_dropout=False, node_dropout=False,
                        training_epoch=-1)[:3]

    def resample_item2node(self, training_epoch):
        # self.item2ent_graph = self.build_item2entities_graph(self.edge_index)
        pass

    def update_q_mask(self, eval_epoch):
        mask, ent_score = self.getmask()

        self.saved_ent_scores = np.concatenate([self.saved_ent_scores, ent_score.detach().unsqueeze(0).cpu().numpy()], axis=0)

        return self.saved_ent_scores
        
    def getmask(self):
        user_ids = torch.LongTensor(list(range(self.n_users))).to(self.device)
        batch_u2e_mat = self.user2ent_mat
        # batch_u2i_mat = torch.sparse.mm(user_ids, self.item_interact_mat)
        # batch_u2e_mat = torch.sparse.mm(self.item_interact_mat, self.item2ent_graph)
        edge_index, edge_type, triplet_mask = self.edge_index, self.edge_type, self.triplet_mask
        return self.generate_user_specific_mask(batch_u2e_mat, user_ids, edge_index, edge_type)

    def rating(self, u_g_embeddings, i_g_embeddings):
        # return torch.cosine_similarity(u_g_embeddings.unsqueeze(1), i_g_embeddings.unsqueeze(0), dim=2)
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())

    def create_bpr_loss(self, users, pos_items, neg_items):
        batch_size = users.shape[0]
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)
        mf_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))

        regularizer = (torch.norm(users) ** 2
                       + torch.norm(pos_items) ** 2
                       + torch.norm(neg_items) ** 2) / 2
        emb_loss = self.decay * regularizer / batch_size

        return mf_loss + emb_loss , mf_loss, emb_loss
