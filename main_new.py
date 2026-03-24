'''
Created on July 1, 2020
@author: Tinglin Huang (tinglin.huang@zju.edu.cn)
'''
__author__ = "huangtinglin"

import random

import json
import torch
import numpy as np
import os
from time import time
from prettytable import PrettyTable
from tqdm import tqdm
from utils.parser import parse_args
from utils.data_loader import load_data

from utils.evaluate import test, evaluate, save_unpruned_node, get_orginal_kg, get_masked_info
from utils.helper import early_stopping
import pandas as pd
n_users = 0
n_items = 0
n_entities = 0
n_nodes = 0
n_relations = 0



def get_feed_dict(train_entity_pairs, start, end, train_user_set, n_items):  #学習に使用するミニバッチの作成、ポジネガで構成

    def negative_sampling(user_item, train_user_set, n):
        neg_items = []
        # cand_rands = torch.randint(0, n_items, (user_item.shape[0], n)).numpy()
        for i, (user, _) in enumerate(user_item.cpu().numpy()):
            user = int(user)
            negitems = []
            # negitems = list(np.setdiff1d(cand_rands[i], train_user_set[user]))
            while len(negitems) < n:  # sample n times
                while True:
                    negitem = random.choice(range(n_items))
                    if negitem not in train_user_set[user]:
                        break
                negitems.append(negitem)
            neg_items.append(negitems)
        return neg_items
    feed_dict = {}
    entity_pairs = train_entity_pairs[start:end].to(device)
    feed_dict['users'] = entity_pairs[:, 0]
    feed_dict['pos_items'] = entity_pairs[:, 1]
    feed_dict['neg_items'] = torch.LongTensor(negative_sampling(entity_pairs,
                                                                train_user_set, 1)).to(device)
    return feed_dict

def save_args_config(args):  #学習時の設定をconfig.jsonに保存
    saved_config_file = os.path.join(args.save_dir, "config.json")
    arg_dict = json.dumps(args.__dict__)
    with open(saved_config_file, "w") as f:
        f.write(arg_dict)

def overwrite_config(args): #テストや予測時に保存されてるconfigを読み込み、学習時と設定を一致させる
    saved_config_file = os.path.join(args.save_dir, "config.json")
    if not os.path.exists(saved_config_file):
        return args
    with open(saved_config_file) as f:
        arg_org = json.loads(f.read())
    args.model = arg_org["model"]

    args.dataset = arg_org["dataset"]
    args.num_sample_user2ent = arg_org["num_sample_user2ent"]
    return args

def train(args):
    global device
    """fix the random seed"""
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_args_config(args)
    
    device = torch.device("cuda:"+str(args.gpu_id)) if args.cuda else torch.device("cpu")

    """build dataset"""
    train_cf, test_cf, user_dict, n_params, graph, mat_list = load_data(args)
    adj_mat_list, norm_mat_list, mean_mat_list = mat_list

    n_users = n_params['n_users']
    n_items = n_params['n_items']
    n_entities = n_params['n_entities']
    n_relations = n_params['n_relations']
    n_nodes = n_params['n_nodes']

    """cf data"""
    train_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in train_cf], np.int32))
    test_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in test_cf], np.int32))

    """define model"""
    
    if args.model == "KGTrimmer":
        from modules.KGTrimmer_new import Recommender
        model = Recommender(n_params, args, graph, mean_mat_list[0]).to(device)

    else:
        raise Exception(f"do not support this model:{args.model}")
    """define optimizer"""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    cur_best_pre_0 = 0
    stopping_step = 0
    should_stop = False

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    print("start training ...")
    
    for epoch in range(args.epoch):
        
        """training CF"""
        # # shuffle training data
        index = np.arange(len(train_cf))
        np.random.shuffle(index)
        train_cf_pairs = train_cf_pairs[index]
        
        # """training"""
        loss, s = 0, 0
        train_s_t = time()
        total_score_time = 0
        while s + args.batch_size <= len(train_cf_pairs):
            batch = {}
            batch = get_feed_dict(train_cf_pairs,
                                  s, s + args.batch_size,
                                  user_dict['train_user_set'], n_items)
            batch_loss, _, _, score_time = model(batch, epoch)
            total_score_time += score_time
            batch_loss = batch_loss
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss
            s += args.batch_size

        train_e_t = time()

        # 【追加箇所】 Holistic View の忘却処理（時間変化対策）
        lambda_decay = 0.98  # 忘却係数 (例: 1エポックで2%忘却する)
        with torch.no_grad():
            # KGPRのTriplet Maskは、1に近づくほど「プルーニング対象(価値なし)」になります。
            # エポックごとに、学習で更新されたマスクを少しずつ 1.0 に近づけて風化させます。
            model.triplet_mask.data = model.triplet_mask.data * lambda_decay + 1.0 * (1.0 - lambda_decay)
        
        ret = model.update_q_mask(epoch)
        if ret is not None:
            saved_ent_scores = ret
            np.save(os.path.join(save_dir, "saved_ent_scores.npy"), saved_ent_scores)
        
        
        if (epoch+1) % 10 == 0 or epoch == 1  :
            """testing"""
            test_s_t = time()
            ret = test(model, user_dict, n_params, epoch)
            # ret = evaluate(model, 2048, n_items, user_dict["train_user_set"], user_dict["test_user_set"], device)
            test_e_t = time()

            train_res = PrettyTable()
            train_res.field_names = ["Epoch", "training time", "tesing time", "Loss", "recall", "ndcg", "precision"]
            # ret = ret[20]
            train_res.add_row(
                [epoch, train_e_t - train_s_t, test_e_t - test_s_t, loss.item(), ret['recall'], ret['ndcg'], ret['precision']]
            )
            print(train_res, flush=True)

            # *********************************************************
            # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
            cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                        stopping_step, expected_order='acc',
                                                                        flag_step=10)
            if should_stop:
                break

            """save weight"""
            if ret['recall'][0] == cur_best_pre_0 and args.save:

                torch.save(model.state_dict(), os.path.join(save_dir, "checkpoint.ckpt"))
        else:
            # logging.info('training loss at epoch %d: %f' % (epoch, loss.item()))
            print('using time %.4f=%.4f+%.4f, training loss at epoch %d: %.4f' % (train_e_t - train_s_t, train_e_t - train_s_t- total_score_time, total_score_time, epoch, loss.item()), flush=True)
    print("Pruning kg...")
    recreate_kgfile_by_percentile(model, args, save_dir)
    print('early stopping at %d, recall@20:%.4f' % (epoch, cur_best_pre_0))


def predict(args):
    global device
    if args.pretrain_model_path:
        args.save_dir = args.pretrain_model_path

    args = overwrite_config(args)
    device = torch.device("cuda:"+str(args.gpu_id)) if args.cuda else torch.device("cpu")

    """build dataset"""
    train_cf, test_cf, user_dict, n_params, graph, mat_list = load_data(args)
    adj_mat_list, norm_mat_list, mean_mat_list = mat_list


    """cf data"""
    train_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in train_cf], np.int32))
    test_cf_pairs = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in test_cf], np.int32))

    if args.model == "KGTrimmer":
        from modules.KGTrimmer_new import Recommender
        model = Recommender(n_params, args, graph, mean_mat_list[0]).to(device)  #学習が完了したモデルを呼び出す
    else:
        raise Exception("unknow model")
    model = load_model(model, args.pretrain_model_path)
    recreate_kgfile_by_percentile(model, args) #学習時に保存されたスコアを元に知識グラフの再生成を行う
    #↓テストデータを使った評価
    test_s_t = time()
    ret = test(model, user_dict, n_params)
    test_e_t = time()

    train_res = PrettyTable()
    train_res.field_names = [ "tesing time", "recall", "ndcg", "precision", "hit_ratio"]
    train_res.add_row(
        [test_e_t - test_s_t, ret['recall'], ret['ndcg'], ret['precision'], ret['hit_ratio']]
    )
    print(train_res)
    # np.save(os.path.join(saved_path, "cf_score.npy"), np.array(ret["ratings"]))

#epoch_~でプルーニングする際に参照するスコアの算出方法を定める
#複数エポックの平均を取る方法
def epoch_mean(args, ent_scores, model):

    epochs= [5]

    for epoch in epochs:
        yield ent_scores[:epoch].mean(axis=0), f"{epoch}"

#指定した特定エポックのスコアを参照する方法
def epoch_single(args, ent_scores, model):

    epochs= [5]

    for epoch in epochs:
        yield ent_scores[epoch], f"{epoch}_str"

#一番最後のエポックのスコアを取る方法
def epoch_last(args, ent_scores, model):
    return [(ent_scores[-1], "last_epoch")]

def model_mask(args, ent_scores, model):
    return [(None, "mask")]

def recreate_kgfile_by_percentile(model, args, save_dir=None):
    save_dir = save_dir if save_dir is not None else args.pretrain_model_path
    saved_ent_scores = os.path.join(save_dir, "saved_ent_scores.npy")
    # epochs = [5, 2000]
    
    head, tail = model.edge_index
    ent_scores = np.load(saved_ent_scores)
    n_relation = (model.n_relations - 1) / 2 - 1
    edge_type = model.edge_type - 1
    org_index = torch.where(edge_type <= n_relation)[0]
    
    entity_emb = model.all_embed[model.n_users:]
    edge_relation_emb = model.weight[edge_type]  # exclude interact, remap [1, n_relations) to [0, n_relations-1)
    # corresponding tail entity * relation 
    neigh_relation_emb = entity_emb[tail] * edge_relation_emb
    # org_edges = self.mlp(torch.cat([self.all_embed[self.n_users:][tail], edge_relation_emb], dim=1))
    # org_edges = F.normalize(org_edges)
    from torch_scatter import scatter_mean
    ent_agg = scatter_mean(src=neigh_relation_emb, index=tail, dim_size=model.n_entities, dim=0)

    ops = ["epoch", "last"]
    tail_index = tail[org_index].detach().cpu().numpy()
    head_index = head[org_index].detach().cpu().numpy()
    tail = tail[org_index]
    head = head[org_index]
    methods =  ["sum",  "mul"]
    can_triplets_np = get_orginal_kg(model)

    op = ""
    pruning_ratio = [30, 70, 90]
    if args.dataset == "amazon-book":
        op = "epoch_single"  #どのエポックのスコアを使うか
        method_name = "mul"  #エッジのスコアをどう計算するか
        pruning_ratio = [90, 95, 98]  #元のグラフから何％のエッジを削るか
    elif args.dataset == "alibaba-fashion":
        op = "epoch"
        method_name = "mul"
        
    elif args.dataset == "last-fm":
        op = "epoch"
        method_name = "sum"
    elif args.dataset == "mind-news":   # ← ここから追加
        op = "epoch"               # last-fmなどと同じ標準的な設定に
        method_name = "sum"        # last-fmなどと同じ標準的な設定に
        pruning_ratio = [30, 70, 90]
    

    
    if op == "epoch":
        method = epoch_mean
    elif op == "epoch_single":
        method = epoch_single
    elif op == "last":
        method = epoch_last
    else:
        raise Exception("op error")
            
    for mean_scores, name in method(args, ent_scores, model):
        
        if method_name == "sum": # i=0 なので、tail_score（繋がれる側のエンティティスコア）だけを採用
            head_score, tail_score = mean_scores[head_index], mean_scores[tail_index]
            edge_scores = []
            for i in [0]:
                edge_scores.append((i * head_score + (1 - i) * tail_score, "beta" + str(i)))
        elif method_name == "min":
            edge_scores = np.min([mean_scores[tail_index], mean_scores[head_index]], axis=0)
        elif method_name == "mul": # i=0.5 なので、headとtailのスコアを掛け合わせてルートを取る（幾何平均）
            edge_scores = []
            head_score, tail_score = mean_scores[head_index], mean_scores[tail_index]
            for i in [ 0.5]:
                edge_scores.append(((head_score** i) * (tail_score** (1 - i)) , "beta" + str(i)))
            # edge_scores = mean_scores[tail_index] * mean_scores[head_index]
        else:
            raise Exception("not support method name")
        # edge_scores = 2 * 1 / ( 1 / (mean_scores[tail_index] + 1e-10) + 1 / (mean_scores[head_index] + 1e-10))
        # edge_scores = np.sqrt(mean_scores[tail_index] * mean_scores[head_index])
        # edge_scores = np.min([mean_scores[tail[org_index].detach().cpu().numpy()], mean_scores[head[org_index].detach().cpu().numpy()]], axis=0)
        
        save_kg_dir = "saved_kg"
        if not os.path.exists(save_kg_dir):
            os.makedirs(save_kg_dir)
        
        if type(edge_scores) != list:
            edge_scores = [(edge_scores, "n")]
        for edge_score, edge_name in edge_scores:
            for percentile in pruning_ratio:
                save_kg_file = os.path.join(save_kg_dir, f"{args.dataset}_kgpr_{percentile}.txt")
                saved_kg_idx = np.argsort(-edge_score)[:len(edge_score) * (100 - percentile) // 100]
                saved_triplets = can_triplets_np[saved_kg_idx]
                np.savetxt(save_kg_file, saved_triplets, fmt='%i')
        
    
def load_model(model, model_path):
    checkpoint = torch.load(os.path.join(model_path, "checkpoint.ckpt"), map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    model.eval()
    return model

if __name__ == '__main__':

    """read args"""
    global args
    args = parse_args()
    args.epoch = 6 #エポック数が6で固定されている、学習エポック数を変えたい場合はここをなくすこと！
    if args.dataset == "last-fm":
        args.num_sample_user2ent = 20 #ユーザからエンティティへのサンプリング数
        args.gamma = 0.7
        # args.batch_size = 8192
    elif args.dataset == "alibaba-fashion":
        args.num_sample_user2ent = 200 
        args.gamma = 0.3
        # args.batch_size =4096
    elif args.dataset == "amazon-book":
        args.gamma = 0.3
        # args.batch_size =4096
    elif args.dataset == "mind-news":       # ← ここから追加
        args.num_sample_user2ent = 100 # 一旦デフォルト値（必要に応じて調整）
        args.gamma = 0.5

    save_dir = "model_{}_{}_{}_{}".format(args.model, args.dataset, args.gamma, args.num_sample_user2ent)
    save_dir = os.path.join(args.out_dir,save_dir)
    args.save_dir = save_dir
    if not args.pretrain_model_path:
        print("training")
        train(args) 
    else:
        print("predicting")
        predict(args)