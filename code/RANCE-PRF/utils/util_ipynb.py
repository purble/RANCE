import sys
sys.path += ['../']
import csv
from tqdm import tqdm 
import collections
import gzip
import pickle
import string
import numpy as np
import faiss
import os
import pytrec_eval
import json
import matplotlib.pyplot as plt
from utils.msmarco_eval import quality_checks_qids, compute_metrics, load_reference
from utils.util import (
    compute_BM25_score,
    BM25_helper,
)


def getPassageData(checkpoint_path, checkpoint):
    
    passage_embedding_data = []
    passage_embedding2id_data = []
    
    for i in range(8): ############### Change 4 according to number of GPUs
        try:
            with open(checkpoint_path + "passage_"+str(checkpoint)+"__emb_p__data_obj_"+str(i)+".pb", 'rb') as handle:
                passage_embedding_data.append(pickle.load(handle))
            with open(checkpoint_path + "passage_"+str(checkpoint)+"__embid_p__data_obj_"+str(i)+".pb", 'rb') as handle:
                passage_embedding2id_data.append(pickle.load(handle))
        except:
            break
    if (not passage_embedding_data) or (not passage_embedding2id_data):
        print("No passage data found for checkpoint: ",checkpoint)

    passage_embedding = np.concatenate(passage_embedding_data, axis=0)
    passage_embedding2id = np.concatenate(passage_embedding2id_data, axis=0)
        
    return passage_embedding, passage_embedding2id


def getQueryData(checkpoint_path, checkpoint, validFold):
    
    valid_query_embedding_data = []
    valid_query_embedding2id_data = []
    test_query_embedding_data = []
    test_query_embedding2id_data = []
    
    for i in range(8): ############### Change according to number of GPUs
        try:
            if os.path.exists(checkpoint_path + "dev_query_"+str(checkpoint)+"__emb_p__data_obj_"+str(i)+".pb"):
                with open(checkpoint_path + "dev_query_"+str(checkpoint)+"__emb_p__data_obj_"+str(i)+".pb", 'rb') as handle:
                    valid_query_embedding_data.append(pickle.load(handle))
                with open(checkpoint_path + "dev_query_"+str(checkpoint)+"__embid_p__data_obj_"+str(i)+".pb", 'rb') as handle:
                    valid_query_embedding2id_data.append(pickle.load(handle))
            if validFold == 1:
                if os.path.exists(checkpoint_path + "test_query_2_"+str(checkpoint)+"__emb_p__data_obj_"+str(i)+".pb"):
                    with open(checkpoint_path + "test_query_2_"+str(checkpoint)+"__emb_p__data_obj_"+str(i)+".pb", 'rb') as handle:
                        test_query_embedding_data.append(pickle.load(handle))
                    with open(checkpoint_path + "test_query_2_"+str(checkpoint)+"__embid_p__data_obj_"+str(i)+".pb", 'rb') as handle:
                        test_query_embedding2id_data.append(pickle.load(handle))
            else:
                if os.path.exists(checkpoint_path + "test_query_1_"+str(checkpoint)+"__emb_p__data_obj_"+str(i)+".pb"):
                    with open(checkpoint_path + "test_query_1_"+str(checkpoint)+"__emb_p__data_obj_"+str(i)+".pb", 'rb') as handle:
                        test_query_embedding_data.append(pickle.load(handle))
                    with open(checkpoint_path + "test_query_1_"+str(checkpoint)+"__embid_p__data_obj_"+str(i)+".pb", 'rb') as handle:
                        test_query_embedding2id_data.append(pickle.load(handle))
        except:
            break
    if (not valid_query_embedding_data) or (not valid_query_embedding2id_data) or (not test_query_embedding_data) or (not test_query_embedding2id_data):
        print("No query data found for checkpoint: ",checkpoint)
        
    valid_query_embedding = np.concatenate(valid_query_embedding_data, axis=0)
    valid_query_embedding2id = np.concatenate(valid_query_embedding2id_data, axis=0)
    test_query_embedding = np.concatenate(test_query_embedding_data, axis=0)
    test_query_embedding2id = np.concatenate(test_query_embedding2id_data, axis=0)
    
    
    return valid_query_embedding, valid_query_embedding2id, test_query_embedding, test_query_embedding2id


def isSame(x, y):
    return np.all(np.array(x).flatten() == np.array(y).flatten())


def update_query_embedding(passage_embedding, query_embedding_orig, I_nearest_neighbor, ndoc, lambda_prf):
    query_embedding = query_embedding_orig.copy()
    if ndoc == 0: return query_embedding
    for query_idx in range(len(I_nearest_neighbor)):
        top_ann_pid = I_nearest_neighbor[query_idx].copy()
        q_emb = query_embedding[query_idx]
        add_emb = None
        n = min(ndoc, len(top_ann_pid))
        for i in range(n):
            pid = top_ann_pid[i]
            if add_emb is None:
                add_emb = passage_embedding[pid].copy()
            else:
                add_emb += passage_embedding[pid].copy()
        add_emb /= (1.0*n)
        query_embedding[query_idx] = q_emb + lambda_prf*add_emb
    return query_embedding


def get_rerank_index(passage_embedding, query_embedding, query_embedding2id, pidmap, bm):
    rerank_data = {}
    dev_I_all = []
    for i,qid in enumerate(query_embedding2id):
        p_set = []
        p_set_map = {}
        if qid not in bm:
            print(qid,"not in bm25")
        else:
            count = 0
            for k,pid in enumerate(bm[qid]):
                if pid in pidmap:
                    for val in pidmap[pid]:
                        p_set.append(passage_embedding[val])
                        p_set_map[count] = val # new rele pos(key) to old rele pos(val)
                        count += 1
                else:
                    print(pid,"not in passages")
        dim = passage_embedding.shape[1]
        faiss.omp_set_num_threads(16)
        cpu_index = faiss.IndexFlatIP(dim)
        p_set =  np.asarray(p_set)
        cpu_index.add(p_set)    
        _, dev_I = cpu_index.search(query_embedding[i:i+1], len(p_set))
        for j in range(len(dev_I[0])):
            dev_I[0][j] = p_set_map[dev_I[0][j]]
        dev_I_all.append(dev_I[0])
    return dev_I_all


def get_best_query_embedding(passage_embedding, passage_embedding2id, query_embedding, query_embedding2id, \
                            query_positive_id, bm, rerank=True, cpu_index=None):
    
    global prf_ndocs
    global vals
    global topN
    
    best_lambda_prf = None
    best_ndocs = None
    max_ndcg = 0.0
    best_query_embedding = None

    pidmap = collections.defaultdict(list)
    for i in range(len(passage_embedding2id)):
        pidmap[passage_embedding2id[i]].append(i)
    
    if rerank:
        dev_I_orig = get_rerank_index(passage_embedding, query_embedding, query_embedding2id, pidmap, bm)
    else:
        if cpu_index is None:
            dim = passage_embedding.shape[1]
            faiss.omp_set_num_threads(16)
            cpu_index = faiss.IndexFlatIP(dim)
            cpu_index.add(passage_embedding)    
        _, dev_I_orig = cpu_index.search(query_embedding, topN)
    
    ndcgs_all = []
    params_all = []
    for ndoc in prf_ndocs:
        for lambda_prf in vals:
            
            query_embedding_new = update_query_embedding(passage_embedding, query_embedding, dev_I_orig, ndoc, lambda_prf)
            
            if rerank:
                all_dev_I = get_rerank_index(passage_embedding, query_embedding_new, query_embedding2id, pidmap, bm)
            else:
                _, all_dev_I = cpu_index.search(query_embedding_new, topN)
            
            result = EvalDevQuery(query_embedding2id, passage_embedding2id, query_positive_id, all_dev_I, topN)
            final_ndcg, ndcgs, mrrs, recalls, eval_query_cnt, final_Map, final_mrr, final_recall, hole_rate, ms_mrr, Ahole_rate, metrics, prediction = result

            print(final_ndcg, lambda_prf, ndoc)
            ndcgs_all.append(final_ndcg)
            params_all.append((ndoc, lambda_prf))
            if final_ndcg > max_ndcg:
                best_lambda_prf = lambda_prf
                best_ndocs = ndoc
                print("*******************")
                print(max_ndcg, final_ndcg, best_lambda_prf, best_ndocs)
                print("*******************")
                max_ndcg = final_ndcg
                best_query_embedding = query_embedding_new.copy()
                
    print("="*20)
    print(best_lambda_prf, best_ndocs)
    
    return best_lambda_prf, best_ndocs, best_query_embedding, cpu_index


def convert_to_string_id(result_dict):
    string_id_dict = {}

    # format [string, dict[string, val]]
    for k, v in result_dict.items():
        _temp_v = {}
        for inner_k, inner_v in v.items():
            _temp_v[str(inner_k)] = inner_v

        string_id_dict[str(k)] = _temp_v

    return string_id_dict


def rerankWithResidualLearningScore(I_nearest_neighbor, rerankTopN, query_embedding2id, passage_embedding2id, dev_query_embedding, passage_embedding, lambda_test, wt_emb, fold=1):
    
    
    residual_scores_all = {}
    for query_idx in range(len(I_nearest_neighbor)):
        query_id = query_embedding2id[query_idx]
        query_emb = dev_query_embedding[query_idx]
        
        top_ann_pid = I_nearest_neighbor[query_idx].copy()
        selected_ann_idx = top_ann_pid[:rerankTopN] # Need to select topN since for some of the chunks the embeddings might be garbage, but no garbage in topN coz..
        rank = 0
        
        residual_scores = {}
        scores = []
        for ix, emb_idx in enumerate(selected_ann_idx):
            passage_emb = passage_embedding[emb_idx]
            s_emb = np.dot(np.array(query_emb), np.array(passage_emb))
            s_lex = get_BM25_score(query_id, passage_embedding2id[emb_idx], fold)
            residual_scores[ix+1] = {"s_emb":s_emb, "s_lex":s_lex}
            score = (lambda_test*s_lex) + (s_emb*wt_emb)
            scores.append(score)
        
        scores = np.array(scores)
        if (query_idx == -1):
            print("scores shape: ", scores.shape)
            print(scores)
        sorted_idxs = scores.argsort()
        if (query_idx == -1):
            print(scores[sorted_idxs[::-1]])
        top_ann_pid[:rerankTopN] = top_ann_pid[:rerankTopN][sorted_idxs[::-1]]
        if (query_idx == -1):
            print("I_nearest_neighbor[query_idx][:10]")
            print(I_nearest_neighbor[query_idx][:10])
        I_nearest_neighbor[query_idx] = top_ann_pid
        if (query_idx == -1):
            print(I_nearest_neighbor[query_idx][:10])
        residual_scores_all[query_idx] = residual_scores
        
    return I_nearest_neighbor, residual_scores_all


def EvalDevQuery(query_embedding2id, passage_embedding2id, dev_query_positive_id, I_nearest_neighbor, topN, \
                 rerankTopN=-1, dev_query_embedding=None, passage_embedding=None, lambda_test=None, wt_emb=1.0, f=1):
    
    if (rerankTopN > 0):
        I_nearest_neighbor, residual_scores_all = rerankWithResidualLearningScore(I_nearest_neighbor, \
                                    rerankTopN, query_embedding2id, passage_embedding2id, dev_query_embedding, \
                                  passage_embedding, lambda_test, wt_emb, fold=f)
    
    prediction = {} #[qid][docid] = docscore, here we use -rank as score, so the higher the rank (1 > 2), the higher the score (-1 > -2)

    total = 0
    labeled = 0
    Atotal = 0
    Alabeled = 0
    qids_to_ranked_candidate_passages = {} 
    for query_idx in range(len(I_nearest_neighbor)): 
        seen_pid = set()
        query_id = query_embedding2id[query_idx]
        prediction[query_id] = {}

        top_ann_pid = I_nearest_neighbor[query_idx].copy()
        selected_ann_idx = top_ann_pid[:topN]
        rank = 0
        
        if query_id in qids_to_ranked_candidate_passages:
            pass    
        else:
            # By default, all PIDs in the list of 1000 are 0. Only override those that are given
            tmp = [0] * 1000
            qids_to_ranked_candidate_passages[query_id] = tmp
                
        for idx in selected_ann_idx:
            pred_pid = passage_embedding2id[idx]
            
            if not pred_pid in seen_pid:
                # this check handles multiple vector per document
                qids_to_ranked_candidate_passages[query_id][rank]=pred_pid
                Atotal += 1
                if pred_pid not in dev_query_positive_id[query_id]:
                    Alabeled += 1
                if rank < 10:
                    total += 1
                    if pred_pid not in dev_query_positive_id[query_id]:
                        labeled += 1
                rank += 1
                prediction[query_id][pred_pid] = -rank
                seen_pid.add(pred_pid)
    
    # use out of the box evaluation script
    evaluator = pytrec_eval.RelevanceEvaluator(
        convert_to_string_id(dev_query_positive_id), {'map_cut', 'ndcg_cut', 'recip_rank','recall'})

    eval_query_cnt = 0
    result = evaluator.evaluate(convert_to_string_id(prediction))
    
    qids_to_relevant_passageids = {}
    for qid in dev_query_positive_id:
        qid = int(qid)
        if qid in qids_to_relevant_passageids:
            pass
        else:
            qids_to_relevant_passageids[qid] = []
            for pid in dev_query_positive_id[qid]:
                if pid>0:
                    qids_to_relevant_passageids[qid].append(pid)
            
    ms_mrr = compute_metrics(qids_to_relevant_passageids, qids_to_ranked_candidate_passages)

    ndcg = 0
    Map = 0
    mrr = 0
    recall = 0
    recall_1000 = 0
    ndcgs = []
    mrrs = []
    recalls = []

    for k in result.keys():
        eval_query_cnt += 1
        ndcg += result[k]["ndcg_cut_10"]
        ndcgs.append(result[k]["ndcg_cut_10"])
        Map += result[k]["map_cut_10"]
        mrr += result[k]["recip_rank"]
        mrrs.append(result[k]["recip_rank"])
        recall += result[k]["recall_"+str(topN)]
        recalls.append(result[k]["recall_"+str(topN)])

    final_ndcg = ndcg / eval_query_cnt
    final_Map = Map / eval_query_cnt
    final_mrr = mrr / eval_query_cnt
    final_recall = recall / eval_query_cnt
    hole_rate = labeled/total
    Ahole_rate = Alabeled/Atotal

    return final_ndcg, ndcgs, mrrs, recalls, eval_query_cnt, final_Map, final_mrr, final_recall, hole_rate, ms_mrr, Ahole_rate, result, prediction


class args_obj:
    
    def __init__(self, freqmap_path, id2offset_path, raw_data_path):
        self.freqmap_path = freqmap_path
        self.id2offset_path = id2offset_path
        self.raw_data_path = raw_data_path


def convert_q_to_tokens(query):
    query = query.translate(query.maketrans('', '', string.punctuation))
    tokens = query.rstrip().lower().split()
    tokens_dic = {}
    for t in tokens:
        tokens_dic[t] = tokens_dic.get(t, 0) + 1
    return tokens_dic


def get_BM25_score(query_id, doc_id, fold):
    global bm25_helper

    ###### During testing on test data, change this to idx2q_v ######
    if fold == 1:
        query = bm25_helper.idx2q_t1[query_id]
    elif fold == 2:
        query = bm25_helper.idx2q_t2[query_id]
    #################################################################
    qtokens = convert_q_to_tokens(query)

    score = 0.0
    N = bm25_helper.N
    avgl = bm25_helper.avgl
    r = 0

    try:
        galago_doc_id = bm25_helper.idx2pid[doc_id]
        # print("doc_id: ", doc_id, "galago_doc_id: ", galago_doc_id)
    except (KeyError):
        f = 0

    # doc_id is internal index of ANCE
    dl = bm25_helper.docidx2len[galago_doc_id]

    num_tokens = 0
    for k, v in qtokens.items():
        try:
            # ANCE iternal idx to galago internal index
            f = bm25_helper.tok2tf[k][galago_doc_id]
        except (KeyError):
            f = 0
        try:
            n = bm25_helper.tok2df[k]
        except (KeyError):
            n = 0
        qf = v
        tmp = compute_BM25_score(n, f, qf, r, N, dl, avgl)
        score += tmp
        num_tokens += v

    score = score / num_tokens

    return score


id2offset_path = "/mnt/nfs/work1/zamani/prafullpraka/mst/ANCE/data/preprocessed_data/passage/roberta_base/firstp/"
raw_data_path = "/mnt/nfs/work1/zamani/prafullpraka/mst/ANCE/data/raw_data/"
freqmap_path = "/home/prafullpraka/work1/mst/ANCE/data/code_data/static/utilsPass/"
args = args_obj(freqmap_path, id2offset_path, raw_data_path)
global bm25_helper
bm25_helper = BM25_helper(args)


