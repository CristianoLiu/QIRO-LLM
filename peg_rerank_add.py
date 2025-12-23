# -*- coding: utf-8 -*-
import argparse
import json
import os
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple
import torch
from torch.nn import functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils import FlagDRESModel
from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from beir.retrieval.evaluation import EvaluateRetrieval


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def save_embeddings(embeddings: Dict[str, torch.Tensor], file_path: str) -> None:
    """
    embeddings: dict[str, torch.Tensor(cpu)]
    """
    emb_data = {
        "ids": list(embeddings.keys()),
        "embeddings": [emb.numpy().tolist() for emb in embeddings.values()],
    }
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(emb_data, f, ensure_ascii=False)


def load_embeddings(file_path: str) -> Dict[str, torch.Tensor]:
    with open(file_path, "r", encoding="utf-8") as f:
        emb_data = json.load(f)
    return {str(cid): torch.tensor(emb) for cid, emb in zip(emb_data["ids"], emb_data["embeddings"])}


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

query_instruction_for_retrieval_dict = {
    "peg": "为这个句子生成表示以用于检索相关文章：",
}

def rerank_with_pids(
    query: str,
    pid_passages: List[Tuple[str, str]],  
    reranker_tokenizer,
    reranker_model,
    device: torch.device,
    batch_size: int = 16,
):
    pids = [pp[0] for pp in pid_passages]
    passages = [pp[1] for pp in pid_passages]

    scores: List[float] = []
    for i in range(0, len(passages), batch_size):
        batch_passages = passages[i: i + batch_size]
        pairs = [[query, passage] for passage in batch_passages]

        with torch.no_grad():
            inputs = reranker_tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            ).to(device)
            outputs = reranker_model(**inputs)
            batch_scores = outputs.logits.view(-1).float().cpu().tolist()

        scores.extend(batch_scores)

    ranked = sorted(zip(pids, passages, scores), key=lambda x: x[2], reverse=True)
    return ranked  

class InstructionalEncoder:
    def __init__(self, model_name_or_path: str, pooling_method: str = "cls"):
        self.model = FlagDRESModel(
            model_name_or_path=model_name_or_path,
            query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
            pooling_method=pooling_method,
        )

    def set_query_instruction(self, instruction: Optional[str]) -> None:
        self.model.query_instruction_for_retrieval = instruction

    def set_doc_instruction(self, instruction: Optional[str]) -> None:
        if hasattr(self.model, "doc_instruction_for_retrieval"):
            self.model.doc_instruction_for_retrieval = instruction

    def encode_query(self, text: str) -> torch.Tensor:
        vecs = self.model.encode_queries([text])
        if isinstance(vecs, torch.Tensor):
            return vecs[0].detach().cpu()
        return torch.tensor(vecs[0])

    def encode_corpus(self, corpus: List[Any], batch_size: int = 32) -> List[torch.Tensor]:
        if len(corpus) == 0:
            return []

        first = corpus[0]
        if isinstance(first, str):
            corpus_items = [{"text": t} for t in corpus]
        elif isinstance(first, dict):
            corpus_items = corpus
        else:
            raise TypeError(f"Unsupported corpus item type: {type(first)}")

        vecs = self.model.encode_corpus(corpus_items, batch_size=batch_size)

        out: List[torch.Tensor] = []
        if isinstance(vecs, torch.Tensor):
            vecs = vecs.detach().cpu()
            for i in range(vecs.shape[0]):
                out.append(vecs[i])
        else:
            for v in vecs:
                out.append(torch.tensor(v))
        return out

class DuBaikeRetrievalMTEB(AbsTaskRetrieval):
    @property
    def description(self):
        return {
            "name": "DuBaikeRetrieval",
            "description": "Custom retrieval dataset for DuBaike",
            "reference": "",
            "type": "Retrieval",
            "category": "s2p",
            "eval_splits": ["test"],
            "eval_langs": ["zh"],
            "main_score": "ndcg_at_10",
        }

    def load_data(self, **kwargs):
        queries = load_jsonl("./data/DuBaikeRetrieval/merged_qwen.jsonl")
        corpus = load_jsonl("./data/DuBaikeRetrieval/corpus.jsonl")
        test_qrels = load_jsonl("./data/DuBaikeRetrieval/qrels/test.jsonl")

        split = "test"

        self.corpus = {split: {}}
        for doc in corpus:
            pid = str(doc["id"])
            self.corpus[split][pid] = {"title": "", "text": doc["text"]}

        self.queries = {split: {}}
        for q in queries:
            qid = str(q["id"])
            self.queries[split][qid] = q["text"]

        self.relevant_docs = {split: defaultdict(dict)}
        for item in test_qrels:
            qid = str(item["q_id"])
            pid = str(item["p_id"])
            rel = int(item["score"])
            self.relevant_docs[split][qid][pid] = rel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder_path", type=str, default="./model/PEG")
    parser.add_argument("--instruction_key", type=str, default="peg")
    parser.add_argument("--pooling_method", type=str, default="cls", choices=["cls", "mean"])
    parser.add_argument("--embedding_cache_path", type=str, default="./data/DuBaikeRetrieval/corpus_embeddings_peg.json")
    parser.add_argument("--results_path", type=str, default="./data/DuBaikeRetrieval/results_with_rerank1217.jsonl")
    parser.add_argument("--text_weight", type=float, default=1)
    parser.add_argument("--op_text_weight", type=float, default=0)
    parser.add_argument("--top_k_for_rerank", type=int, default=100)
    parser.add_argument("--final_top_k", type=int, default=10)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    task = DuBaikeRetrievalMTEB()
    task.load_data()
    split = "test"

    corpus_dict = {pid: v["text"] for pid, v in task.corpus[split].items()}
    qrels = {
        str(qid): {str(pid): int(rel) for pid, rel in rels.items()}
        for qid, rels in task.relevant_docs[split].items()
    }

    encoder = InstructionalEncoder(args.encoder_path, pooling_method=args.pooling_method)
    instruction = query_instruction_for_retrieval_dict.get(args.instruction_key, None)
    encoder.set_query_instruction(instruction)
    encoder.set_doc_instruction(None)
    print("current instruction_key:", args.instruction_key)
    print("current query_instruction_for_retrieval:", instruction)

    reranker_tokenizer = AutoTokenizer.from_pretrained("./model/bge-reranker-v2-m3")
    reranker_model = AutoModelForSequenceClassification.from_pretrained("./model/bge-reranker-v2-m3").to(device)
    reranker_model.eval()

    corpus_ids = list(task.corpus[split].keys())
    corpus_items = [{"id": str(pid), "text": task.corpus[split][pid]["text"]} for pid in corpus_ids]

    need_recompute = True
    if os.path.exists(args.embedding_cache_path):
        try:
            print("Loading precomputed corpus embeddings from cache...")
            corpus_embeddings = load_embeddings(args.embedding_cache_path)

            missing = [str(pid) for pid in corpus_ids if str(pid) not in corpus_embeddings]
            if missing:
                print(f"Cache missing {len(missing)} ids, will recompute embeddings...")
            else:
                need_recompute = False

        except Exception as e:
            print(f"Failed to load cache ({args.embedding_cache_path}): {e}")
            print("Will recompute embeddings...")

    if need_recompute:
        print("No usable cache found, computing corpus embeddings ...")
        corpus_emb_list = encoder.encode_corpus(corpus_items, batch_size=32)

        corpus_embeddings = {item["id"]: emb for item, emb in zip(corpus_items, corpus_emb_list)}

        os.makedirs(os.path.dirname(args.embedding_cache_path), exist_ok=True)
        print("Saving corpus embeddings to:", args.embedding_cache_path)
        save_embeddings(corpus_embeddings, args.embedding_cache_path)

    run_before: Dict[str, Dict[str, float]] = {}
    run_after: Dict[str, Dict[str, float]] = {}
    all_results = []

    queries_raw = load_jsonl("./data/DuBaikeRetrieval/merged_qwen.jsonl")

    for q in tqdm(queries_raw, desc="Retrieval + Rerank"):
        q_id = str(q["id"])
        q_text = q["text"]
        q_op_text = q["op_text"]
        full_query = q_text + " " + q_op_text

        q_text_emb = encoder.encode_query(q_text)
        q_op_text_emb = encoder.encode_query(q_op_text)

        scores = []
        for pid, emb in corpus_embeddings.items():
            pid = str(pid)
            sim_text = cosine_sim(q_text_emb, emb)
            sim_op_text = cosine_sim(q_op_text_emb, emb)
            weighted_sim = args.text_weight * sim_text + args.op_text_weight * sim_op_text
            scores.append((pid, float(weighted_sim)))

        scores.sort(key=lambda x: x[1], reverse=True)
        top_candidates = scores[: args.top_k_for_rerank]

        run_before[q_id] = {pid: score for pid, score in top_candidates[: args.final_top_k]}

        candidate_pids = [pid for pid, _ in top_candidates]
        pid_passages = [(pid, corpus_dict[pid]) for pid in candidate_pids]

        reranked = rerank_with_pids(
            full_query,
            pid_passages,
            reranker_tokenizer=reranker_tokenizer,
            reranker_model=reranker_model,
            device=device,
            batch_size=16,
        )

        final_reranked = reranked[: args.final_top_k]
        run_after[q_id] = {pid: float(score) for pid, _, score in final_reranked}

        results = [
            {"p_id": pid, "text": passage, "rerank_score": float(score)}
            for pid, passage, score in final_reranked
        ]
        all_results.append({"q_id": q_id, "results": results})

    k_values = [args.final_top_k]

    ndcg_b, map_b, recall_b, precision_b = EvaluateRetrieval.evaluate(qrels, run_before, k_values)
    ndcg_a, map_a, recall_a, precision_a = EvaluateRetrieval.evaluate(qrels, run_after, k_values)

    k = args.final_top_k
    print("\n===== BEIR Metrics (same evaluator as MTEB Retrieval) =====")
    print("text权重:", args.text_weight, "op_text权重:", args.op_text_weight)
    print(
        f"[Before] nDCG@{k}: {ndcg_b[f'NDCG@{k}']:.4f}  MAP@{k}: {map_b[f'MAP@{k}']:.4f}  "
        f"Recall@{k}: {recall_b[f'Recall@{k}']:.4f}  P@{k}: {precision_b[f'P@{k}']:.4f}"
    )
    print(
        f"[After ] nDCG@{k}: {ndcg_a[f'NDCG@{k}']:.4f}  MAP@{k}: {map_a[f'MAP@{k}']:.4f}  "
        f"Recall@{k}: {recall_a[f'Recall@{k}']:.4f}  P@{k}: {precision_a[f'P@{k}']:.4f}"
    )

    os.makedirs(os.path.dirname(args.results_path), exist_ok=True)
    with open(args.results_path, "w", encoding="utf-8") as f:
        for item in all_results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

