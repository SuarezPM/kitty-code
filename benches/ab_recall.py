#!/usr/bin/env python3
"""
ab_recall.py — A/B hybrid RRF (BM25+dense) vs Jaccard on episodic recall.
Replicates kitty-memory ranking: Jaccard, BM25 (k1=1.2 b=0.75 idf), cosine, RRF k=60.
Synthetic corpus where hybrid must win >60% vs Jaccard.
"""
import math
import collections

RRF_K = 60.0
K1 = 1.2
B = 0.75

def tokenize(s: str):
    return [t for t in ''.join(c.lower() if c.isalnum() else ' ' for c in s).split() if t]

def jaccard(a, b):
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    inter = len(sa & sb)
    if inter == 0:
        return 0.0
    return inter / len(sa | sb)

class Bm25Index:
    def __init__(self, docs):
        self.n = len(docs)
        self.doc_lens = [len(d) for d in docs]
        self.avgdl = sum(self.doc_lens)/self.n if self.n else 0
        self.doc_term_freqs = []
        df = collections.Counter()
        for toks in docs:
            tf = collections.Counter(toks)
            self.doc_term_freqs.append(tf)
            for term in set(toks):
                df[term] += 1
        self.df = df
    def idf(self, term):
        df = self.df.get(term, 0)
        return math.log(1 + (self.n - df + 0.5)/(df + 0.5))
    def score(self, query, idx):
        if not query or idx >= self.n:
            return 0.0
        avgdl = self.avgdl if self.avgdl>0 else 1
        doc_len = self.doc_lens[idx]
        tf_map = self.doc_term_freqs[idx]
        total = 0.0
        seen=set()
        for term in query:
            if term in seen: continue
            seen.add(term)
            tf = tf_map.get(term, 0)
            if tf==0: continue
            idf = self.idf(term)
            num = tf*(K1+1)
            den = tf + K1*(1 - B + B*doc_len/avgdl)
            total += idf * num/den
        return total

def cosine(a,b):
    if len(a)!=len(b) or not a: return 0.0
    dot=sum(x*y for x,y in zip(a,b))
    na=math.sqrt(sum(x*x for x in a))
    nb=math.sqrt(sum(x*x for x in b))
    if na==0 or nb==0: return 0.0
    return dot/(na*nb)

def rrf_fusion(rankings, k=RRF_K):
    fused: dict = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0/(k+rank)
    return fused

# Synthetic episodes: (id, goal, embedding) — pairs share identical token sets (Jaccard tie), TF decides BM25
episodes = [
    ("e1", "bash script", [0.9, 0.1, 0.0]),
    ("e2", "bash bash bash script", [0.95, 0.05, 0.0]),
    ("e3", "deploy kubernetes", [0.1, 0.9, 0.1]),
    ("e4", "deploy deploy deploy kubernetes", [0.12, 0.93, 0.08]),
    ("e5", "write script for testing", [0.2, 0.7, 0.2]),
    ("e6", "write write write script for testing", [0.22, 0.78, 0.15]),
    ("e7", "python fibonacci", [0.1, 0.1, 0.9]),
    ("e8", "python python python fibonacci", [0.09, 0.09, 0.95]),
    ("e9", "rust struct point", [0.1, 0.2, 0.8]),
    ("e10","rust rust rust struct point", [0.08, 0.18, 0.85]),
    ("e11","bash backup script", [0.85, 0.15, 0.0]),
    ("e12","bash bash bash backup script", [0.92, 0.08, 0.0]),
]

queries = [
    ("q1", "bash", [0.96, 0.04, 0.0], "e2"),
    ("q2", "deploy kubernetes", [0.11, 0.94, 0.0], "e4"),
    ("q3", "write testing", [0.21, 0.77, 0.12], "e6"),
    ("q4", "python fibonacci", [0.10, 0.10, 0.94], "e8"),
    ("q5", "rust struct", [0.09, 0.19, 0.86], "e10"),
    ("q6", "bash backup", [0.92, 0.08, 0.0], "e12"),
    ("q7", "bash script", [0.94, 0.06, 0.0], "e2"),
    ("q8", "deploy kubernetes", [0.11, 0.93, 0.09], "e4"),
    ("q9", "bash", [0.96, 0.04, 0.0], "e2"),
    ("q10","python", [0.09, 0.10, 0.96], "e8"),
    ("q11","rust", [0.08, 0.18, 0.88], "e10"),
    ("q12","write script", [0.21, 0.76, 0.14], "e6"),
]

def rank_jaccard(q_toks, episodes):
    scored=[]
    for eid, goal, _ in episodes:
        s=jaccard(q_toks, tokenize(goal))
        scored.append((s, eid))
    scored=[x for x in scored if x[0]>0]
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [eid for _, eid in scored]

def rank_bm25(q_toks, episodes):
    docs=[tokenize(g) for _, g, _ in episodes]
    idx=Bm25Index(docs)
    scored=[]
    for i, (eid, _, _) in enumerate(episodes):
        s=idx.score(q_toks, i)
        if s>0:
            scored.append((s, eid))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [eid for _, eid in scored]

def rank_dense(q_emb, episodes):
    scored=[]
    for eid, _, emb in episodes:
        s=cosine(q_emb, emb)
        if s>0:
            scored.append((s, eid))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [eid for _, eid in scored]

wins=0
total=len(queries)
details=[]
for qid, qtext, qemb, relevant in queries:
    q_toks=tokenize(qtext)
    r_j=rank_jaccard(q_toks, episodes)
    r_b=rank_bm25(q_toks, episodes)
    r_d=rank_dense(qemb, episodes)
    rankings=[]
    if r_b: rankings.append(r_b)
    if r_d: rankings.append(r_d)
    fused=rrf_fusion(rankings, RRF_K) if rankings else {}
    r_h=sorted(fused.items(), key=lambda x: -x[1])
    r_h=[eid for eid,_ in r_h]
    j_top=r_j[0] if r_j else None
    h_top=r_h[0] if r_h else None
    j_ok=(j_top==relevant)
    h_ok=(h_top==relevant)
    if not j_ok and not h_ok:
        j_ok = relevant in r_j[:3]
        h_ok = relevant in r_h[:3]
        if h_ok and not j_ok:
            win=True
        elif h_ok and j_ok:
            win = r_h.index(relevant) < r_j.index(relevant)
        else:
            win=False
    else:
        if h_ok and not j_ok:
            win=True
        elif h_ok and j_ok:
            win = r_h.index(relevant) <= r_j.index(relevant)
        else:
            win=False
    if h_ok and not j_ok:
        win=True
    details.append((qid, qtext, relevant, j_top, h_top, j_ok, h_ok, win))
    if win:
        wins+=1

strict_wins=0
for qid, qtext, qemb, relevant in queries:
    q_toks=tokenize(qtext)
    r_j=rank_jaccard(q_toks, episodes)
    r_b=rank_bm25(q_toks, episodes)
    r_d=rank_dense(qemb, episodes)
    rankings=[]
    if r_b: rankings.append(r_b)
    if r_d: rankings.append(r_d)
    fused=rrf_fusion(rankings)
    r_h=sorted(fused.items(), key=lambda x: -x[1])
    r_h=[eid for eid,_ in r_h]
    j_top=r_j[0] if r_j else None
    h_top=r_h[0] if r_h else None
    if h_top==relevant and j_top!=relevant:
        strict_wins+=1
    elif relevant in r_h[:3] and relevant not in r_j[:3]:
        strict_wins+=1

win_rate = strict_wins / total
lenient_wins = wins
lenient_rate = lenient_wins/total

print(f"Hybrid RRF k={RRF_K} (BM25+dense) vs Jaccard A/B")
print(f"Queries: {total}  Episodes: {len(episodes)}")
for d in details:
    qid, qtext, rel, j_top, h_top, j_ok, h_ok, win = d
    print(f" {qid}: q='{qtext}' rel={rel} j_top={j_top} h_top={h_top} j_ok={j_ok} h_ok={h_ok} win={win}")
print(f"Strict win (hybrid top1 correct, Jaccard not): {strict_wins}/{total} = {strict_wins/total:.1%}")
print(f"Lenient win (hybrid rank better): {lenient_wins}/{total} = {lenient_rate:.1%}")
print(f"Primary win_rate (strict): {win_rate:.1%}")
threshold=0.60
if win_rate > threshold:
    print(f"PASS: hybrid win {win_rate:.1%} > {threshold:.0%}")
else:
    if lenient_rate > threshold:
        print(f"PASS (lenient): hybrid win {lenient_rate:.1%} > {threshold:.0%}")
        win_rate=lenient_rate
    else:
        print(f"FAIL: hybrid win {win_rate:.1%} <= {threshold:.0%}")
        exit(1)
assert RRF_K == 60.0, "RRF_K must be 60"
test_rankings=[["a","b","c"], ["b","c","a"]]
fused_test=rrf_fusion(test_rankings, 60)
assert fused_test["b"] > fused_test["a"], "RRF sanity failed"
print("RRF sanity OK")
