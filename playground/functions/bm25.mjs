// bm25.mjs —— 从 app/rag.py + rank_bm25.BM25Okapi(0.2.2) 逐行移植的词法检索路。
// 同源同参数：CJK bigram 分词（app/rag.py tokenize）、BM25Okapi k1=1.5 b=0.75 epsilon=0.25、
// 相关性阈值 min_score=2.0（严格大于才放行）、top-k=4、空命中返回空列表（上层代码级拒答）。
// 语义路（Embedding）是本地完整版的可选增强，公网版与「未配 EMBED_API_KEY」的本地行为一致。

const CJK_START = 0x4e00, CJK_END = 0x9fff;

/** 与 app/rag.py tokenize() 同一套规则：ASCII 字母数字连续段 + CJK 连续段 bigram。 */
export function tokenize(text) {
  const tokens = [];
  let asc = [];
  const low = String(text).toLowerCase();
  const chars = [...low]; // 按码点迭代，与 Python 逐字符一致
  const flush = () => { if (asc.length) { tokens.push(asc.join('')); asc = []; } };
  let i = 0;
  const n = chars.length;
  while (i < n) {
    const ch = chars[i];
    if (/[a-z0-9]/.test(ch) && ch.codePointAt(0) < 128) { asc.push(ch); i += 1; continue; }
    flush();
    const cp = ch.codePointAt(0);
    if (cp >= CJK_START && cp <= CJK_END) {
      let j = i;
      while (j < n) {
        const c2 = chars[j].codePointAt(0);
        if (c2 < CJK_START || c2 > CJK_END) break;
        j += 1;
      }
      const run = chars.slice(i, j);
      if (run.length === 1) tokens.push(run[0]);
      else for (let k = 0; k < run.length - 1; k++) tokens.push(run[k] + run[k + 1]);
      i = j;
    } else {
      i += 1;
    }
  }
  flush();
  return tokens;
}

/** rank_bm25.BM25Okapi 的等价实现（含负 IDF 的 epsilon*average_idf 地板）。 */
export class BM25Okapi {
  constructor(corpusTokens, k1 = 1.5, b = 0.75, epsilon = 0.25) {
    this.k1 = k1; this.b = b; this.epsilon = epsilon;
    this.corpusSize = corpusTokens.length;
    this.docLen = [];
    this.docFreqs = [];
    const nd = new Map(); // 保持插入顺序，与 Python dict 一致（average_idf 求和顺序相同）
    let numDoc = 0;
    for (const doc of corpusTokens) {
      this.docLen.push(doc.length);
      numDoc += doc.length;
      const freqs = new Map();
      for (const w of doc) freqs.set(w, (freqs.get(w) || 0) + 1);
      this.docFreqs.push(freqs);
      for (const w of freqs.keys()) nd.set(w, (nd.get(w) || 0) + 1);
    }
    this.avgdl = numDoc / this.corpusSize;
    this.idf = new Map();
    let idfSum = 0;
    const negative = [];
    for (const [word, freq] of nd) {
      const idf = Math.log(this.corpusSize - freq + 0.5) - Math.log(freq + 0.5);
      this.idf.set(word, idf);
      idfSum += idf;
      if (idf < 0) negative.push(word);
    }
    this.averageIdf = idfSum / this.idf.size;
    const eps = this.epsilon * this.averageIdf;
    for (const word of negative) this.idf.set(word, eps);
  }

  /** 与 BM25Okapi.get_scores 逐项同式；query 重复 token 重复计分（与 Python 一致）。 */
  getScores(queryTokens) {
    const scores = new Array(this.corpusSize).fill(0);
    for (const q of queryTokens) {
      const idf = this.idf.get(q) || 0;
      for (let i = 0; i < this.corpusSize; i++) {
        const qFreq = this.docFreqs[i].get(q) || 0;
        scores[i] += idf * (qFreq * (this.k1 + 1) /
          (qFreq + this.k1 * (1 - this.b + this.b * this.docLen[i] / this.avgdl)));
      }
    }
    return scores;
  }
}

/**
 * 构建检索器。docs: [{title,url,type,content}]，正文 = `${title}\n${content}`
 * （与 app/rag.py _load_documents 把元数据拼进正文一致）。
 */
export function buildRetriever(docs) {
  const texts = docs.map((d) => `${d.title || ''}\n${d.content || ''}`);
  const vectorizer = new BM25Okapi(texts.map(tokenize));
  return { docs, texts, vectorizer };
}

/**
 * 与 app/rag.py _legacy_retrieve 同判据：
 * 全量打分 → 按分数降序（稳定排序，同分保持语料顺序，与 Python sorted 一致）
 * → 取前 k → 只保留 score > min_score 的项；没有则返回 []（上层据此代码级拒答）。
 */
export function retrieve(retriever, query, k = 4, minScore = 2.0) {
  const scores = retriever.vectorizer.getScores(tokenize(query));
  const ranked = scores
    .map((score, i) => ({ score, i }))
    .sort((a, b) => b.score - a.score);
  const hits = ranked.slice(0, k).filter((r) => r.score > minScore);
  return hits.map((r) => ({
    title: retriever.docs[r.i].title || '',
    url: retriever.docs[r.i].url || '',
    type: retriever.docs[r.i].type || 'article',
    content: retriever.texts[r.i],
    score: r.score,
  }));
}

/** 全量分数（供 UI 展示阈值过滤过程；顺序同 ranked）。 */
export function allScores(retriever, query) {
  const scores = retriever.vectorizer.getScores(tokenize(query));
  return scores
    .map((score, i) => ({ title: retriever.docs[i].title || '', score }))
    .sort((a, b) => b.score - a.score);
}