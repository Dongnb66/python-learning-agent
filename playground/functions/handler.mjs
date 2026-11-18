// handler.mjs —— python-learning-agent 公网演示版（StudyHack）业务 handler。
// 与本地完整版同源的部分：BM25(bigram)+阈值 2.0 检索、空检索代码级拒答、URL 白名单、
// 画像姓名/专业正则、规则版规划/题库（app/mock_llm.py 零 Key 样例）、ReAct 辅导规则策略
// （app/agents/tutor_agent.py _run_policy_react，4 轮上限 + trace 审计）。
// 公网版为无状态演示：学情由浏览器 localStorage 随请求传入；完整 LangGraph 8 节点编排、
// SQLite 跨会话记忆与真实 LLM 调用在本地完整版。无 Key 一律走规则兜底，绝不编造。
// v1.1：证据锚点（hits/resources 附 excerpt/offset/marks 字符区间）+ simulateBadModel 坏模型对照
//（注入永远编造的演示模型，阈值/拒答/白名单/引用强制照常执行，blockedItems 逐条示众）。
// v1.2：Tangle 式反应式参数——ask/resources 可选 params 覆盖 {k1,b,eps,threshold,k}，
// 严格边界校验（越界 400）；不传 = 生产参数，与本地完整版对拍口径不变。
import { SEED, CORPUS_SOURCE, RESOURCE_RELEVANCE } from './seed.mjs';
import { buildRetriever, retrieve, allScores, tokenize } from './bm25.mjs';

const MAX_BODY = 100_000;
const MIN_SCORE = 2.0;   // app/rag.py retrieve() 默认阈值
const K = 4;             // app/rag.py retrieve() 默认 k
const MAX_ROUNDS = 4;    // app/agents/tutor_agent.py MAX_ROUNDS
const VERSION = 'learning-agent-playground-1.2.0';

const RETRIEVER = buildRetriever(SEED.corpus);

/* ---- Tangle 实验参数：生产默认值 + 严格边界（越界/非法类型一律 400） ---- */
const PROD_PARAMS = Object.freeze({ k1: 1.5, b: 0.75, eps: 0.25, threshold: MIN_SCORE, k: K });
const PARAM_BOUNDS = Object.freeze({
  k1: { min: 0.5, max: 3, int: false },
  b: { min: 0, max: 1, int: false },
  eps: { min: 0, max: 1, int: false },
  threshold: { min: 0, max: 6, int: false },
  k: { min: 1, max: 12, int: true },
});
export function resolveParams(input) {
  if (input === undefined) return { ...PROD_PARAMS };
  if (!input || typeof input !== 'object' || Array.isArray(input)) throw new Error('params 必须是对象');
  const out = { ...PROD_PARAMS };
  for (const [key, v] of Object.entries(input)) {
    const bound = PARAM_BOUNDS[key];
    if (!bound) throw new Error(`params 含未知字段 ${key}（允许：${Object.keys(PARAM_BOUNDS).join('/')}）`);
    if (typeof v !== 'number' || !Number.isFinite(v)) throw new Error(`params.${key} 必须是有限数字`);
    if (v < bound.min || v > bound.max) throw new Error(`params.${key} 越界（允许 ${bound.min}–${bound.max}）`);
    if (bound.int && !Number.isInteger(v)) throw new Error(`params.${key} 必须是整数`);
    out[key] = v;
  }
  return out;
}
/* 实验参数检索器缓存：同参数复用，避免每次拖动重建索引；生产参数直接用 RETRIEVER */
const retrieverCache = new Map();
function retrieverFor(p) {
  if (p.k1 === PROD_PARAMS.k1 && p.b === PROD_PARAMS.b && p.eps === PROD_PARAMS.eps) return RETRIEVER;
  const key = `${p.k1}|${p.b}|${p.eps}`;
  let r = retrieverCache.get(key);
  if (!r) {
    r = buildRetriever(SEED.corpus, { k1: p.k1, b: p.b, epsilon: p.eps });
    if (retrieverCache.size >= 64) retrieverCache.delete(retrieverCache.keys().next().value);
    retrieverCache.set(key, r);
  }
  return r;
}

const json = (status, body) => Response.json(body, { status, headers: { 'cache-control': 'no-store' } });
const fail = (msg) => json(400, { ok: false, error: msg });

function str(v, name, max = 300) {
  if (typeof v !== 'string' || !v.trim()) throw new Error(`${name} 必填`);
  const s = v.trim();
  if (s.length > max) throw new Error(`${name} 超长（上限 ${max}）`);
  return s;
}
function optStr(v, max = 300) {
  if (v == null) return '';
  if (typeof v !== 'string') throw new Error('字段必须是字符串');
  if (v.length > max) throw new Error(`字段超长（上限 ${max}）`);
  return v.trim();
}
async function body(req) {
  const t = await req.text();
  if (t.length > MAX_BODY) throw new Error('请求体过大');
  const v = t ? JSON.parse(t) : {};
  if (!v || typeof v !== 'object' || Array.isArray(v)) throw new Error('请求体必须是对象');
  return v;
}

/* ---- 与 app/tools.py _clip 同逻辑：压成单行并截断 ---- */
function clip(text, n) {
  const flat = String(text ?? '').split(/\s+/).filter(Boolean).join(' ');
  return flat.length <= n ? flat : flat.slice(0, n) + '…';
}

/* ---- 画像正则：与 app/agents/profile_agent.py 逐字同源 ---- */
const NAME_RE = /(?:我叫|我是|我的名字是|姓名是|大家好?，?我?叫)\s*([^\s,，。.!！?？]{1,10})/;
const NAME_BLACKLIST = new Set(['小明', '小红', '老师', '同学', '大家', '我们', '自己', '那个人']);
const MAJOR_RE = /(?:我是|我在|我读|我学|他是|她是|你是|就读|学习|专业是|我|他|她|你|是|在|的|于)*([^\s,，。.!！?？]{2,8})(?:专业|系|学院|方向)/;
const MAJOR_BAD_WORDS = ['没有', '任何', '什么', '哪', '啥', '不', '无', '没'];

export function extractName(text) {
  const m = NAME_RE.exec(text);
  if (m) {
    const cand = m[1].trim();
    if (cand && !NAME_BLACKLIST.has(cand)) return cand;
  }
  return null;
}
export function extractMajor(text) {
  const m = MAJOR_RE.exec(text);
  if (!m) return null;
  const cand = m[1].trim();
  if (!cand || MAJOR_BAD_WORDS.some((bad) => cand.includes(bad))) return null;
  return cand;
}
function summarize(p) {
  const parts = [];
  if (p.name) parts.push(p.name);
  if (p.major) parts.push(`${p.major}专业`);
  parts.push(`基础${p.knowledgeBase.level}`);
  const goal = p.learningGoal.details || p.learningGoal.level;
  if (goal) parts.push(`目标：${goal}`);
  return parts.length ? parts.join('，') + '。' : '画像已生成。';
}

/* ---- 画像（规则版：表单 + 同源正则覆盖） ---- */
export function buildProfile(b) {
  const intro = optStr(b.intro, 500);
  const base = ['beginner', 'intermediate', 'advanced'].includes(b.knowledgeBase) ? b.knowledgeBase : 'beginner';
  const goal = optStr(b.goal, 200);
  const style = optStr(b.style, 40) || '逻辑型';
  const weak = optStr(b.weakPoints, 200);
  const pref = optStr(b.preference, 40) || '图文';
  const hours = Math.max(0, Math.min(80, Number(b.hours) || 0));
  const baseDetails = { beginner: '零基础或刚入门，需要从语法与基本练习补起', intermediate: 'Python 基础一般，能写简单脚本', advanced: '基础扎实，能独立完成中小型项目' };
  const p = {
    name: extractName(intro),
    major: extractMajor(intro),
    knowledgeBase: { level: base, details: baseDetails[base] },
    learningGoal: { level: goal ? 'clear' : 'vague', details: goal || '方向待明确，建议先跑通一个最小项目' },
    cognitiveStyle: { level: style, details: `偏好${style}的学习方式` },
    weakPoints: { level: weak ? 'high' : 'low', details: weak || '暂未识别出明确薄弱点' },
    resourcePreference: { level: pref, details: `偏好${pref}类学习材料` },
    studyTime: { level: hours < 5 ? 'low' : hours <= 10 ? 'medium' : 'high', details: `每周约 ${hours} 小时` },
    summary: '',
  };
  p.summary = summarize(p);
  return p;
}

/* ---- 证据锚点：从语料原文截取命中 bigram 周边窗口，给出可高亮的字符区间 ----
 * marks 为 content 内的绝对字符区间（合并重叠后），excerpt = content.slice(offset.start, offset.end)。
 * UI 点击 [n] 时展示原文并按 marks 打 <mark>，实现「结论 → 原文字符级」溯源。 */
export function annotateEvidence(content, query, before = 60, after = 100) {
  const text = String(content ?? '');
  const low = text.toLowerCase();
  const tokens = [...new Set(tokenize(query))].filter(Boolean);
  const ranges = [];
  for (const t of tokens) {
    let idx = low.indexOf(t);
    while (idx !== -1 && ranges.length < 400) {
      ranges.push({ start: idx, end: idx + t.length });
      idx = low.indexOf(t, idx + t.length);
    }
  }
  ranges.sort((a, b) => a.start - b.start || a.end - b.end);
  const marks = [];
  for (const r of ranges) {
    const last = marks[marks.length - 1];
    if (last && r.start <= last.end) last.end = Math.max(last.end, r.end);
    else marks.push({ start: r.start, end: r.end });
  }
  if (!marks.length) {
    const end = Math.min(text.length, before + after);
    return { excerpt: text.slice(0, end), offset: { start: 0, end }, marks: [] };
  }
  // 取命中最密集的锚点，向两侧扩窗
  let best = marks[0], bestCount = 0;
  for (const m of marks) {
    const ws = Math.max(0, m.start - before), we = Math.min(text.length, m.end + after);
    const count = marks.filter((x) => x.start < we && x.end > ws).length;
    if (count > bestCount) { bestCount = count; best = m; }
  }
  const start = Math.max(0, best.start - before);
  const end = Math.min(text.length, best.end + after);
  return { excerpt: text.slice(start, end), offset: { start, end }, marks: marks.slice(0, 24) };
}
const withEvidence = (hits, query) => hits.map((h) => ({ ...h, ...annotateEvidence(h.content, query) }));

/* ---- 坏模型对照（simulateBadModel）：把「模型」换成永远编造的坏模型，
 * 阈值 / 空检索拒答 / 白名单 / 引用强制照常执行——约束不依赖模型自觉，机制可交互证明。 ---- */
const FAKE_ASK_URL = 'https://fake-masterclass.example.com/secret-course';
function badAskModel(query, hits) {
  const real = hits[0];
  return [
    `我强烈推荐这个独家内部速成课，三天包会：${FAKE_ASK_URL} （全网找不到第二份）。`,
    `另外据统计 99% 的学生看完就能精通，「${real.title}」已经被官方废弃，别学了。`,
    `有据可查的部分：${real.title}（${real.url}）——这条来自资料库真实命中。`,
  ].join('\n');
}
const URL_RE = /https?:\/\/[^\s（）()，。;；、]+/g;
/* 引用强制闸：bad-model 输出逐行审计——含白名单外 URL = 编造链接；
 * 未引用任何白名单来源 = 无据断言；两者都拦，只放行可溯源行。 */
export function filterBadAnswer(raw, hits) {
  const allowed = new Set(hits.map((h) => h.url));
  const kept = [];
  const blockedItems = [];
  for (const line of String(raw).split('\n')) {
    if (!line.trim()) continue;
    const urls = line.match(URL_RE) || [];
    const fake = urls.filter((u) => !allowed.has(u));
    if (fake.length) blockedItems.push({ text: line, urls: fake, reason: '编造链接：URL 不在本轮检索白名单内' });
    else if (!urls.length) blockedItems.push({ text: line, urls: [], reason: '无据断言：未引用任何白名单来源' });
    else kept.push(line);
  }
  return { kept, blockedItems };
}
function badResourceModel(context, allowed) {
  const real = Object.entries(allowed)[0];
  const items = [
    { title: '三天精通一切·内部绝密速成课', url: FAKE_ASK_URL, type: 'video', description: '编造的独家课程，白名单里根本没有它' },
    { title: '官方已废弃全部现有教程', url: 'https://another-made-up.example.com/deprecated', type: 'doc', description: '编造链接 + 无据断言' },
  ];
  if (real) items.push({ title: real[0], url: real[1], type: 'doc', description: '这条是资料库真实命中，应当放行' });
  return items;
}

/* ---- 检索问答：BM25 + 阈值过滤 + 空检索代码级拒答（防幻觉第①②道） ---- */
export function answerQuery(query, opts = {}) {
  const p = resolveParams(opts.params);
  const retriever = retrieverFor(p);
  const bad = opts.simulateBadModel === true;
  // 注入点：等价于 pytest 里 patch get_structured_model；bad 模式注入「永远编造」的坏模型
  const callModel = opts.callModel || (bad ? badAskModel : null);
  const scores = allScores(retriever, query);
  const hits = withEvidence(retrieve(retriever, query, p.k, p.threshold), query);
  const maxScore = scores.length ? scores[0].score : 0;
  if (!hits.length) {
    // 第②道守卫：无事实依据 → 直接拒答，根本不调用模型（机制，不是期望；坏模型同样不被调用）
    return {
      ok: true, query, threshold: p.threshold, k: p.k, params: p, refused: true, modelCalled: false, mode: 'refused',
      reason: `资料库中无相关内容：全部 ${scores.length} 条候选的 BM25 分均 ≤ 阈值 ${p.threshold}（最高 ${maxScore.toFixed(4)}）。代码级拒答，未调用模型。`,
      answer: '资料库里确实没有这个主题的内容，我拒绝编造答案。建议换用资料库覆盖的话题（Python / FastAPI / LangGraph / RAG / Docker / 算法 / 网络 / SQL / 机器学习），或在本地完整版中扩充知识库。',
      hits: [], scores, simulateBadModel: bad,
    };
  }
  // 无 Key 规则兜底：答案只由检索命中的真实正文拼成，逐条附 [n] 编号出处（证据锚点）
  const ruleAnswer = '我只在资料库命中的真实资料范围内回答：\n' +
    hits.map((h, i) => `[${i + 1}]【${h.title}】${h.content.split('\n').slice(1).join(' ')}（出处：${h.url}）`).join('\n');
  let answer;
  let modelCalled = false;
  let mode = 'rules';
  if (callModel) {
    answer = String(callModel(query, hits) ?? '');
    modelCalled = true;
    mode = bad ? 'bad-model' : 'llm';
  }
  if (!modelCalled || !answer.trim()) {
    mode = modelCalled ? 'rule-fallback' : 'rules';
    answer = ruleAnswer;
  }
  const base = { ok: true, query, threshold: p.threshold, k: p.k, params: p, refused: false, modelCalled, mode, reason: '', answer, hits, scores, simulateBadModel: bad };
  if (mode === 'bad-model') {
    // 第③道闸（引用强制）：坏模型照常返回编造内容，但输出逐行审计后才放行
    const rawAnswer = answer;
    const { kept, blockedItems } = filterBadAnswer(rawAnswer, hits);
    return {
      ...base,
      rawAnswer,
      answer: kept.length ? kept.join('\n') : ruleAnswer,
      blocked: blockedItems.length,
      blockedItems,
      whitelist: hits.map((h) => ({ title: h.title, url: h.url })),
      reason: `坏模型被真实调用（modelCalled=true），输出 ${rawAnswer.split('\n').filter((l) => l.trim()).length} 条断言；引用强制闸拦下 ${blockedItems.length} 条（编造链接 / 无据断言），仅放行可溯源条目。`,
    };
  }
  return base;
}

/* ---- 资源推荐：白名单兜底（防幻觉第③道），与 resource_agent.py 同判据 ---- */
export function enforceWhitelist(items, allowed) {
  const allowedUrls = new Set(Object.values(allowed));
  const kept = [];
  const blockedItems = [];
  for (const it of items) {
    if (it.url && allowedUrls.has(it.url)) kept.push(it);
    else if (it.title && Object.prototype.hasOwnProperty.call(allowed, it.title)) kept.push({ ...it, url: allowed[it.title] });
    else blockedItems.push({ title: it.title || '', url: it.url || '', description: it.description || '', reason: 'URL 与标题均不在本轮检索白名单内，无从溯源' });
  }
  return { kept, blocked: blockedItems.length, blockedItems };
}

export function buildResources(input, opts = {}) {
  const p = resolveParams(opts.params);
  const retriever = retrieverFor(p);
  const bad = opts.simulateBadModel === true;
  const callModel = opts.callModel || (bad ? badResourceModel : null);
  const steps = Array.isArray(input?.steps) ? input.steps.slice(0, 8) : [];
  // 第①步：按计划步骤逐一检索（默认 k=4、阈值 2.0；params 可实验覆盖），构建本轮白名单
  const seen = new Set();
  const allowed = {};
  const parts = [];
  for (const q of steps) {
    const title = optStr(q, 120);
    if (!title) continue;
    for (const doc of retrieve(retriever, title, p.k, p.threshold)) {
      if (doc.title && !seen.has(doc.title)) {
        seen.add(doc.title);
        allowed[doc.title] = doc.url;
        parts.push(`[${doc.type}] ${doc.title} (${doc.url})\n${doc.content}`);
      }
    }
  }
  const context = parts.join('\n\n');
  // 第②步：检索为空 → 代码级拒答，不调用模型（坏模型同样一次都不被调用）
  if (!context) {
    return { ok: true, refused: true, modelCalled: false, mode: 'refused', resources: [], blocked: 0, blockedItems: [], whitelist: [], simulateBadModel: bad, params: p, reason: '计划步骤在资料库中均无相关命中（BM25 阈值过滤后为空），代码级拒答，未调用模型。' };
  }
  // 第③步：候选条目（有模型走模型；无 Key 用规则候选）→ 白名单过滤
  let items;
  let modelCalled = false;
  let mode = 'rules';
  if (callModel) {
    items = callModel(context, allowed) || [];
    modelCalled = true;
    mode = bad ? 'bad-model' : 'llm';
  } else if (Array.isArray(input?.candidateItems)) {
    items = input.candidateItems.slice(0, 10); // 演示注入：模拟模型输出（含编造条目）以展示拦截
    mode = 'injected';
  } else {
    items = Object.entries(allowed).slice(0, 4).map(([title, url]) => {
      const doc = SEED.corpus.find((d) => d.title === title);
      return { title, url, type: doc?.type || 'doc', description: (doc?.content || '').slice(0, 60) + '…', relevance: RESOURCE_RELEVANCE };
    });
  }
  const { kept, blocked, blockedItems } = enforceWhitelist(items, allowed);
  // 证据锚点：放行条目附上语料原文窗口与字符区间（按计划步骤词命中高亮）
  const stepQuery = steps.map((s) => optStr(s, 120)).filter(Boolean).join(' ');
  const resources = kept.map((r) => {
    const doc = SEED.corpus.find((d) => d.url === r.url);
    const content = doc ? `${doc.title}\n${doc.content}` : '';
    return { ...r, content, ...annotateEvidence(content, stepQuery) };
  });
  return {
    ok: true, refused: false, modelCalled, mode, resources, blocked, blockedItems,
    whitelist: Object.entries(allowed).map(([title, url]) => ({ title, url })),
    simulateBadModel: bad, params: p,
    reason: blocked ? `白名单拦下 ${blocked} 条无法溯源的条目（URL 与标题均不在本轮检索候选内）。` : '',
  };
}

/* ---- 测验批改 ---- */
export function gradeQuiz(answers) {
  const qs = SEED.quiz.questions;
  const results = qs.map((q, i) => {
    const given = Array.isArray(answers) ? optStr(answers[i], 100) : '';
    return { q: q.q, given, correct: given === q.answer, answer: q.answer, explanation: q.explanation };
  });
  const score = results.filter((r) => r.correct).length;
  return { ok: true, score, total: qs.length, results };
}

/* ---- 复盘（规则版）：上次 vs 本次学情对比，记忆随请求传入（无状态） ---- */
export function buildReview(b) {
  const profile = b.profile && typeof b.profile === 'object' ? b.profile : null;
  const qr = b.quizResult && typeof b.quizResult === 'object' ? b.quizResult : null;
  const memory = b.memory && typeof b.memory === 'object' ? b.memory : null;
  const score = Number(qr?.score) || 0;
  const total = Number(qr?.total) || SEED.quiz.questions.length;
  const ratio = total ? score / total : 0;
  const mastery = ratio >= 0.99 ? '掌握扎实，本轮目标基本达成'
    : ratio >= 2 / 3 ? '掌握良好，核心概念已建立'
    : ratio >= 1 / 3 ? '基础入门，部分核心概念仍需巩固'
    : '起步阶段，建议先补齐基础再刷题';
  const strengths = [];
  if (profile?.learningGoal?.level === 'clear') strengths.push('目标清晰，方向聚焦');
  if (profile?.knowledgeBase?.level && profile.knowledgeBase.level !== 'beginner') strengths.push('已具备 Python 基础');
  if (profile?.studyTime?.level === 'high') strengths.push('学习意愿强、每周时间充足');
  if (ratio >= 0.99) strengths.push('自测全对，知识点掌握到位');
  strengths.push('完成了规划→检索→测验→复盘的完整学习闭环');
  const gaps = [];
  const weakText = profile?.weakPoints?.details || '';
  if (weakText && weakText !== '暂未识别出明确薄弱点') {
    for (const g of weakText.split(/[、，,;；]/).map((s) => s.trim()).filter(Boolean)) gaps.push(g);
  }
  for (const t of (Array.isArray(qr?.wrongTopics) ? qr.wrongTopics.slice(0, 3) : [])) {
    const topic = optStr(t, 80);
    if (topic) gaps.push(`测验暴露：${topic}`);
  }
  const uniqGaps = [...new Set(gaps)].slice(0, 3);
  if (!uniqGaps.length) uniqGaps.push('暂无明确薄弱项，可进阶项目实战');
  const suggestions = [];
  if (uniqGaps.some((g) => /算法|数据结构/.test(g))) suggestions.push('每天 1-2 道 LeetCode 打底');
  if (ratio < 2 / 3) suggestions.push('重做错题对应知识点，再测一轮');
  suggestions.push('先完整跑通本项目再扩展功能');
  suggestions.push('把项目部署上线，形成可点开的作品集');
  const finalSuggestions = suggestions.slice(0, 4);
  let comparison = '';
  if (memory) {
    const prevScore = memory.score == null ? '未记录' : `${memory.score}/${memory.total ?? '?'}`;
    const persist = (Array.isArray(memory.gaps) ? memory.gaps : []).filter((g) => uniqGaps.includes(g));
    comparison = `上次（${memory.created_at || '时间未记录'}）掌握度「${memory.mastery || '未记录'}」→ 本次「${mastery}」；测验得分 ${prevScore} → ${score}/${total}。` +
      (persist.length ? `仍未改善的薄弱项：${persist.join('、')}。` : '上次的薄弱项本轮未再出现，继续保持。');
  }
  return {
    ok: true,
    review: { mastery, strengths: strengths.slice(0, 3), gaps: uniqGaps, suggestions: finalSuggestions, comparison },
    snapshot: {
      created_at: new Date().toISOString(),
      goal: profile?.learningGoal?.details || '',
      mastery, strengths: strengths.slice(0, 3), gaps: uniqGaps, suggestions: finalSuggestions, score, total,
    },
    hadMemory: Boolean(memory),
  };
}

/* ---- ReAct 辅导：工具层与 app/tools.py 同文案；规则策略与 _run_policy_react 同构 ---- */
function toolDispatch(state) {
  const profile = state.profile || null;
  const memory = state.memory || null;
  const historyCount = Math.max(0, Math.min(10000, Number(state.historyCount) || 0));
  return {
    count_history_sessions: () => `该学生历史学习会话数：${historyCount}`,
    get_last_session: () => {
      if (!memory) return '（该学生为首次学习，没有历史学情，不要编造「上次」的数据）';
      return `时间：${memory.created_at || '未记录'}；掌握度：${memory.mastery || '未记录'}；` +
        `亮点：${(memory.strengths || []).join('、') || '无'}；薄弱项：${(memory.gaps || []).join('、') || '无'}；` +
        `当时建议：${(memory.suggestions || []).join('、') || '无'}`;
    },
    get_student_profile: () => profile ? clip(JSON.stringify(profile), 600) : '（该学生暂无画像记录，可能是首次使用）',
    search_learning_resources: (args) => {
      const docs = retrieve(RETRIEVER, String(args.query || ''), 3, MIN_SCORE);
      if (!docs.length) return '（资料库中未找到匹配资料——请勿编造，建议如实告知学生该主题暂无现成材料）';
      return docs.map((d) => `- [${d.type}] ${d.title} (${d.url})：${clip(d.content, 240)}`).join('\n');
    },
  };
}

function dedup(seq) { return [...new Set(seq)]; }

export function runPolicyReact(question, state, hints) {
  const disp = toolDispatch(state);
  const trace = [];
  const used = [];
  const observations = {};
  let rounds = 0;
  const act = (name, args) => {
    if (rounds >= MAX_ROUNDS) return ''; // 轮数上限：防止无限工具调用
    rounds += 1;
    const fn = disp[name];
    const obs = fn ? String(fn(args)) : `（未知工具 ${name}，可用工具：${Object.keys(disp).join('、')}）`;
    used.push(name);
    observations[name] = obs;
    trace.push({ round: rounds, tool: name, args: JSON.stringify(args), observation: obs.slice(0, 200) });
    return obs;
  };
  // 决策 1：新学员还是老学员？
  act('count_history_sessions', { user_id: 'demo-user' });
  const digits = (observations.count_history_sessions.match(/\d+/g) || []).join('');
  const isReturning = Boolean(digits) && Number(digits) > 0;
  // 决策 2：老学员才读上次学情
  if (isReturning) act('get_last_session', { user_id: 'demo-user' });
  // 决策 3：拿薄弱点去真实资料库检索
  let weak = (Array.isArray(hints.gaps) ? hints.gaps : []).slice(0, 2);
  if (!weak.length) weak = [hints.weak_point || '学习方法'];
  const query = weak.map((w) => String(w)).filter(Boolean).join(' ').trim() || '学习方法';
  act('search_learning_resources', { query });
  // 决策 4：补读长期画像
  act('get_student_profile', { user_id: 'demo-user' });

  const gapsText = weak.map((g) => String(g)).join('、') || '暂未识别出明确薄弱点';
  const resourceText = observations.search_learning_resources || '';
  const hasMaterial = !resourceText.includes('未找到匹配资料');
  const memoryText = observations.get_last_session || '';
  const parts = [`当前主要卡点：${gapsText}。`];
  if (isReturning && memoryText) parts.push(`对照历史学情：${memoryText.slice(0, 120)}`);
  if (hasMaterial) {
    const firstLine = resourceText.split('\n').find((ln) => ln.trim().startsWith('-')) || '';
    parts.push(`建议先补：${firstLine.replace(/^-\s*/, '').trim().slice(0, 120)}`);
  } else {
    parts.push('资料库暂无该主题的现成材料，建议如实告知学生并给出通用的学习路径建议。');
  }
  return { question, answer: parts.join(' '), tools_used: dedup(used), rounds, mode: 'policy', trace, max_rounds: MAX_ROUNDS };
}

/* ---- 可选 LLM 路径：仅在平台配置了可用 Key 时启用（默认无 Key → 规则策略） ---- */
const PLACEHOLDER_MARKERS = ['your', 'xxx', 'placeholder', 'change', 'example', 'todo', 'none'];
function hasUsableKey(key) {
  const k = String(key || '').trim();
  if (!k) return false;
  const low = k.toLowerCase();
  return !PLACEHOLDER_MARKERS.some((m) => low.includes(m));
}
const TUTOR_TOOLS_SCHEMA = [
  { type: 'function', function: { name: 'count_history_sessions', description: '统计该学生历史学习会话次数（0 表示新学员）', parameters: { type: 'object', properties: {} } } },
  { type: 'function', function: { name: 'get_last_session', description: '读取最近一次学情快照', parameters: { type: 'object', properties: {} } } },
  { type: 'function', function: { name: 'get_student_profile', description: '读取长期画像', parameters: { type: 'object', properties: {} } } },
  { type: 'function', function: { name: 'search_learning_resources', description: '在真实资料库检索学习材料（BM25）', parameters: { type: 'object', properties: { query: { type: 'string' } }, required: ['query'] } } },
];
async function runLlmReact(question, state, hints, cfg) {
  const disp = toolDispatch(state);
  const messages = [
    { role: 'system', content: '你是「自主辅导智能体」。先调工具获取事实依据再下结论；检索不到的内容如实说明暂无材料，严禁编造资料名或链接。结论 120 字以内。' },
    { role: 'user', content: question },
  ];
  const trace = [];
  const used = [];
  let rounds = 0;
  for (let rnd = 1; rnd <= MAX_ROUNDS; rnd++) {
    const resp = await fetch(`${cfg.baseUrl}/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${cfg.apiKey}` },
      body: JSON.stringify({ model: cfg.model, messages, tools: TUTOR_TOOLS_SCHEMA, tool_choice: 'auto', temperature: 0.2 }),
    });
    if (!resp.ok) throw new Error('LLM HTTP ' + resp.status);
    const msg = (await resp.json()).choices?.[0]?.message;
    if (!msg?.tool_calls?.length) {
      return { question, answer: String(msg?.content || '').trim() || '（模型未给出结论）', tools_used: dedup(used), rounds, mode: 'llm', trace, max_rounds: MAX_ROUNDS };
    }
    rounds = rnd;
    messages.push(msg);
    for (const tc of msg.tool_calls) {
      const name = tc.function?.name || '';
      let args = {};
      try { args = JSON.parse(tc.function?.arguments || '{}'); } catch { /* 保持空参数 */ }
      const fn = disp[name];
      const obs = fn ? String(fn(args)) : `（未知工具 ${name}）`;
      used.push(name);
      trace.push({ round: rnd, tool: name, args: JSON.stringify(args), observation: obs.slice(0, 200) });
      messages.push({ role: 'tool', tool_call_id: tc.id, content: obs });
    }
  }
  messages.push({ role: 'user', content: '已达到工具调用轮数上限，请基于以上已获取的信息直接给出结论。' });
  const resp = await fetch(`${cfg.baseUrl}/chat/completions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${cfg.apiKey}` },
    body: JSON.stringify({ model: cfg.model, messages, temperature: 0.2 }),
  });
  if (!resp.ok) throw new Error('LLM HTTP ' + resp.status);
  const final = (await resp.json()).choices?.[0]?.message?.content;
  return { question, answer: String(final || '').trim() || '（已达轮数上限）', tools_used: dedup(used), rounds, mode: 'llm', trace, max_rounds: MAX_ROUNDS };
}

export async function runTutor(b, env) {
  const state = {
    profile: b.profile && typeof b.profile === 'object' ? b.profile : null,
    memory: b.memory && typeof b.memory === 'object' ? b.memory : null,
    historyCount: b.historyCount,
  };
  const hints = {
    gaps: Array.isArray(b.gaps) ? b.gaps.slice(0, 4).map((g) => optStr(g, 80)) : [],
    weak_point: optStr(b.weakPoint, 120),
  };
  const mastery = optStr(b.mastery, 120) || '未评估';
  const gapText = hints.gaps.length ? hints.gaps.join('、') : '未明确';
  const question = `该学生本轮掌握度：${mastery}；复盘识别出的薄弱项：${gapText}。\n请自主判断需要查询哪些信息，找出他真正的卡点，并给出下一步的具体行动建议。`;
  const key = env ? env('LLM_API_KEY') || '' : '';
  const cfg = { apiKey: hasUsableKey(key) ? key.trim() : '', baseUrl: (env && env('LLM_BASE_URL')) || 'https://api.deepseek.com/v1', model: (env && env('LLM_MODEL')) || 'deepseek-chat' };
  if (!cfg.apiKey) return { ok: true, ...runPolicyReact(question, state, hints) };
  try {
    const session = await runLlmReact(question, state, hints, cfg);
    if (!session.answer.trim()) return { ok: true, ...runPolicyReact(question, state, hints) };
    return { ok: true, ...session };
  } catch {
    return { ok: true, ...runPolicyReact(question, state, hints), degraded: true };
  }
}

/* ---- 路由 ---- */
export async function handleLearningAgent({ request, env }) {
  const tool = new URL(request.url).pathname.split('/').filter(Boolean).pop() ?? '';
  try {
    if (request.method === 'GET') {
      if (tool === 'health') return json(200, { ok: true, version: VERSION, mode: 'rules', corpusSize: SEED.corpus.length, threshold: MIN_SCORE, k: K, maxRounds: MAX_ROUNDS, features: ['evidence-anchors', 'bad-model-simulation', 'reactive-params'], paramBounds: PARAM_BOUNDS, time: new Date().toISOString() });
      if (tool === 'corpus') return json(200, { ok: true, source: CORPUS_SOURCE, docs: SEED.corpus });
      if (tool === 'quiz') return json(200, { ok: true, quiz: SEED.quiz });
      if (tool === 'plan') return json(200, { ok: true, plan: SEED.plan, source: '规则版规划（与本地零 Key 样例同源）' });
      if (['profile', 'ask', 'resources', 'grade', 'review', 'tutor'].includes(tool)) return json(405, { ok: false, error: `${tool} 仅支持 POST` });
      return json(404, { ok: false, error: '未知路由' });
    }
    if (request.method !== 'POST') return json(405, { ok: false, error: '仅支持 GET/POST' });
    const b = await body(request);
    if (tool === 'profile') return json(200, { ok: true, profile: buildProfile(b) });
    if (tool === 'plan') return json(200, { ok: true, plan: SEED.plan, source: '规则版规划（与本地零 Key 样例同源）' });
    if (tool === 'ask') return json(200, answerQuery(str(b.query, 'query', 200), { simulateBadModel: b.simulateBadModel === true, params: b.params }));
    if (tool === 'resources') return json(200, buildResources(b, { simulateBadModel: b.simulateBadModel === true, params: b.params }));
    if (tool === 'grade') return json(200, gradeQuiz(b.answers));
    if (tool === 'review') return json(200, buildReview(b));
    if (tool === 'tutor') return json(200, await runTutor(b, env));
    return json(404, { ok: false, error: '未知路由' });
  } catch (e) {
    return fail(e instanceof Error ? e.message : String(e));
  }
}

export { tokenize, MIN_SCORE, K, MAX_ROUNDS, RETRIEVER, PROD_PARAMS, PARAM_BOUNDS };