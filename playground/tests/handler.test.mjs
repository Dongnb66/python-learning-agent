// handler.test.mjs —— 公网版防幻觉行为断言（判据移植自 tests/test_resource_guard.py 等 pytest）
// + BM25 与真实后端逐值对拍（fixtures/bm25_expected.json 由 .venv 真实 app/rag.py 生成）。
// 运行：node --test playground/tests/
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  handleLearningAgent, answerQuery, buildResources, enforceWhitelist,
  buildProfile, extractName, extractMajor, buildReview, gradeQuiz, runPolicyReact,
  tokenize, MIN_SCORE, K, MAX_ROUNDS, RETRIEVER,
} from '../functions/handler.mjs';
import { retrieve, allScores } from '../functions/bm25.mjs';
import { SEED } from '../functions/seed.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIX = JSON.parse(fs.readFileSync(path.join(HERE, 'fixtures', 'bm25_expected.json'), 'utf-8'));

const REAL_TITLE = 'LangGraph 多智能体编排入门';
const REAL_URL = 'https://langchain-ai.github.io/langgraph/';
const FAKE_URL = 'https://definitely-not-in-corpus.example.com/awesome-course';

const req = (tool, body, method = 'POST') =>
  new Request(`https://demo.example/functions/v1/app/${tool}`, {
    method,
    headers: { 'Content-Type': 'application/json' },
    body: method === 'POST' ? JSON.stringify(body ?? {}) : undefined,
  });

/* ================= 1. 分词与 BM25 逐值对拍（同源同参数） ================= */

test('tokenize 与后端 app/rag.py 逐 token 一致', () => {
  for (const [text, expected] of Object.entries(FIX.tokenize)) {
    assert.deepEqual(tokenize(text), expected, `tokenize(${text})`);
  }
});

test('语料与后端 resources.json 标题序列一致（12 条全量）', () => {
  assert.deepEqual(SEED.corpus.map((d) => d.title), FIX.corpus_titles);
});

test('全部查询的全量 BM25 分数与后端逐值对拍（误差 < 1e-9）', () => {
  for (const [q, exp] of Object.entries(FIX.queries)) {
    const got = RETRIEVER.vectorizer.getScores(tokenize(q));
    assert.equal(got.length, exp.scores.length);
    for (let i = 0; i < got.length; i++) {
      assert.ok(Math.abs(got[i] - exp.scores[i]) < 1e-9,
        `query=${q} doc#${i} got=${got[i]} expected=${exp.scores[i]}`);
    }
  }
});

test('retrieve 命中序列（标题/URL/分数/排序）与后端一致', () => {
  for (const [q, exp] of Object.entries(FIX.queries)) {
    // 用后端分数重建期望命中：降序稳定排序 → top-k → score > 2.0
    const expected = exp.scores
      .map((score, i) => ({ score, i }))
      .sort((a, b) => b.score - a.score)
      .slice(0, K)
      .filter((r) => r.score > MIN_SCORE)
      .map((r) => ({ title: FIX.corpus_titles[r.i], score: r.score }));
    const got = retrieve(RETRIEVER, q, K, MIN_SCORE);
    assert.equal(got.length, expected.length, `query=${q} 命中数`);
    for (let i = 0; i < got.length; i++) {
      assert.equal(got[i].title, expected[i].title, `query=${q} 第${i + 1}条标题`);
      assert.ok(Math.abs(got[i].score - expected[i].score) < 1e-9, `query=${q} 第${i + 1}条分数`);
    }
    assert.deepEqual(got.map((h) => h.title), exp.hits.map((h) => h.title), `query=${q} 与后端 retrieve() 标题序列`);
    assert.deepEqual(got.map((h) => h.url), exp.hits.map((h) => h.url), `query=${q} URL 序列`);
  }
});

/* ============ 2. 防幻觉第①道：相关性阈值（移植 test_resource_guard.py） ============ */

test('语料外主题必须返回空，而不是拿低分项凑数', () => {
  assert.deepEqual(retrieve(RETRIEVER, '摩托车发动机化油器怎么清洗'), []);
  assert.deepEqual(retrieve(RETRIEVER, '我想学点编程'), []);
});

test('有相关命中时保留且最相关排最前', () => {
  const docs = retrieve(RETRIEVER, '学习 LangGraph');
  assert.ok(docs.length > 0);
  assert.equal(docs[0].title, REAL_TITLE);
});

test('阈值下结果不进入候选：所有放行项 score > 2.0，且被过滤项确实 ≤ 2.0', () => {
  for (const q of Object.keys(FIX.queries)) {
    const hits = retrieve(RETRIEVER, q, K, MIN_SCORE);
    for (const h of hits) assert.ok(h.score > MIN_SCORE, `${q}: ${h.title} score=${h.score} 不应放行`);
    const ranked = allScores(RETRIEVER, q);
    const passed = ranked.slice(0, K).filter((r) => r.score > MIN_SCORE);
    assert.equal(hits.length, passed.length);
    const dropped = ranked.slice(0, K).filter((r) => r.score <= MIN_SCORE);
    for (const d of dropped) assert.ok(!hits.some((h) => h.title === d.title), `${q}: ${d.title} 应被阈值拦下`);
  }
});

/* ============ 3. 防幻觉第②道：空检索 → 代码级拒答，不调用模型 ============ */

test('resources：空检索 → 拒答且模型一次都没被调用', () => {
  let calls = 0;
  const spy = () => { calls += 1; return []; };
  const out = buildResources({ steps: ['乳腺癌的靶向治疗方案有哪些'] }, { callModel: spy });
  assert.equal(calls, 0, '模型不得被调用');
  assert.equal(out.refused, true);
  assert.deepEqual(out.resources, []);
  assert.equal(out.modelCalled, false);
});

test('resources：没有计划就没有检索词 → 同样拒答', () => {
  let calls = 0;
  const out = buildResources({}, { callModel: () => { calls += 1; return []; } });
  assert.equal(calls, 0);
  assert.equal(out.refused, true);
  assert.deepEqual(out.resources, []);
});

test('ask：空检索 → 拒答、展示拒答理由、不调用模型', () => {
  let calls = 0;
  const out = answerQuery('摩托车发动机化油器怎么清洗', { callModel: () => { calls += 1; return 'x'; } });
  assert.equal(calls, 0);
  assert.equal(out.refused, true);
  assert.equal(out.modelCalled, false);
  assert.deepEqual(out.hits, []);
  assert.ok(out.reason.includes('阈值'));
  assert.ok(out.answer.includes('拒绝编造'));
});

test('ask：有命中时规则答案只引用真实资料（标题与 URL 均来自命中集）', () => {
  const out = answerQuery('学习 LangGraph');
  assert.equal(out.refused, false);
  assert.equal(out.modelCalled, false);
  for (const h of out.hits) {
    assert.ok(out.answer.includes(h.title));
    assert.ok(out.answer.includes(h.url));
    assert.ok(SEED.corpus.some((d) => d.url === h.url), '输出链接必须在语料白名单内');
  }
});

/* ============ 4. 防幻觉第③道：URL 白名单（移植 test_resource_guard.py） ============ */

const planSteps = { steps: ['学习 LangGraph'] };

test('白名单剔除编造链接，只留真实 URL', () => {
  const items = [
    { title: REAL_TITLE, url: REAL_URL, type: 'doc', description: '真实资料' },
    { title: '史上最强速成课', url: FAKE_URL, type: 'video', description: '幻觉产物' },
  ];
  const out = buildResources(planSteps, { callModel: () => items });
  assert.deepEqual(out.resources.map((r) => r.url), [REAL_URL]);
  assert.equal(out.blocked, 1);
});

test('只给标题的条目被归一化成资料库真实 URL', () => {
  const out = buildResources(planSteps, { callModel: () => [{ title: REAL_TITLE, url: null, type: 'doc', description: '有标题无链接' }] });
  assert.deepEqual(out.resources.map((r) => r.url), [REAL_URL]);
});

test('标题也不在资料库 → 无从溯源，直接丢弃', () => {
  const out = buildResources(planSteps, { callModel: () => [{ title: '某不存在的秘籍', url: null, type: 'doc', description: '无出处' }] });
  assert.deepEqual(out.resources, []);
  assert.equal(out.blocked, 1);
});

for (const badUrl of [FAKE_URL, 'http://[::1]:9999/x', 'javascript:alert(1)']) {
  test(`白名单是精确 URL 匹配：${badUrl} 必须被拦`, () => {
    const out = buildResources(planSteps, { callModel: () => [{ title: '伪装资料', url: badUrl, type: 'doc', description: '' }] });
    assert.deepEqual(out.resources, []);
  });
}

test('enforceWhitelist：输出链接全部在本轮白名单内（不变量）', () => {
  const out = buildResources({ steps: SEED.plan.steps.map((s) => s.title) });
  const allowed = new Set(out.whitelist.map((w) => w.url));
  for (const r of out.resources) assert.ok(allowed.has(r.url));
  assert.ok(out.resources.length > 0);
});

test('candidateItems 注入编造条目也被拦（演示路径与模型路径同一道闸）', () => {
  const out = buildResources({ steps: ['学习 LangGraph'], candidateItems: [{ title: '幻觉课程', url: FAKE_URL }] });
  assert.deepEqual(out.resources, []);
  assert.equal(out.blocked, 1);
});

/* ============ 5. 画像正则（与 profile_agent.py 同源） ============ */

test('姓名/专业正则提取与黑名单', () => {
  assert.equal(extractName('大家好，我叫李华，想学 AI Agent'), '李华');
  assert.equal(extractName('我是小明'), null);
  assert.equal(extractMajor('我是计算机科学专业的大二学生'), '计算机科学');
  assert.equal(extractMajor('我没有选任何专业'), null);
  const p = buildProfile({ intro: '我叫李华，我是软件工程专业', goal: '交付一个 Agent 项目', knowledgeBase: 'intermediate', weakPoints: '算法与数据结构', hours: 10 });
  assert.equal(p.name, '李华');
  assert.equal(p.major, '软件工程');
  assert.equal(p.learningGoal.level, 'clear');
  assert.equal(p.studyTime.level, 'medium');
  assert.ok(p.summary.includes('李华'));
});

/* ============ 6. 测验 / 复盘 / 记忆对比 ============ */

test('测验批改：全对 3/3，错答给出正确答案与解析', () => {
  const right = gradeQuiz(SEED.quiz.questions.map((q) => q.answer));
  assert.equal(right.score, 3);
  const wrong = gradeQuiz(['错', '错', '错']);
  assert.equal(wrong.score, 0);
  assert.equal(wrong.results[0].answer, 'StateGraph');
  assert.ok(wrong.results[0].explanation.length > 0);
});

test('复盘：首次学习 comparison 为空（不编造「上次」）', () => {
  const out = buildReview({ profile: buildProfile({ goal: 'x', weakPoints: '算法' }), quizResult: { score: 2, total: 3, wrongTopics: ['RAG'] }, memory: null });
  assert.equal(out.review.comparison, '');
  assert.equal(out.hadMemory, false);
  assert.ok(out.review.gaps.length >= 1);
});

test('复盘：有上次学情 → 上次 vs 本次对比（localStorage 记忆随请求传入）', () => {
  const prev = { created_at: '2026-11-01T00:00:00Z', mastery: '基础入门', gaps: ['算法与数据结构'], score: 1, total: 3 };
  const out = buildReview({ profile: buildProfile({ weakPoints: '算法与数据结构' }), quizResult: { score: 3, total: 3 }, memory: prev });
  assert.ok(out.review.comparison.includes('上次'));
  assert.ok(out.review.comparison.includes('1/3'));
  assert.ok(out.review.comparison.includes('3/3'));
  assert.ok(out.review.comparison.includes('算法与数据结构'));
  assert.equal(out.snapshot.score, 3);
});

/* ============ 7. ReAct 辅导（规则策略，4 轮上限 + trace 审计） ============ */

test('新学员：不做纵向对比，轮数 ≤ 4，trace 完整', () => {
  const s = runPolicyReact('q', { profile: null, memory: null, historyCount: 0 }, { gaps: ['算法与数据结构'], weak_point: '' });
  assert.equal(s.mode, 'policy');
  assert.ok(s.rounds <= MAX_ROUNDS);
  assert.equal(s.rounds, 3); // count → search → profile（新学员跳过 get_last_session）
  assert.ok(!s.tools_used.includes('get_last_session'));
  assert.deepEqual(s.trace.map((t) => t.round), [1, 2, 3]);
  for (const t of s.trace) assert.ok(t.observation.length <= 200);
  assert.ok(s.answer.includes('当前主要卡点'));
});

test('老学员：读上次学情并做对照，轮数 ≤ 4', () => {
  const memory = { created_at: '2026-11-01', mastery: '基础入门', gaps: ['算法'], strengths: [], suggestions: [] };
  const s = runPolicyReact('q', { profile: { name: '演示' }, memory, historyCount: 2 }, { gaps: ['算法'], weak_point: '' });
  assert.equal(s.rounds, 4);
  assert.ok(s.tools_used.includes('get_last_session'));
  assert.ok(s.answer.includes('对照历史学情'));
  assert.ok(s.rounds <= MAX_ROUNDS);
});

test('辅导检索不到资料 → 如实说暂无材料，不编造链接', () => {
  const s = runPolicyReact('q', { profile: null, memory: null, historyCount: 0 }, { gaps: ['摩托车发动机化油器'], weak_point: '' });
  assert.ok(s.answer.includes('资料库暂无该主题的现成材料'));
  assert.ok(!s.answer.includes('http'));
  const searchTrace = s.trace.find((t) => t.tool === 'search_learning_resources');
  assert.ok(searchTrace.observation.includes('未找到匹配资料'));
});

test('辅导检索到资料 → 建议引用真实语料标题', () => {
  const s = runPolicyReact('q', { profile: null, memory: null, historyCount: 0 }, { gaps: ['LangGraph 多智能体'], weak_point: '' });
  assert.ok(s.answer.includes('建议先补'));
  assert.ok(SEED.corpus.some((d) => s.answer.includes(d.title)));
});

/* ============ 8. HTTP 路由层 ============ */

test('GET health：参数与后端同源（阈值 2.0 / k=4 / 4 轮）', async () => {
  const r = await handleLearningAgent({ request: req('health', null, 'GET'), env: () => undefined });
  const j = await r.json();
  assert.equal(j.ok, true);
  assert.equal(j.threshold, 2.0);
  assert.equal(j.k, 4);
  assert.equal(j.maxRounds, 4);
  assert.equal(j.corpusSize, 12);
});

test('POST ask：线上路由与本地函数逐值一致', async () => {
  const r = await handleLearningAgent({ request: req('ask', { query: '学习 LangGraph' }), env: () => undefined });
  const j = await r.json();
  const local = answerQuery('学习 LangGraph');
  assert.deepEqual(j, local);
});

test('POST tutor：无 Key 走 policy，trace 轮数上限可见', async () => {
  const r = await handleLearningAgent({ request: req('tutor', { gaps: ['算法与数据结构'], historyCount: 0 }), env: () => undefined });
  const j = await r.json();
  assert.equal(j.mode, 'policy');
  assert.equal(j.max_rounds, 4);
  assert.ok(j.rounds <= 4);
});

test('占位符 Key 视为未配置（与 app/config.py 同规则）', async () => {
  const r = await handleLearningAgent({ request: req('tutor', { gaps: ['算法'], historyCount: 0 }), env: (n) => (n === 'LLM_API_KEY' ? 'sk-your-api-key-here' : undefined) });
  const j = await r.json();
  assert.equal(j.mode, 'policy');
});

test('未知路由 404 / GET 写端点 405 / 非法请求体 400', async () => {
  const r404 = await handleLearningAgent({ request: req('nope', null, 'GET'), env: () => undefined });
  assert.equal(r404.status, 404);
  const r405 = await handleLearningAgent({ request: req('ask', null, 'GET'), env: () => undefined });
  assert.equal(r405.status, 405);
  const r400 = await handleLearningAgent({ request: req('ask', {}), env: () => undefined });
  assert.equal(r400.status, 400);
});