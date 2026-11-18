// tangle_params.test.mjs —— v1.2 新增：Tangle 式反应式参数（ask/resources 可选 params 覆盖）。
// 核心不变量：不传 params = 生产参数（对拍口径不变）；越界/非法类型一律 400；
// 阈值拖动可翻转拒答边界（高阈值拒答 / 低阈值放行），k1/b/eps 真实改变全量分数。
// 运行：node --test playground/tests/tangle_params.test.mjs
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  handleLearningAgent, answerQuery, buildResources, resolveParams,
  PROD_PARAMS, PARAM_BOUNDS, tokenize,
} from '../functions/handler.mjs';
import { BM25Okapi } from '../functions/bm25.mjs';
import { SEED } from '../functions/seed.mjs';

const req = (tool, body, method = 'POST') =>
  new Request(`https://demo.example/functions/v1/app/${tool}`, {
    method,
    headers: { 'Content-Type': 'application/json' },
    body: method === 'POST' ? JSON.stringify(body ?? {}) : undefined,
  });
const post = (tool, body) => handleLearningAgent({ request: req(tool, body), env: () => undefined });

const Q_HIT = '学习 LangGraph';        // 生产阈值 2.0 下命中（最高分 ≈ 2.74）
const Q_LOW = '我想学点编程';          // 生产阈值下拒答（最高分 ≈ 1.69）

/* ================= 1. 默认 = 生产参数（现有对拍不变） ================= */

test('不传 params / 传空对象 / 显式生产值：三者与默认逐值一致', () => {
  const base = answerQuery(Q_HIT);
  assert.deepEqual(answerQuery(Q_HIT, { params: {} }), base);
  assert.deepEqual(answerQuery(Q_HIT, { params: { ...PROD_PARAMS } }), base);
  assert.equal(base.threshold, 2.0);
  assert.equal(base.k, 4);
  assert.deepEqual(base.params, { k1: 1.5, b: 0.75, eps: 0.25, threshold: 2.0, k: 4 });
  const rBase = buildResources({ steps: [Q_HIT] });
  assert.deepEqual(buildResources({ steps: [Q_HIT] }, { params: {} }), rBase);
  assert.deepEqual(buildResources({ steps: [Q_HIT] }, { params: { ...PROD_PARAMS } }), rBase);
});

test('路由层：不传 params 的 ask/resources 与本地默认逐值一致', async () => {
  const j = await (await post('ask', { query: Q_HIT })).json();
  assert.deepEqual(j, answerQuery(Q_HIT));
  const r = await (await post('resources', { steps: [Q_HIT] })).json();
  assert.deepEqual(r, buildResources({ steps: [Q_HIT] }));
});

/* ================= 2. 严格边界校验：越界 / 非法类型一律 400 ================= */

test('五个参数各自越下界/越上界 → 400 且报错指明字段', async () => {
  const cases = [
    ['k1', 0.49], ['k1', 3.01],
    ['b', -0.01], ['b', 1.01],
    ['eps', -0.01], ['eps', 1.01],
    ['threshold', -0.01], ['threshold', 6.01],
    ['k', 0], ['k', 13],
  ];
  for (const [key, v] of cases) {
    for (const tool of ['ask', 'resources']) {
      const body = tool === 'ask' ? { query: Q_HIT, params: { [key]: v } } : { steps: [Q_HIT], params: { [key]: v } };
      const r = await post(tool, body);
      assert.equal(r.status, 400, `${tool} params.${key}=${v} 应 400`);
      const j = await r.json();
      assert.equal(j.ok, false);
      assert.ok(j.error.includes(key), `报错应指明 ${key}：${j.error}`);
    }
  }
});

test('非法类型 / 非整数 k / 未知字段 / params 非对象 → 400', async () => {
  const badParams = [
    { k1: '1.5' }, { b: true }, { eps: null }, { threshold: '2' },
    { k: 2.5 }, { k: '4' }, { foo: 1 }, { k1: 1.5, extra: 0 },
  ];
  for (const params of badParams) {
    const r = await post('ask', { query: Q_HIT, params });
    assert.equal(r.status, 400, `params=${JSON.stringify(params)} 应 400`);
  }
  for (const params of [[1.5], 'k1=1.5', 42]) {
    const r = await post('ask', { query: Q_HIT, params });
    assert.equal(r.status, 400, `params=${JSON.stringify(params)} 应 400`);
  }
});

test('边界值本身合法：min 组与 max 组都 200，resolveParams 回显生效值', async () => {
  const minSet = { k1: 0.5, b: 0, eps: 0, threshold: 0, k: 1 };
  const maxSet = { k1: 3, b: 1, eps: 1, threshold: 6, k: 12 };
  for (const params of [minSet, maxSet]) {
    assert.deepEqual(resolveParams(params), params);
    const r = await post('ask', { query: Q_HIT, params });
    assert.equal(r.status, 200);
    const j = await r.json();
    assert.deepEqual(j.params, params);
    assert.equal(j.threshold, params.threshold);
    assert.equal(j.k, params.k);
  }
  assert.deepEqual(PARAM_BOUNDS.k, { min: 1, max: 12, int: true });
});

/* ================= 3. 阈值拖动 → 拒答边界翻转 ================= */

test('高阈值拒答翻转：生产阈值命中的查询，threshold=3.5/6 时被代码级拒答', () => {
  const base = answerQuery(Q_HIT);
  assert.equal(base.refused, false);
  for (const threshold of [3.5, 6]) {
    const out = answerQuery(Q_HIT, { params: { threshold } });
    assert.equal(out.refused, true, `threshold=${threshold}`);
    assert.equal(out.modelCalled, false);
    assert.deepEqual(out.hits, []);
    assert.ok(out.reason.includes(String(threshold)), '拒答理由应回显实验阈值');
    assert.ok(out.scores.length === 12, '全量分数表仍在（只是被阈值拦下）');
  }
});

test('低阈值放行：生产阈值拒答的查询，threshold=1.2/0 时放行真实命中', () => {
  assert.equal(answerQuery(Q_LOW).refused, true);
  for (const threshold of [1.2, 0]) {
    const out = answerQuery(Q_LOW, { params: { threshold } });
    assert.equal(out.refused, false, `threshold=${threshold}`);
    assert.ok(out.hits.length >= 1);
    assert.equal(out.hits[0].title, 'Python 入门：变量、循环与函数');
    for (const h of out.hits) assert.ok(h.score > threshold);
    assert.ok(out.answer.includes(out.hits[0].url), '放行答案仍只引用真实资料');
  }
});

test('top-k 覆盖：k=1 只留最相关一条，k=12 放行更多候选', () => {
  const one = answerQuery(Q_HIT, { params: { threshold: 0, k: 1 } });
  assert.equal(one.hits.length, 1);
  assert.equal(one.hits[0].title, 'LangGraph 多智能体编排入门');
  const broad = '学习 Python 编程 入门 基础 教程'; // 8 条候选分数 > 0，足以区分 k=4 与 k=12
  const four = answerQuery(broad, { params: { threshold: 0, k: 4 } });
  const twelve = answerQuery(broad, { params: { threshold: 0, k: 12 } });
  assert.equal(four.hits.length, 4);
  assert.ok(twelve.hits.length > four.hits.length, 'k=12 应比 k=4 放行更多');
  assert.ok(twelve.hits.length <= 12);
  assert.deepEqual(twelve.hits.slice(0, 4), four.hits, '前 4 条与 k=4 完全一致');
});

/* ================= 4. k1/b/eps 真实改变打分（与 BM25Okapi 逐值一致） ================= */

test('k1/b/eps 覆盖：全量分数与同参数 BM25Okapi 逐值一致，且不同于生产分数', () => {
  const texts = SEED.corpus.map((d) => `${d.title}\n${d.content}`);
  for (const params of [{ k1: 3, b: 0.2, eps: 0.5 }, { k1: 0.5, b: 1, eps: 0 }, { eps: 1 }]) {
    const out = answerQuery(Q_HIT, { params });
    const vec = new BM25Okapi(texts.map(tokenize), params.k1 ?? 1.5, params.b ?? 0.75, params.eps ?? 0.25);
    const expected = vec.getScores(tokenize(Q_HIT))
      .map((score, i) => ({ title: SEED.corpus[i].title, score }))
      .sort((a, b) => b.score - a.score);
    assert.equal(out.scores.length, expected.length);
    for (let i = 0; i < expected.length; i++) {
      assert.equal(out.scores[i].title, expected[i].title);
      assert.ok(Math.abs(out.scores[i].score - expected[i].score) < 1e-9, `params=${JSON.stringify(params)} doc#${i}`);
    }
  }
  const tuned = answerQuery(Q_HIT, { params: { k1: 3 } });
  const base = answerQuery(Q_HIT);
  assert.notDeepEqual(tuned.scores, base.scores, 'k1 改变必须真实反映到分数');
});

test('实验参数下证据锚点不变量仍成立（excerpt/offset/marks 自洽）', () => {
  const out = answerQuery(Q_HIT, { params: { k1: 2.5, b: 0.3, threshold: 1, k: 8 } });
  assert.ok(out.hits.length > 0);
  for (const h of out.hits) {
    assert.equal(h.excerpt, h.content.slice(h.offset.start, h.offset.end));
    assert.ok(Array.isArray(h.marks));
  }
});

/* ================= 5. resources 同口径 + 坏模型组合 ================= */

test('resources：高阈值拒答 / 低阈值放行，白名单随参数移动', () => {
  const base = buildResources({ steps: [Q_HIT] });
  assert.equal(base.refused, false);
  const high = buildResources({ steps: [Q_HIT] }, { params: { threshold: 6 } });
  assert.equal(high.refused, true);
  assert.equal(high.modelCalled, false);
  assert.deepEqual(high.resources, []);
  assert.equal(buildResources({ steps: [Q_LOW] }).refused, true);
  const low = buildResources({ steps: [Q_LOW] }, { params: { threshold: 1.2 } });
  assert.equal(low.refused, false);
  assert.ok(low.resources.length >= 1);
  const allowed = new Set(low.whitelist.map((w) => w.url));
  for (const r of low.resources) assert.ok(allowed.has(r.url));
});

test('坏模型 + 实验参数：高阈值照样拒答不调模型；低阈值放行后引用强制闸照常拦截', () => {
  const refused = answerQuery(Q_HIT, { simulateBadModel: true, params: { threshold: 3.5 } });
  assert.equal(refused.refused, true);
  assert.equal(refused.modelCalled, false);
  const passed = answerQuery(Q_HIT, { simulateBadModel: true, params: { threshold: 1.2 } });
  assert.equal(passed.mode, 'bad-model');
  assert.equal(passed.modelCalled, true);
  assert.ok(passed.blocked >= 2);
  assert.ok(!passed.answer.includes('fake-masterclass.example.com'));
  const res = buildResources({ steps: [Q_HIT] }, { simulateBadModel: true, params: { threshold: 6 } });
  assert.equal(res.refused, true);
  assert.equal(res.modelCalled, false);
});

/* ================= 6. 路由透传与 health ================= */

test('路由 ask/resources params 透传：线上口径 = 本地函数', async () => {
  const params = { k1: 2, b: 0.4, eps: 0.1, threshold: 1.2, k: 6 };
  const j = await (await post('ask', { query: Q_LOW, params })).json();
  assert.deepEqual(j, answerQuery(Q_LOW, { params }));
  assert.equal(j.refused, false);
  const r = await (await post('resources', { steps: [Q_LOW], params })).json();
  assert.deepEqual(r, buildResources({ steps: [Q_LOW] }, { params }));
  assert.equal(r.refused, false);
});

test('health：版本 1.2 + reactive-params 特性 + 边界声明', async () => {
  const j = await (await handleLearningAgent({ request: req('health', null, 'GET'), env: () => undefined })).json();
  assert.ok(j.version.includes('1.2'));
  assert.ok(j.features.includes('reactive-params'));
  assert.deepEqual(j.paramBounds, PARAM_BOUNDS);
  assert.equal(j.threshold, 2.0);
  assert.equal(j.k, 4);
});