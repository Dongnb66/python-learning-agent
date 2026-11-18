// missed_calls.test.mjs —— v1.2.1 新增：未接来电台账（拒答=未接来电，回拨=补证据重答）。
// 核心不变量：
// 1) 不传 extraDocs = 生产语料与参数，对拍口径逐字不变（IDF/avgdl 不被污染）；
// 2) extraDocs 严格校验：>5 条 / 缺字段 / 超长 / 非法 URL / 未知字段 / 重复 URL 一律 400；
// 3) 回拨可把代码级拒答翻转为回答：补充语料并入检索（IDF/avgdl 按并入后重算），
//    命中标 local:true，证据锚点指向新语料；白名单同步含 extraDocs 的 url；
// 4) 补了仍不过阈值 → 照样拒答、不调模型（约束不因「补证据」而松动）。
// 运行：node --test playground/tests/missed_calls.test.mjs
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  handleLearningAgent, answerQuery, buildResources, resolveExtraDocs,
  EXTRA_MAX_DOCS, EXTRA_CONTENT_MAX,
} from '../functions/handler.mjs';
import { SEED } from '../functions/seed.mjs';

const req = (tool, body, method = 'POST') =>
  new Request(`https://demo.example/functions/v1/app/${tool}`, {
    method,
    headers: { 'Content-Type': 'application/json' },
    body: method === 'POST' ? JSON.stringify(body ?? {}) : undefined,
  });
const post = (tool, body) => handleLearningAgent({ request: req(tool, body), env: () => undefined });

const Q_HIT = '学习 LangGraph';                 // 生产阈值下命中
const Q_LOW = '我想学点编程';                   // 生产阈值下拒答（最高分 ≈ 1.69）
const Q_MISS = '摩托车发动机化油器怎么清洗';    // 语料外话题，零命中拒答

// 回拨补充的本地演示语料：与 Q_LOW 共享大量 bigram，足以翻转为回答
const DOC_PROG = {
  title: '编程入门学习方法',
  url: 'https://example.com/learn-programming',
  content: '我想学点编程，应该从 Python 入门开始。学点编程的第一步是掌握变量与循环，编程入门需要每天练习。我想学点编程的同学可以先写小脚本，编程学习重在动手。',
};
// 与 Q_MISS 同主题：化油器清洗（语料外话题的回拨证据）
const DOC_CARB = {
  title: '摩托车化油器清洗指南',
  url: 'https://example.com/carburetor-clean',
  content: '摩托车发动机化油器怎么清洗：先拆下化油器，用化油器清洗剂喷洗主量孔与怠速量孔，再用压缩空气吹通。摩托车化油器清洗后装回发动机，检查怠速是否平稳。',
};
// 与 Q_LOW 完全无关：补了也翻不动（阈值照常执行）
const DOC_UNRELATED = {
  title: '摩托车保养杂记',
  url: 'https://example.com/moto-notes',
  content: '摩托车链条上油与胎压检查的例行保养记录，与编程学习无关。',
};

/* ================= 1. 默认对拍不变（不传 extraDocs = 生产口径） ================= */

test('不传 / 传 [] / 传 null 的 extraDocs：与默认逐值一致，无 extraDocsUsed 字段', () => {
  for (const q of [Q_HIT, Q_LOW]) {
    const base = answerQuery(q);
    assert.deepEqual(answerQuery(q, { extraDocs: [] }), base);
    assert.deepEqual(answerQuery(q, { extraDocs: null }), base);
    assert.equal('extraDocsUsed' in base, false);
    assert.equal(base.scores.length, 12, '全量分数表仍是生产 12 条（IDF/avgdl 未被污染）');
    assert.equal(base.hits.every((h) => !('local' in h)), true);
  }
  const rBase = buildResources({ steps: [Q_HIT] });
  assert.deepEqual(buildResources({ steps: [Q_HIT] }, { extraDocs: [] }), rBase);
  assert.deepEqual(buildResources({ steps: [Q_HIT] }, { extraDocs: null }), rBase);
  assert.equal('extraDocsUsed' in rBase, false);
});

test('路由层：不传 extraDocs 的 ask/resources 与本地默认逐值一致', async () => {
  const j = await (await post('ask', { query: Q_HIT })).json();
  assert.deepEqual(j, answerQuery(Q_HIT));
  const r = await (await post('resources', { steps: [Q_HIT] })).json();
  assert.deepEqual(r, buildResources({ steps: [Q_HIT] }));
});

/* ================= 2. 严格校验：越界 / 非法一律 400 ================= */

test('extraDocs 非法输入 → 400 且报错指明条目（ask 与 resources 同闸）', async () => {
  const long = 'x'.repeat(EXTRA_CONTENT_MAX + 1);
  const cases = [
    ['非数组', {}],
    ['超过 5 条', Array(EXTRA_MAX_DOCS + 1).fill(DOC_PROG)],
    ['条目非对象', ['字符串']],
    ['缺 title', [{ url: DOC_PROG.url, content: DOC_PROG.content }]],
    ['缺 url', [{ title: DOC_PROG.title, content: DOC_PROG.content }]],
    ['缺 content', [{ title: DOC_PROG.title, url: DOC_PROG.url }]],
    ['空 title', [{ ...DOC_PROG, title: '   ' }]],
    ['title 超 120', [{ ...DOC_PROG, title: 't'.repeat(121) }]],
    ['url 超 300', [{ ...DOC_PROG, url: 'https://example.com/' + 'u'.repeat(300) }]],
    ['content 超 2000', [{ ...DOC_PROG, content: long }]],
    ['非 http(s) URL', [{ ...DOC_PROG, url: 'ftp://example.com/x' }]],
    ['URL 含空白', [{ ...DOC_PROG, url: 'https://example.com/a b' }]],
    ['未知字段', [{ ...DOC_PROG, score: 9.9 }]],
    ['批内重复 URL', [DOC_PROG, { ...DOC_PROG, title: '另一个标题' }]],
    ['与生产语料 URL 撞车', [{ ...DOC_PROG, url: SEED.corpus[0].url }]],
    ['content 非字符串', [{ ...DOC_PROG, content: 123 }]],
  ];
  for (const [name, extraDocs] of cases) {
    for (const tool of ['ask', 'resources']) {
      const body = tool === 'ask' ? { query: Q_LOW, extraDocs } : { steps: [Q_LOW], extraDocs };
      const r = await post(tool, body);
      assert.equal(r.status, 400, `${tool} ${name} 应 400`);
      const j = await r.json();
      assert.equal(j.ok, false);
      assert.ok(j.error.includes('extraDocs'), `${tool} ${name} 报错应指明 extraDocs：${j.error}`);
    }
  }
});

test('边界值恰好合法：5 条 / content 恰 2000 字 / title 恰 120 字放行', () => {
  const five = Array.from({ length: EXTRA_MAX_DOCS }, (_, i) => ({
    title: `补充语料${i}`, url: `https://example.com/extra-${i}`, content: '编程入门练习。',
  }));
  assert.equal(resolveExtraDocs(five).length, 5);
  const edge = [{ title: 't'.repeat(120), url: 'https://example.com/edge', content: 'x'.repeat(EXTRA_CONTENT_MAX) }];
  assert.equal(resolveExtraDocs(edge)[0].content.length, EXTRA_CONTENT_MAX);
  assert.equal(resolveExtraDocs(undefined).length, 0);
});

/* ================= 3. 拒答翻转为回答（回拨 = 补证据重答） ================= */

test('Q_LOW 默认拒答；补一条贴题语料后翻转为回答，命中标 local 且证据锚点指向新语料', () => {
  const before = answerQuery(Q_LOW);
  assert.equal(before.refused, true);
  assert.equal(before.modelCalled, false);
  const after = answerQuery(Q_LOW, { extraDocs: [DOC_PROG] });
  assert.equal(after.refused, false, '补证据后应翻转');
  assert.equal(after.extraDocsUsed, 1);
  assert.equal(after.scores.length, 13, '全量打分含补充语料（12+1）');
  const local = after.hits.filter((h) => h.local === true);
  assert.ok(local.length >= 1, '命中里应有本地补充语料');
  assert.equal(local[0].url, DOC_PROG.url);
  assert.ok(local[0].content.includes(DOC_PROG.content), '证据原文逐字来自补充语料');
  assert.ok(local[0].marks.length >= 1, '字符级高亮区间存在');
  assert.equal(local[0].excerpt, local[0].content.slice(local[0].offset.start, local[0].offset.end));
  assert.ok(after.answer.includes(DOC_PROG.url), '回答引用补充语料出处');
  assert.ok(after.answer.includes('[1]'), '回答带证据编号');
});

test('语料外话题 Q_MISS：零命中拒答 → 补化油器语料后接通', () => {
  assert.equal(answerQuery(Q_MISS).refused, true);
  const j = answerQuery(Q_MISS, { extraDocs: [DOC_CARB] });
  assert.equal(j.refused, false);
  assert.equal(j.hits[0].local, true);
  assert.equal(j.hits[0].title, DOC_CARB.title);
});

test('补了不贴题的语料：仍拒答、不调模型，reason 说明已并入但未过阈值', () => {
  const j = answerQuery(Q_LOW, { extraDocs: [DOC_UNRELATED] });
  assert.equal(j.refused, true);
  assert.equal(j.modelCalled, false);
  assert.equal(j.extraDocsUsed, 1);
  assert.ok(j.reason.includes('并入 1 条本地补充语料'));
  assert.equal(j.scores.length, 13);
});

test('IDF/avgdl 按并入后重算：生产条目分数随合并语料变化（口径注释所述策略）', () => {
  const base = answerQuery(Q_HIT);
  const merged = answerQuery(Q_HIT, { extraDocs: [DOC_PROG] });
  assert.equal(merged.scores.length, 13);
  const baseTop = base.scores.find((s) => s.title === base.hits[0].title);
  const mergedTop = merged.scores.find((s) => s.title === base.hits[0].title);
  assert.notEqual(baseTop.score, mergedTop.score, '合并后 IDF/avgdl 重算，生产条目分数应变化');
});

/* ================= 4. 白名单同步含 extraDocs 的 url ================= */

test('resources：补充语料 url 进入本轮白名单，注入该 url 的候选放行、编造 url 照拦', () => {
  const steps = ['摩托车发动机化油器怎么清洗'];
  assert.equal(buildResources({ steps }).refused, true, '不补语料时该步骤拒答');
  const r = buildResources({ steps }, { extraDocs: [DOC_CARB] });
  assert.equal(r.refused, false);
  assert.ok(r.whitelist.some((w) => w.url === DOC_CARB.url), '白名单应含补充语料 url');
  assert.equal(r.extraDocsUsed, 1);
  const local = r.resources.filter((x) => x.local === true);
  assert.ok(local.length >= 1);
  assert.ok(local[0].content.includes(DOC_CARB.content));
  assert.ok(local[0].excerpt.length > 0, '证据锚点窗口存在');
  const injected = buildResources({
    steps,
    candidateItems: [
      { title: '补充语料候选', url: DOC_CARB.url, type: 'doc', description: '来自回拨补充' },
      { title: '编造课程', url: 'https://made-up.example.com/fake', type: 'video', description: '白名单外' },
    ],
  }, { extraDocs: [DOC_CARB] });
  assert.equal(injected.blocked, 1);
  assert.ok(injected.resources.some((x) => x.url === DOC_CARB.url));
  assert.ok(injected.blockedItems[0].url.includes('made-up'));
});

test('坏模型 + extraDocs：补充语料 url 在白名单内可放行，编造链接照拦（ask 与 resources）', () => {
  const j = answerQuery(Q_LOW, { simulateBadModel: true, extraDocs: [DOC_PROG] });
  assert.equal(j.refused, false);
  assert.equal(j.mode, 'bad-model');
  assert.equal(j.modelCalled, true);
  assert.ok(j.blocked >= 2, '编造链接与无据断言照拦');
  assert.ok(j.whitelist.some((w) => w.url === DOC_PROG.url));
  assert.ok(!j.answer.includes('fake-masterclass.example.com'));
  const r = buildResources({ steps: ['编程入门学习方法'] }, { simulateBadModel: true, extraDocs: [DOC_PROG] });
  assert.equal(r.mode, 'bad-model');
  assert.ok(r.blocked >= 2);
  assert.ok(r.whitelist.some((w) => w.url === DOC_PROG.url));
});

/* ================= 5. 与 Tangle 实验参数正交组合 ================= */

test('extraDocs + params：高阈值下补了也拒答；低阈值放行；非法 params 仍 400', async () => {
  const high = answerQuery(Q_HIT, { extraDocs: [DOC_PROG], params: { threshold: 5.9 } });
  assert.equal(high.refused, true);
  assert.equal(high.modelCalled, false);
  assert.equal(high.extraDocsUsed, 1);
  const low = answerQuery(Q_LOW, { extraDocs: [DOC_PROG], params: { threshold: 1.0, k1: 2, b: 0.4, eps: 0.1, k: 6 } });
  assert.equal(low.refused, false);
  assert.ok(low.hits.some((h) => h.local === true));
  const r = await post('ask', { query: Q_LOW, extraDocs: [DOC_PROG], params: { k1: 99 } });
  assert.equal(r.status, 400);
});

test('路由 ask/resources extraDocs 透传：线上口径 = 本地函数', async () => {
  const j = await (await post('ask', { query: Q_LOW, extraDocs: [DOC_PROG] })).json();
  assert.deepEqual(j, answerQuery(Q_LOW, { extraDocs: [DOC_PROG] }));
  assert.equal(j.refused, false);
  const r = await (await post('resources', { steps: ['编程入门学习方法'], extraDocs: [DOC_PROG] })).json();
  assert.deepEqual(r, buildResources({ steps: ['编程入门学习方法'] }, { extraDocs: [DOC_PROG] }));
  assert.equal(r.refused, false);
});

test('health：声明 missed-call-callback 特性与 extraDocs 边界', async () => {
  const j = await (await handleLearningAgent({ request: req('health', null, 'GET'), env: () => undefined })).json();
  assert.ok(j.features.includes('missed-call-callback'));
  assert.ok(j.features.includes('reactive-params'), '原有特性保持');
  assert.deepEqual(j.extraDocsBounds, { maxDocs: 5, titleMax: 120, urlMax: 300, contentMax: 2000 });
});