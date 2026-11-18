// evidence_badmodel.test.mjs —— v1.1 新增：证据锚点（excerpt/offset/marks）+ 坏模型对照模式。
// 核心不变量：把模型换成「永远编造」的坏模型，阈值 / 空检索拒答 / 白名单 / 引用强制依然成立。
// 运行：node --test playground/tests/evidence_badmodel.test.mjs
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  handleLearningAgent, answerQuery, buildResources, annotateEvidence, filterBadAnswer,
  tokenize, MIN_SCORE, RETRIEVER,
} from '../functions/handler.mjs';
import { retrieve } from '../functions/bm25.mjs';
import { SEED } from '../functions/seed.mjs';

const req = (tool, body, method = 'POST') =>
  new Request(`https://demo.example/functions/v1/app/${tool}`, {
    method,
    headers: { 'Content-Type': 'application/json' },
    body: method === 'POST' ? JSON.stringify(body ?? {}) : undefined,
  });

const FAKE_HOSTS = ['fake-masterclass.example.com', 'another-made-up.example.com'];

/* ================= 1. 证据锚点：excerpt / offset / marks ================= */

test('annotateEvidence：excerpt 与 offset 自洽，marks 区间落在原文内且确为 query token', () => {
  const doc = SEED.corpus.find((d) => d.title.includes('LangGraph'));
  const content = `${doc.title}\n${doc.content}`;
  const ev = annotateEvidence(content, '学习 LangGraph');
  assert.equal(ev.excerpt, content.slice(ev.offset.start, ev.offset.end));
  assert.ok(ev.offset.start >= 0 && ev.offset.end <= content.length);
  assert.ok(ev.marks.length > 0, '应有命中片段');
  const tokens = new Set(tokenize('学习 LangGraph'));
  for (const m of ev.marks) {
    assert.ok(m.start >= 0 && m.end > m.start && m.end <= content.length);
    const slice = content.slice(m.start, m.end).toLowerCase();
    assert.ok([...tokens].some((t) => slice.includes(t)),
      `mark 区间必须覆盖 query token：${slice}`);
  }
});

test('annotateEvidence：无命中词时给出头部窗口且 marks 为空（不伪造高亮）', () => {
  const ev = annotateEvidence('一段完全无关的正文内容', '量子色动力学');
  assert.deepEqual(ev.marks, []);
  assert.equal(ev.excerpt, '一段完全无关的正文内容'.slice(ev.offset.start, ev.offset.end));
});

test('ask：每条 hit 带 excerpt/offset/marks，规则答案带 [n] 编号引用', () => {
  const out = answerQuery('学习 LangGraph');
  assert.ok(out.hits.length > 0);
  out.hits.forEach((h, i) => {
    assert.equal(typeof h.excerpt, 'string');
    assert.ok(h.excerpt.length > 0);
    assert.equal(h.excerpt, h.content.slice(h.offset.start, h.offset.end));
    assert.ok(Array.isArray(h.marks) && h.marks.length > 0);
    assert.ok(out.answer.includes(`[${i + 1}]`), `答案应含编号 [${i + 1}]`);
  });
});

test('resources：放行条目带语料原文与证据区间（点击可溯源）', () => {
  const out = buildResources({ steps: ['学习 LangGraph'] });
  assert.ok(out.resources.length > 0);
  for (const r of out.resources) {
    assert.ok(r.content.length > 0);
    assert.equal(r.excerpt, r.content.slice(r.offset.start, r.offset.end));
    assert.ok(SEED.corpus.some((d) => d.url === r.url));
  }
});

/* ================= 2. 坏模型对照：ask ================= */

test('坏模型 ask（有命中）：modelCalled=true 但编造链接与无据断言被拦，放行行全部可溯源', () => {
  const out = answerQuery('学习 LangGraph', { simulateBadModel: true });
  assert.equal(out.refused, false);
  assert.equal(out.modelCalled, true, '坏模型必须被真实调用（证明约束不靠模型自觉）');
  assert.equal(out.mode, 'bad-model');
  assert.ok(out.rawAnswer.includes('fake-masterclass.example.com'), '原始输出确实含编造链接');
  assert.ok(out.blocked >= 2, '编造链接行 + 无据断言行都应被拦');
  assert.ok(out.blockedItems.some((b) => b.reason.includes('编造链接')));
  assert.ok(out.blockedItems.some((b) => b.reason.includes('无据断言')));
  for (const host of FAKE_HOSTS) assert.ok(!out.answer.includes(host), `过滤后答案不得含 ${host}`);
  assert.ok(!out.answer.includes('99%'), '无据断言不得进入放行答案');
  const allowed = new Set(out.hits.map((h) => h.url));
  for (const u of out.answer.match(/https?:\/\/[^\s（）()，。;；、]+/g) || []) {
    assert.ok(allowed.has(u), `放行答案中的 URL 必须在白名单内：${u}`);
  }
});

test('坏模型 ask（空检索）：阈值/拒答照常 → 仍然拒答，坏模型一次都没被调用', () => {
  for (const q of ['摩托车发动机化油器怎么清洗', '我想学点编程']) {
    const out = answerQuery(q, { simulateBadModel: true });
    assert.equal(out.refused, true, q);
    assert.equal(out.modelCalled, false, q);
    assert.deepEqual(out.hits, []);
    assert.ok(out.answer.includes('拒绝编造'));
    assert.equal(out.rawAnswer, undefined, '拒答路径不存在模型原始输出');
  }
});

test('坏模型 ask：放行答案的 [n]/出处仍指向真实命中（证据链不断）', () => {
  const out = answerQuery('RAG 防幻觉', { simulateBadModel: true });
  assert.equal(out.mode, 'bad-model');
  assert.ok(out.whitelist.length > 0);
  for (const w of out.whitelist) assert.ok(SEED.corpus.some((d) => d.url === w.url));
});

test('filterBadAnswer：只放行「全部 URL 均在白名单」的行（逐行审计不变量）', () => {
  const hits = [{ url: 'https://real.example.com/a' }];
  const raw = [
    '真实引用：https://real.example.com/a 的内容。',
    '编造：https://evil.example.com/x 独家速成。',
    '混合真假：https://real.example.com/a 与 https://evil.example.com/y。',
    '没有任何出处的断言。',
  ].join('\n');
  const { kept, blockedItems } = filterBadAnswer(raw, hits);
  assert.equal(kept.length, 1);
  assert.equal(blockedItems.length, 3);
});

/* ================= 3. 坏模型对照：resources ================= */

test('坏模型 resources：白名单必拦编造条目，blockedItems 可见，真实条目照常放行', () => {
  const out = buildResources({ steps: ['学习 LangGraph'] }, { simulateBadModel: true });
  assert.equal(out.refused, false);
  assert.equal(out.modelCalled, true);
  assert.equal(out.mode, 'bad-model');
  assert.ok(out.blocked >= 2, '两条编造条目必须全被拦');
  assert.equal(out.blockedItems.length, out.blocked);
  for (const b of out.blockedItems) {
    assert.ok(FAKE_HOSTS.some((h) => (b.url || '').includes(h)));
    assert.ok(b.reason.includes('白名单'));
  }
  const allowed = new Set(out.whitelist.map((w) => w.url));
  assert.ok(out.resources.length >= 1, '真实命中条目应放行');
  for (const r of out.resources) {
    assert.ok(allowed.has(r.url), '放行条目 URL 必须在本轮白名单内');
    for (const host of FAKE_HOSTS) assert.ok(!r.url.includes(host));
  }
});

test('坏模型 resources（空检索）：仍然代码级拒答，坏模型不被调用', () => {
  for (const steps of [['乳腺癌的靶向治疗方案有哪些'], []]) {
    const out = buildResources({ steps }, { simulateBadModel: true });
    assert.equal(out.refused, true);
    assert.equal(out.modelCalled, false);
    assert.deepEqual(out.resources, []);
    assert.equal(out.blocked, 0);
  }
});

test('坏模型不改变检索本身：hits/scores 与正常模式逐值一致（约束在输出侧，不在检索侧）', () => {
  const normal = answerQuery('怎么用 Docker 部署 Python 应用');
  const bad = answerQuery('怎么用 Docker 部署 Python 应用', { simulateBadModel: true });
  assert.deepEqual(bad.hits, normal.hits);
  assert.deepEqual(bad.scores, normal.scores);
  const hits = retrieve(RETRIEVER, '怎么用 Docker 部署 Python 应用');
  for (const h of hits) assert.ok(h.score > MIN_SCORE);
});

/* ================= 4. HTTP 路由层透传 ================= */

test('POST ask simulateBadModel：路由透传，线上口径=本地函数', async () => {
  const body = { query: '学习 LangGraph', simulateBadModel: true };
  const r = await handleLearningAgent({ request: req('ask', body), env: () => undefined });
  const j = await r.json();
  assert.deepEqual(j, answerQuery('学习 LangGraph', { simulateBadModel: true }));
  assert.equal(j.mode, 'bad-model');
  assert.ok(j.blocked >= 2);
});

test('POST resources simulateBadModel：路由透传 + 空检索拒答口径一致', async () => {
  const r = await handleLearningAgent({ request: req('resources', { steps: ['学习 LangGraph'], simulateBadModel: true }), env: () => undefined });
  const j = await r.json();
  assert.deepEqual(j, buildResources({ steps: ['学习 LangGraph'] }, { simulateBadModel: true }));
  const r2 = await handleLearningAgent({ request: req('resources', { steps: ['乳腺癌的靶向治疗方案有哪些'], simulateBadModel: true }), env: () => undefined });
  const j2 = await r2.json();
  assert.equal(j2.refused, true);
  assert.equal(j2.modelCalled, false);
});

test('simulateBadModel 非 true 值不生效（严格布尔判定）', () => {
  for (const v of ['true', 1, 'yes']) {
    const out = answerQuery('学习 LangGraph', { simulateBadModel: v });
    assert.equal(out.mode, 'rules');
    assert.equal(out.modelCalled, false);
  }
});