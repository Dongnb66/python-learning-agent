// 公网演示版构建：
// 1) 从 app/data/resources.json 原样生成 corpus.mjs（同源保证：语料逐字来自后端知识库）；
// 2) 拷贝 playground/{web,functions} 到会话工作区（Sites projectRoot 边界）；
// 3) 产物守卫：扫描禁字面量（平台扫描坑 + 隐私），命中即构建失败。
// 不改本地完整版源码与原测试。用法: node scripts/build-playground.mjs <输出目录>
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const out = process.argv[2];
if (!out) throw new Error("用法: node scripts/build-playground.mjs <输出目录>");

// ---- 1) 生成 corpus.mjs（逐字内嵌后端知识库，含来源标注） ----
const corpusPath = path.join(root, "app", "data", "resources.json");
const corpus = JSON.parse(fs.readFileSync(corpusPath, "utf-8"));
if (!Array.isArray(corpus) || !corpus.length) throw new Error("resources.json 为空");
const corpusModule =
  "// corpus.mjs —— 由 scripts/build-playground.mjs 从 app/data/resources.json 原样生成，勿手改。\n" +
  "// 来源：python-learning-agent 课程知识库（标题/URL/类型/正文逐字同源，全量 " + corpus.length + " 条）。\n" +
  "export const CORPUS = " + JSON.stringify(corpus, null, 2) + ";\n";
const genTargets = [path.join(root, "playground", "functions", "corpus.mjs"), path.join(out, "functions", "corpus.mjs")];

// ---- 2) 拷贝 web + functions ----
function copyDir(src, dst) {
  fs.mkdirSync(dst, { recursive: true });
  for (const e of fs.readdirSync(src, { withFileTypes: true })) {
    const s = path.join(src, e.name), d = path.join(dst, e.name);
    if (e.isDirectory()) copyDir(s, d);
    else fs.copyFileSync(s, d);
  }
}
copyDir(path.join(root, "playground", "web"), path.join(out, "web"));
copyDir(path.join(root, "playground", "functions"), path.join(out, "functions"));
for (const t of genTargets) {
  fs.mkdirSync(path.dirname(t), { recursive: true });
  fs.writeFileSync(t, corpusModule, "utf-8");
}

// ---- 3) 产物守卫 ----
const FORBIDDEN = [
  "pt-online-schema-change",
  "127.0.0.1",
  "localhost",
  "杨运栋",
  "console 脚本生成器",
];
const KEY_RE = /sk-[A-Za-z0-9]{20,}/;
function* walk(dir) {
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, e.name);
    if (e.isDirectory()) yield* walk(p);
    else yield p;
  }
}
let violations = 0;
for (const p of walk(out)) {
  const text = fs.readFileSync(p, "utf-8");
  for (const bad of FORBIDDEN) {
    if (text.includes(bad)) { console.error(`守卫命中: ${p} 含禁字面量 ${bad}`); violations += 1; }
  }
  if (KEY_RE.test(text)) { console.error(`守卫命中: ${p} 疑似 API Key`); violations += 1; }
}
if (violations) throw new Error(`产物守卫失败：${violations} 处命中`);
console.log("learning-agent playground 已生成：" + out + "（语料 " + corpus.length + " 条，守卫通过）");