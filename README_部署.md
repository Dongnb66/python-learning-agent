# 在线 Demo 部署指南（HuggingFace Spaces / Render）

本目录下的 `app.py` 是一个 **Gradio 在线演示**，直接复用本仓库真实的 LangGraph 多智能体系统
（`画像 → 计划 → 资源(RAG防幻觉) → 测验 → 复盘`），让面试官/HR 点开链接就能玩，比单纯给 GitHub 仓库更有说服力。

> 仓库本体：https://github.com/Dongnb66/python-learning-agent

---

## 一、两种运行模式（自动判断）

| 条件 | 模式 | 说明 |
|---|---|---|
| **未设置** `LLM_API_KEY` | 🟢 Mock 模式（默认） | 无需任何 Key，零费用即可展示完整 5 智能体流程。LLM 生成部分用写死的合法样例，但**画像正则提取、BM25 真实链接检索都是真实代码**。 |
| **已设置** `LLM_API_KEY` | 🔵 真实模式 | 调用 DeepSeek（默认），生成内容随输入真实变化。 |

**部署到 HuggingFace Spaces 时什么都不用配，默认就是 Mock 模式，直接能跑。**

---

## 二、部署到 HuggingFace Spaces（推荐，免费）

1. 登录 https://huggingface.co ，右上角 **New** → **Space**。
2. 填写：
   - **Space name**：`learning-agent-demo`（或任意）
   - **SDK**：选 **Gradio**
   - **Space 可见性**：Public（方便放简历链接）
3. 创建后进入 Space 页面 → **Files** 标签页：
   - 方式 A（最简单）：把本仓库这些文件/目录**全部上传**到 Space 根目录：
     `app.py`、`mock_llm.py`、`requirements.txt`、`app/`（整个目录，含 `data/resources.json`）
   - 方式 B（用 GitHub 同步）：Space 设置里选 **Import from GitHub**，指向你 fork/推送了本 demo 的仓库。
4. 等待构建（约 1–3 分钟）。构建完成后页面会给出一个 `https://xxx.hf.space` 的公网链接。
5. 把这个链接写进简历「项目经历」第 1 项的末尾，例如：
   > 在线 Demo：https://你的用户名-learning-agent-demo.hf.space

### 想让 Demo 用真实 DeepSeek 生成？
Space 页面 → **Settings** → **Repository secrets / Variables** → 新增：
- **Name**：`LLM_API_KEY`
- **Value**：你的 DeepSeek API Key（`sk-...`）

保存后 Space 会自动重建，之后走「真实模式」。不填则一直是 Mock 模式。

---

## 三、部署到 Render（备选）

1. 登录 https://render.com → **New** → **Web Service**。
2. 关联你的 GitHub 仓库（需先把本 demo 推送到仓库）。
3. 配置：
   - **Build Command**：`pip install -r requirements.txt`
   - **Start Command**：`python app.py`
   - **Environment**：加一个变量 `PORT=7860`（Render 会注入自己的 PORT，代码已兼容）
4. 部署完成后得到 `https://xxx.onrender.com`。

> Render 免费版会休眠（长时间无人访问后冷启动较慢），演示足够用。

---

## 四、本地跑起来看看

```bash
pip install -r requirements.txt
# Mock 模式（默认，无需 Key）
python app.py
# 打开 http://localhost:7860

# 真实模式
export LLM_API_KEY=sk-你的deepseekkey
python app.py
```

---

## 五、文件说明

| 文件 | 作用 |
|---|---|
| `app.py` | Gradio 演示界面，HuggingFace 会自动识别 |
| `mock_llm.py` | Mock LLM 层，无 Key 时让流程跑通（含真实资料链接） |
| `requirements.txt` | 部署依赖（主项目依赖 + gradio） |
| `app/` | 真实多智能体系统源码（被 demo 直接 import 复用） |

---

## 六、给简历的写法建议

在「python-learning-agent」项目条目里补一句：

> 在线 Demo（可点开体验完整 5 智能体流程）：https://你的用户名-learning-agent-demo.hf.space

这比「GitHub: Dongnb66/python-learning-agent」更有冲击力——面试官点一下就能看到你的系统真在跑。
