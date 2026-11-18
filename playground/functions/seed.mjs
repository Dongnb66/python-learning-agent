// seed.mjs —— 公网演示版种子数据（无托管 DB：平台迁移 API 不可用，按已验证兜底模式内置种子）。
// CORPUS 由 scripts/build-playground.mjs 从 app/data/resources.json 原样生成（corpus.mjs），
// 12 条全部内嵌：BM25 的 IDF/avgdl 依赖整个语料，取子集会改变分数、破坏与后端同参数对拍。
// 来源：python-learning-agent 仓库 app/data/resources.json（课程知识库，标题/URL/正文逐字同源）。
// PLAN / QUIZ / PROFILE_SAMPLE 逐字取自 app/mock_llm.py 的零 Key 样例输出（规则版规划/题库同源）。
import { CORPUS } from './corpus.mjs';

export const CORPUS_SOURCE = 'app/data/resources.json（python-learning-agent 课程知识库，12 条全量内嵌）';

// 与 app/mock_llm.py _sample_plan() 逐字同源
export const PLAN = {
  goal: '从零构建并部署第一个 AI Agent 项目',
  steps: [
    { step: 1, title: 'Python 与 FastAPI 基础巩固', description: '补齐类型注解、Pydantic、异步视图，能独立写一个小 API。', est_minutes: 180 },
    { step: 2, title: 'LangChain / LangGraph 多智能体入门', description: '理解 StateGraph 如何把多个 Agent 串成有状态工作流。', est_minutes: 240 },
    { step: 3, title: 'RAG 检索增强实践', description: '用 BM25 接入本地资料库，理解防幻觉机制。', est_minutes: 200 },
    { step: 4, title: '动手做一个校园场景 Agent', description: '把前几步串起来，做一个能跑的最小可用 Agent。', est_minutes: 300 },
    { step: 5, title: '部署与工程化（Docker）', description: '容器化并部署到 HuggingFace / Render，形成作品集。', est_minutes: 150 },
  ],
  total_minutes: 1070,
};

// 与 app/mock_llm.py _sample_quiz() 逐字同源
export const QUIZ = {
  questions: [
    {
      q: 'LangGraph 中用什么把多个智能体串成有状态工作流？',
      options: ['StateGraph', '普通函数调用', 'SQL 触发器', 'CSS 动画'],
      answer: 'StateGraph',
      explanation: 'StateGraph 用节点+边定义多智能体流程，支持状态持久化与条件分支。',
    },
    {
      q: 'RAG 主要通过什么手段缓解大模型幻觉？',
      options: ['把相关资料检索后拼进提示', '增大模型参数量', '更换 GPU', '减少训练数据'],
      answer: '把相关资料检索后拼进提示',
      explanation: 'RAG 用检索到的真实资料作为事实依据约束生成，降低编造风险。',
    },
    {
      q: 'FastAPI 依靠哪个库实现请求体类型校验与自动文档？',
      options: ['Pydantic', 'NumPy', 'Pandas', 'Requests'],
      answer: 'Pydantic',
      explanation: 'FastAPI 基于 Pydantic 做数据校验，并据此自动生成 OpenAPI 文档。',
    },
  ],
};

// 与 app/mock_llm.py _sample_resources() 的 relevance 文案同源
export const RESOURCE_RELEVANCE = '与学习计划中的对应步骤直接匹配，且为官方 / 权威资料。';

export const SEED = { corpus: CORPUS, plan: PLAN, quiz: QUIZ };