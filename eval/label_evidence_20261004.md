# 标注集机械证据校验报告（2026-10-04）

> 由 scripts/label_evidence_check.py 自动生成；规则客观、可复现，不替代人工判断。
> 用途：把「要人工看 36 条」缩小成「只看下面这些候选」，并为每条给出可读证据句。

- 参与校验：36 条（A/B 组应命中 + 争议条）
- 语料：12 篇（app/data/resources.json）
- 判据：score = 查询与某句共有的中文双字词/英文词个数；score < 2 记为语料无依据候选
- 候选：**9 条**（建议优先人工确认；若确认无依据应归入 C 组）

## 一、语料无依据候选（优先看这些）

| 组 | 查询 | 期望文档 | 最佳证据句 | score |
|---|---|---|---|---|
| A | FastAPI 怎么写接口 | FastAPI 官方教程 | FastAPI 官方教程 https://fastapi.tiangolo.com/zh/ doc FastAPI 中文… | 1 |
| B | 写 HTTP 接口的 Python 框架推荐 | FastAPI 官方教程 | 是构建 AI 服务 API 的首选框架，求职 Agent 岗位高频要求 | 1 |
| B | 怎么用代码自动生成接口文档 | FastAPI 官方教程 | FastAPI 官方教程 https://fastapi.tiangolo.com/zh/ doc FastAPI 中文… | 1 |
| B | 切片、向量召回、重排这一套怎么做 | RAG 检索增强生成原理与实践 | RAG 检索增强生成原理与实践 https://python.langchain.com.cn/docs/modules… | 1 |
| B | ORM 怎么建模型和写查询 | SQLAlchemy 2.0 快速上手 | （该文档里没有任何一句与查询有共同词） | 0 |
| B | 用容器一条命令起一套环境 | Docker 容器化 Python 应用 | Docker 容器化 Python 应用 https://docs.docker.com/get-started/ do… | 1 |
| B | 刷题应该从哪一类题开始做 | 数据结构与算法：数组、链表、哈希表 | （该文档里没有任何一句与查询有共同词） | 0 |
| B | 请求头和状态码分别是什么意思 | 计算机网络：HTTP/TCP 核心概念 | （该文档里没有任何一句与查询有共同词） | 0 |
| B | 怎么写查询语句并建索引 | 数据库原理：SQL 与索引 | 数据库原理：SQL 与索引 https://www.runoob.com/sql/sql-tutorial.html d… | 1 |

## 二、全部查询的证据句与重合度

| 组 | 查询 | 期望文档 | 最佳证据句 | score | ratio |
|---|---|---|---|---|---|
| B | ORM 怎么建模型和写查询 | SQLAlchemy 2.0 快速上手 | （无） | 0 | 0% |
| B | 刷题应该从哪一类题开始做 | 数据结构与算法：数组、链表、哈希表 | （无） | 0 | 0% |
| B | 请求头和状态码分别是什么意思 | 计算机网络：HTTP/TCP 核心概念 | （无） | 0 | 0% |
| A | FastAPI 怎么写接口 | FastAPI 官方教程 | FastAPI 官方教程 https://fastapi.tiangolo.com/zh/ doc FastAPI 中文文档，讲解路径参数、… | 1 | 25% |
| B | 写 HTTP 接口的 Python 框架推荐 | FastAPI 官方教程 | 是构建 AI 服务 API 的首选框架，求职 Agent 岗位高频要求 | 1 | 14% |
| B | 怎么用代码自动生成接口文档 | FastAPI 官方教程 | FastAPI 官方教程 https://fastapi.tiangolo.com/zh/ doc FastAPI 中文文档，讲解路径参数、… | 1 | 9% |
| B | 切片、向量召回、重排这一套怎么做 | RAG 检索增强生成原理与实践 | RAG 检索增强生成原理与实践 https://python.langchain.com.cn/docs/modules/data_conn… | 1 | 10% |
| B | 用容器一条命令起一套环境 | Docker 容器化 Python 应用 | Docker 容器化 Python 应用 https://docs.docker.com/get-started/ doc Docker 入… | 1 | 9% |
| B | 怎么写查询语句并建索引 | 数据库原理：SQL 与索引 | 数据库原理：SQL 与索引 https://www.runoob.com/sql/sql-tutorial.html doc SQL 教程，… | 1 | 11% |
| A | SQLAlchemy 2.0 怎么上手 | SQLAlchemy 2.0 快速上手 | SQLAlchemy 2.0 快速上手 https://docs.sqlalchemy.org/en/20/ doc SQLAlchemy … | 2 | 67% |
| A | 提示词工程有什么技巧 | Prompt Engineering 指南 | Prompt Engineering 指南 https://www.promptingguide.ai/zh article 系统讲解 Ze… | 2 | 25% |
| A | SQL 索引和查询语法 | 数据库原理：SQL 与索引 | 数据库原理：SQL 与索引 https://www.runoob.com/sql/sql-tutorial.html doc SQL 教程，… | 2 | 29% |
| B | 零基础学门语言，先搞懂变量和循环 | Python 入门：变量、循环与函数 | Python 入门：变量、循环与函数 https://docs.python.org/zh-cn/3/tutorial/ doc Pytho… | 2 | 15% |
| B | 函数怎么定义和调用 | Python 入门：变量、循环与函数 | Python 入门：变量、循环与函数 https://docs.python.org/zh-cn/3/tutorial/ doc Pytho… | 2 | 29% |
| B | 提示模板和模型调用怎么封装 | LangChain 中文文档：模型与提示 | LangChain 中文文档：模型与提示 https://python.langchain.com.cn/ doc LangChain 中文… | 2 | 18% |
| B | 支持有环流程图的编排框架 | LangGraph 多智能体编排入门 | LangGraph 多智能体编排入门 https://langchain-ai.github.io/langgraph/ doc LangG… | 2 | 18% |
| B | Python 里用对象来操作数据库表 | SQLAlchemy 2.0 快速上手 | 用于把学生画像、学习记录持久化到数据库 | 2 | 18% |
| B | 思维链和少样本示例该怎么给 | Prompt Engineering 指南 | Prompt Engineering 指南 https://www.promptingguide.ai/zh article 系统讲解 Ze… | 2 | 18% |
| B | 我的第一个监督学习模型怎么训 | 机器学习基础：回归与分类 | 机器学习基础：回归与分类 https://scikit-learn.org/stable/tutorial/basic/tutorial.h… | 2 | 18% |
| A | LangChain 的模型和提示怎么用 | LangChain 中文文档：模型与提示 | LangChain 中文文档：模型与提示 https://python.langchain.com.cn/ doc LangChain 中文… | 3 | 38% |
| B | 把大模型接进应用要学什么框架 | LangChain 中文文档：模型与提示 | 理解 Prompt 工程是落地大模型应用的第一步 | 3 | 25% |
| B | 链表和哈希表不会，想系统补一遍 | 数据结构与算法：数组、链表、哈希表 | 数据结构与算法：数组、链表、哈希表 https://leetcode.cn/leetbook/ course LeetCode 精选题库，覆… | 3 | 25% |
| B | 三次握手为什么必须是三次 | 计算机网络：HTTP/TCP 核心概念 | 计算机网络：HTTP/TCP 核心概念 https://www.bilibili.com/video/BV1c4411d7jb video … | 3 | 33% |
| B | 回归和分类的区别与入门 | 机器学习基础：回归与分类 | 机器学习基础：回归与分类 https://scikit-learn.org/stable/tutorial/basic/tutorial.h… | 3 | 30% |
| A | Python 变量和循环怎么入门 | Python 入门：变量、循环与函数 | Python 入门：变量、循环与函数 https://docs.python.org/zh-cn/3/tutorial/ doc Pytho… | 4 | 50% |
| A | Python 应用怎么容器化部署 | Docker 容器化 Python 应用 | Docker 容器化 Python 应用 https://docs.docker.com/get-started/ doc Docker 入… | 4 | 50% |
| A | 数据结构和算法怎么刷题 | 数据结构与算法：数组、链表、哈希表 | 数据结构与算法：数组、链表、哈希表 https://leetcode.cn/leetbook/ course LeetCode 精选题库，覆… | 4 | 44% |
| B | 让模型基于我自己的文档回答问题 | RAG 检索增强生成原理与实践 | RAG 检索增强生成原理与实践 https://python.langchain.com.cn/docs/modules/data_conn… | 4 | 29% |
| B | 怎么把服务打包成镜像部署上线 | Docker 容器化 Python 应用 | 把 Agent 服务容器化是部署上线的必备技能 | 4 | 33% |
| B | 怎么靠指令设计把模型输出稳定住 | Prompt Engineering 指南 | Prompt Engineering 指南 https://www.promptingguide.ai/zh article 系统讲解 Ze… | 4 | 31% |
| B | 关系数据库的表连接怎么理解 | 数据库原理：SQL 与索引 | 理解关系型数据库是后端与数据岗的基本功 | 4 | 36% |
| A | HTTP 和 TCP 的核心概念 | 计算机网络：HTTP/TCP 核心概念 | 计算机网络：HTTP/TCP 核心概念 https://www.bilibili.com/video/BV1c4411d7jb video … | 5 | 83% |
| A | LangGraph 怎么做多智能体编排 | LangGraph 多智能体编排入门 | LangGraph 多智能体编排入门 https://langchain-ai.github.io/langgraph/ doc LangG… | 6 | 75% |
| A | 机器学习回归分类入门 | 机器学习基础：回归与分类 | 机器学习基础：回归与分类 https://scikit-learn.org/stable/tutorial/basic/tutorial.h… | 6 | 67% |
| B | 多个智能体之间怎么编排状态流转 | LangGraph 多智能体编排入门 | LangGraph 多智能体编排入门 https://langchain-ai.github.io/langgraph/ doc LangG… | 6 | 46% |
| A | RAG 检索增强生成的原理 | RAG 检索增强生成原理与实践 | RAG 检索增强生成原理与实践 https://python.langchain.com.cn/docs/modules/data_conn… | 7 | 78% |

## 三、边界说明（必须一起读）

1. 本报告是机械筛选，不是标注结论：重合度低不等于标错。
2. B 组本来就是「词面不匹配」的同义改写 —— 它的低重合是设计使然，不能据此判错。
3. 真正要人工确认的是第一节那批候选：它们连一句共享词都找不出来。
4. 本脚本不修改任何标注、不改动任何指标口径（主口径仍是 36 条全算）。
