"""智能体包。

每个智能体实现为一个 LangGraph 节点函数，接收 AgentState，返回需要更新的字段字典。
当前包含：
- ProfileAgent  学生画像抽取
- PlannerAgent  学习计划生成
- ResourceAgent RAG 增强的资源推荐
- QuizAgent     自测题生成
- ReviewAgent   学情复盘
"""
