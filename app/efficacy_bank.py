"""前后测诊断题库（学习效果实证模块）。

从 A3 前端的 efficacyBank.js 等价移植：每个知识点一组诊断题，
用于「学前 pre / 学后 post」两轮测验对比，量化系统对成绩的提升。
提交后由服务端判分（题目下发时隐藏答案），防止前端改分。
"""
from __future__ import annotations

_BANK: dict[str, list[dict]] = {
    "机器学习": [
        {"id": "ml1", "knowledgePoint": "监督学习概念", "question": "以下哪项属于监督学习？", "options": ["A. K-Means 聚类", "B. 决策树分类", "C. PCA 降维", "D. 关联规则"], "answer": 1, "explanation": "决策树分类使用带标签数据训练，属于监督学习。"},
        {"id": "ml2", "knowledgePoint": "过拟合", "question": "缓解过拟合最有效的是？", "options": ["A. 增加模型复杂度", "B. 减少训练数据", "C. 正则化与更多数据", "D. 提高学习率"], "answer": 2, "explanation": "正则化和增加数据都能抑制过拟合。"},
        {"id": "ml3", "knowledgePoint": "模型评估", "question": "F1 值是哪两个指标的调和平均？", "options": ["A. 准确率与召回率", "B. 精确率与召回率", "C. 精确率与准确率", "D. 准确率与FPR"], "answer": 1, "explanation": "F1 = 2PR/(P+R)，P为精确率，R为召回率。"},
        {"id": "ml4", "knowledgePoint": "交叉验证", "question": "交叉验证的主要目的是？", "options": ["A. 加速训练", "B. 更可靠评估泛化性能", "C. 降低特征维度", "D. 提升精度上限"], "answer": 1, "explanation": "交叉验证通过多次划分训练/验证集，更稳健地估计泛化能力。"},
        {"id": "ml5", "knowledgePoint": "无监督学习", "question": "以下哪个是无监督学习任务？", "options": ["A. 邮件分类", "B. 房价回归", "C. 客户分群", "D. 图像标注"], "answer": 2, "explanation": "客户分群（聚类）在无标签数据上发现结构，属无监督学习。"},
    ],
    "决策树": [
        {"id": "dt1", "knowledgePoint": "ID3", "question": "ID3 算法使用什么指标选择划分属性？", "options": ["A. 基尼指数", "B. 信息增益", "C. 信息增益率", "D. 均方误差"], "answer": 1, "explanation": "ID3 用信息增益，C4.5 用信息增益率，CART 用基尼指数。"},
        {"id": "dt2", "knowledgePoint": "C4.5", "question": "C4.5 相比 ID3 主要改进了什么？", "options": ["A. 无法处理连续值", "B. 偏向取值多的属性", "C. 解决取值多属性偏向问题", "D. 计算更慢"], "answer": 2, "explanation": "C4.5 用信息增益率，缓解了 ID3 偏向多值属性的缺陷。"},
        {"id": "dt3", "knowledgePoint": "随机森林", "question": "随机森林每棵树用什么数据训练？", "options": ["A. 全部数据", "B. 一半数据", "C. Bootstrap 有放回采样", "D. 测试集"], "answer": 2, "explanation": "随机森林对样本 bootstrap 采样、对特征随机子集训练，再投票。"},
        {"id": "dt4", "knowledgePoint": "剪枝", "question": "预剪枝的主要作用是？", "options": ["A. 提升过拟合", "B. 提前停止树生长防过拟合", "C. 增加树深度", "D. 加速预测"], "answer": 1, "explanation": "预剪枝在生成时限制深度等，防止过拟合。"},
        {"id": "dt5", "knowledgePoint": "CART", "question": "CART 分类树使用的划分准则是？", "options": ["A. 信息增益", "B. 基尼指数", "C. 卡方", "D. 互信息"], "answer": 1, "explanation": "CART 使用基尼指数选择划分。"},
    ],
    "神经网络": [
        {"id": "nn1", "knowledgePoint": "激活函数", "question": "ReLU 的表达式是？", "options": ["A. 1/(1+e^-x)", "B. max(0,x)", "C. (e^x-e^-x)/(e^x+e^-x)", "D. e^x/Σe^x"], "answer": 1, "explanation": "ReLU(x)=max(0,x)，缓解梯度消失。"},
        {"id": "nn2", "knowledgePoint": "反向传播", "question": "反向传播计算梯度依赖什么数学工具？", "options": ["A. 泰勒展开", "B. 链式法则", "C. 拉格朗日乘子", "D. 牛顿法"], "answer": 1, "explanation": "反向传播用链式法则逐层求梯度。"},
        {"id": "nn3", "knowledgePoint": "梯度消失", "question": "缓解梯度消失的方法不包括？", "options": ["A. ReLU", "B. 残差连接", "C. 批量归一化", "D. 加深网络"], "answer": 3, "explanation": "单纯加深网络会加剧梯度消失，需用 ReLU/残差/BN 等缓解。"},
        {"id": "nn4", "knowledgePoint": "CNN", "question": "以下哪项是 CNN 的核心组件？", "options": ["A. 卷积层", "B. 循环层", "C. 自注意力", "D. 门控单元"], "answer": 0, "explanation": "CNN 核心是卷积层与池化层。"},
        {"id": "nn5", "knowledgePoint": "Transformer", "question": "Transformer 的核心机制是？", "options": ["A. 循环结构", "B. 自注意力机制", "C. 卷积", "D. 记忆网络"], "answer": 1, "explanation": "Transformer 以自注意力替代循环结构。"},
    ],
}

SEED_SAMPLES = [
    {"studentName": "李同学", "topic": "机器学习", "pre": 58, "post": 86},
    {"studentName": "王同学", "topic": "决策树", "pre": 62, "post": 88},
    {"studentName": "赵同学", "topic": "神经网络", "pre": 55, "post": 82},
    {"studentName": "陈同学", "topic": "机器学习", "pre": 60, "post": 84},
    {"studentName": "刘同学", "topic": "决策树", "pre": 57, "post": 85},
]


def topics() -> list[str]:
    return list(_BANK.keys())


def get_questions(topic: str) -> list[dict]:
    return _BANK.get(topic) or _BANK["机器学习"]


def strip_answers(questions: list[dict]) -> list[dict]:
    """下发题目时去掉答案与解析，防止前端作弊。"""
    return [{k: v for k, v in q.items() if k not in ("answer", "explanation")} for q in questions]


def score(answers: dict, topic: str) -> dict:
    """服务端判分。answers 形如 {question_id: 选项下标}。"""
    qs = get_questions(topic)
    correct = 0
    detail = []
    for q in qs:
        user_ans = answers.get(q["id"])
        is_correct = user_ans is not None and str(user_ans) == str(q["answer"])
        if is_correct:
            correct += 1
        detail.append(
            {
                "id": q["id"],
                "knowledgePoint": q["knowledgePoint"],
                "correct": is_correct,
                "userAnswer": user_ans,
                "correctAnswer": q["answer"],
                "explanation": q["explanation"],
            }
        )
    total = len(qs)
    return {"correct": correct, "total": total, "score": round(correct / total * 100) if total else 0, "detail": detail}
