import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { api } from '../api/client';
import Markdown from '../components/Markdown';
import MindMapView from '../components/MindMapView';

function ResourceDetail() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [resource, setResource] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [activeTab, setActiveTab] = useState('lecture');
  const [quizAnswers, setQuizAnswers] = useState({});
  const [showAnswers, setShowAnswers] = useState(false);

  useEffect(() => {
    setLoading(true);
    setError('');
    setResource(null);
    api.getResource(id).then(res => {
      setResource(res.data?.resource);
      setLoading(false);
    }).catch(err => {
      setError(err.message || '加载资源失败');
      setLoading(false);
    });
  }, [id]);

  if (loading) return <div className="loading-overlay"><div className="loading-spinner"></div><span className="loading-text">加载资源中...</span></div>;
  if (error || !resource) return (
    <div className="card" style={{textAlign: 'center', padding: '40px'}}>
      <div className="error-msg" style={{marginBottom: '20px'}}>
        {error || '资源不存在或已被删除'}
      </div>
      <p style={{color: '#666', fontSize: '14px', marginBottom: '20px'}}>
        URL 中的资源 ID：{id}
      </p>
      <button className="btn btn-primary" onClick={() => navigate('/')}>
        返回首页
      </button>
    </div>
  );

  const tabs = [
    { key: 'lecture', label: '📖 讲解文档' },
    { key: 'mindmap', label: '🧠 思维导图' },
    { key: 'quiz', label: '✏️ 练习题' },
    { key: 'reading', label: '📚 拓展阅读' },
    { key: 'case', label: '💻 实操案例' },
    { key: 'review', label: '🛡️ 质量检查' }
  ];

  return (
    <div>
      <div className="card">
        <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center'}}>
          <div>
            <h2 style={{color: '#1a237e'}}>{resource.course} - {resource.topic}</h2>
            <p style={{color: '#666', fontSize: '14px', marginTop: '4px'}}>
              学生: {resource.studentName} | 生成时间: {new Date(resource.createdAt).toLocaleString('zh-CN')}
            </p>
          </div>
          <span className={`badge ${resource.review?.passed ? 'badge-success' : 'badge-warning'}`} style={{fontSize: '14px', padding: '6px 14px'}}>
            质量评分: {resource.review?.score}
          </span>
        </div>
      </div>

      {resource.knowledgeSources && resource.knowledgeSources.length > 0 && (
        <div className="card">
          <h4 style={{color: '#e65100', marginBottom: '8px'}}>📌 引用来源（RAG 知识库）</h4>
          {resource.knowledgeSources.map((src, i) => (
            <div key={i} className="source-item">
              <span>📄</span>
              <span className="source-doc">{src.document}</span>
              <span>→</span>
              <span>{src.section}</span>
            </div>
          ))}
        </div>
      )}

      <div className="card">
        <div style={{display: 'flex', gap: '4px', marginBottom: '16px', flexWrap: 'wrap'}}>
          {tabs.map(tab => (
            <button
              key={tab.key}
              className={`btn btn-sm ${activeTab === tab.key ? 'btn-primary' : 'btn-secondary'}`}
              onClick={() => setActiveTab(tab.key)}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {activeTab === 'lecture' && <Markdown content={resource.lecture} />}

        {activeTab === 'mindmap' && <MindMapView data={resource.mindMap} />}

        {activeTab === 'quiz' && (
          <div>
            {resource.quiz && resource.quiz.map((q, i) => (
              <div key={i} className="quiz-item">
                <div className="quiz-question">{i + 1}. [{q.type === 'choice' ? '选择题' : q.type === 'fill' ? '填空题' : '简答题'}] {q.question}</div>
                {q.options && q.options.map((opt, j) => {
                  const letter = opt[0];
                  const isCorrect = showAnswers && letter === q.answer;
                  const isWrong = showAnswers && quizAnswers[i] === letter && letter !== q.answer;
                  return (
                    <div
                      key={j}
                      className={`quiz-option ${isCorrect ? 'correct' : ''} ${isWrong ? 'wrong' : ''}`}
                      onClick={() => setQuizAnswers({...quizAnswers, [i]: letter})}
                    >
                      {opt}
                    </div>
                  );
                })}
                {showAnswers && (
                  <>
                    <div className="quiz-answer">✓ 正确答案: {q.answer}</div>
                    <div className="quiz-explanation">解析: {q.explanation}</div>
                  </>
                )}
              </div>
            ))}
            <button className="btn btn-primary" onClick={() => setShowAnswers(!showAnswers)}>
              {showAnswers ? '隐藏答案' : '显示答案'}
            </button>
          </div>
        )}

        {activeTab === 'reading' && (
          <div>
            {resource.reading && resource.reading.map((r, i) => (
              <div key={i} className="resource-card">
                <div className="resource-card-header">
                  <span className="resource-card-icon">📄</span>
                  <span className="resource-card-title">{r.title}</span>
                  <span className="badge badge-info">{r.source}</span>
                </div>
                <p style={{fontSize: '13px', color: '#666'}}>{r.description}</p>
                {r.url && <a href={r.url} target="_blank" rel="noreferrer" style={{color: '#1a73e8', fontSize: '13px'}}>访问链接 →</a>}
              </div>
            ))}
          </div>
        )}

        {activeTab === 'case' && resource.caseStudy && (
          <div>
            <h3 style={{color: '#1a237e', marginBottom: '8px'}}>{resource.caseStudy.title}</h3>
            <p style={{color: '#666', fontSize: '14px', marginBottom: '12px'}}>{resource.caseStudy.description}</p>
            <div className="code-block">
              <pre>{resource.caseStudy.code}</pre>
            </div>
            <div style={{background: '#e8f5e9', padding: '12px', borderRadius: '8px', marginTop: '12px'}}>
              <strong>💡 说明：</strong>{resource.caseStudy.explanation}
            </div>
          </div>
        )}

        {activeTab === 'review' && resource.review && (
          <div>
            <div style={{display: 'flex', gap: '16px', marginBottom: '20px'}}>
              <div style={{textAlign: 'center', padding: '20px', background: resource.review.passed ? '#e8f5e9' : '#fff3e0', borderRadius: '12px', flex: 1}}>
                <div style={{fontSize: '36px', fontWeight: 700, color: resource.review.passed ? '#34a853' : '#ff9800'}}>
                  {resource.review.score}
                </div>
                <div style={{fontSize: '13px', color: '#666'}}>质量评分</div>
              </div>
              <div style={{textAlign: 'center', padding: '20px', background: '#e3f2fd', borderRadius: '12px', flex: 1}}>
                <div style={{fontSize: '36px'}}>{resource.review.passed ? '✅' : '⚠️'}</div>
                <div style={{fontSize: '13px', color: '#666'}}>{resource.review.passed ? '检查通过' : '需要关注'}</div>
              </div>
            </div>
            {resource.review.issues && resource.review.issues.length > 0 ? (
              <div>
                <h4 style={{marginBottom: '8px'}}>检查问题</h4>
                {resource.review.issues.map((issue, i) => (
                  <div key={i} className="source-item" style={{background: issue.severity === 'critical' ? '#fce4ec' : '#fff3e0'}}>
                    <span>{issue.severity === 'critical' ? '🔴' : '🟡'}</span>
                    <span>{issue.message}</span>
                  </div>
                ))}
              </div>
            ) : (
              <div style={{textAlign: 'center', padding: '20px', color: '#34a853'}}>
                ✅ 所有检查项均通过
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default ResourceDetail;
