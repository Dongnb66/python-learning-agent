import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../api/client';

function Home() {
  const navigate = useNavigate();
  const [resources, setResources] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.getResources().then(res => {
      setResources(res.data?.resources || []);
      setLoading(false);
    }).catch(() => setLoading(false));
  }, []);

  return (
    <div>
      <div className="card" style={{background: 'linear-gradient(135deg, #1a237e 0%, #283593 100%)', color: '#fff'}}>
        <h1 style={{fontSize: '28px', marginBottom: '8px'}}>欢迎使用个性化学习智能体系统</h1>
        <p style={{fontSize: '15px', opacity: 0.9, marginBottom: '20px'}}>基于大模型的多智能体协同学习资源生成系统</p>
        <div style={{display: 'flex', gap: '12px'}}>
          <button className="btn btn-secondary" onClick={() => navigate('/profile/chat')}>💬 开始画像对话</button>
          <button className="btn btn-secondary" onClick={() => navigate('/generate')}>⚡ 生成学习资源</button>
          <button className="btn btn-secondary" onClick={() => navigate('/tutor')}>❓ 智能辅导</button>
        </div>
      </div>

      <div className="grid-3">
        <div className="card">
          <div style={{fontSize: '36px', marginBottom: '8px'}}>🎯</div>
          <h3 style={{marginBottom: '8px', color: '#1a237e'}}>个性化画像</h3>
          <p style={{fontSize: '13px', color: '#666'}}>通过自然语言对话，自动构建 6 维度学生画像</p>
        </div>
        <div className="card">
          <div style={{fontSize: '36px', marginBottom: '8px'}}>🤖</div>
          <h3 style={{marginBottom: '8px', color: '#1a237e'}}>多智能体协同</h3>
          <p style={{fontSize: '13px', color: '#666'}}>5 个智能体协作生成 5 类学习资源</p>
        </div>
        <div className="card">
          <div style={{fontSize: '36px', marginBottom: '8px'}}>🛡️</div>
          <h3 style={{marginBottom: '8px', color: '#1a237e'}}>RAG 防幻觉</h3>
          <p style={{fontSize: '13px', color: '#666'}}>基于知识库的问答，确保内容准确性</p>
        </div>
      </div>

      <div className="card">
        <h2 className="card-title">最近生成记录</h2>
        {loading ? (
          <div className="loading-overlay"><div className="loading-spinner"></div><span className="loading-text">加载中...</span></div>
        ) : resources.length === 0 ? (
          <div className="empty-state">
            <div className="empty-state-icon">📭</div>
            <p>暂无生成记录，点击"生成学习资源"开始体验</p>
          </div>
        ) : (
          <table className="data-table">
            <thead>
              <tr>
                <th>课程</th>
                <th>知识点</th>
                <th>学生</th>
                <th>质量评分</th>
                <th>状态</th>
                <th>创建时间</th>
                <th>操作</th>
              </tr>
            </thead>
            <tbody>
              {resources.map(r => (
                <tr key={r.id}>
                  <td>{r.course}</td>
                  <td>{r.topic}</td>
                  <td>{r.studentName}</td>
                  <td>{r.reviewScore || '-'}</td>
                  <td>{r.reviewPassed ? <span className="badge badge-success">通过</span> : <span className="badge badge-warning">待检查</span>}</td>
                  <td>{new Date(r.createdAt).toLocaleString('zh-CN')}</td>
                  <td><button className="btn btn-sm btn-primary" onClick={() => navigate(`/resource/${r.id}`)}>查看</button></td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}

export default Home;
