import { useState, useEffect } from 'react';
import { api } from '../api/client';

function LearningPath() {
  const [profile, setProfile] = useState(null);
  const [path, setPath] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    const p = sessionStorage.getItem('currentProfile');
    if (p) {
      const parsed = JSON.parse(p);
      setProfile(parsed);
      loadPath(parsed.id);
    }
  }, []);

  const loadPath = async (profileId) => {
    setLoading(true);
    setError('');
    try {
      const res = await api.getPath(profileId);
      setPath(res.data?.path);
    } catch (err) {
      setError(err.message);
    }
    setLoading(false);
  };

  if (!profile) {
    return (
      <div className="card">
        <div className="empty-state">
          <div className="empty-state-icon">🗺️</div>
          <p>请先完成画像对话，再查看学习路径</p>
        </div>
      </div>
    );
  }

  return (
    <div>
      <div className="card">
        <h2 className="card-title">🗺️ 个性化学习路径</h2>
        <p style={{color: '#666', fontSize: '14px'}}>
          为 {profile.studentName}（{profile.major}）定制的 {path?.course || '人工智能导论'} 学习路径
        </p>
      </div>

      {loading && <div className="loading-overlay"><div className="loading-spinner"></div><span className="loading-text">规划学习路径中...</span></div>}
      {error && <div className="error-msg">❌ {error}</div>}

      {path && path.stages && (
        <div className="card">
          {path.stages.map((stage, i) => (
            <div key={i} className="path-stage">
              <div className="path-stage-dot"></div>
              <div style={{marginLeft: '12px'}}>
                <div style={{display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px'}}>
                  <span style={{background: '#1a237e', color: '#fff', borderRadius: '50%', width: '24px', height: '24px', display: 'inline-flex', alignItems: 'center', justifyContent: 'center', fontSize: '12px', fontWeight: 600}}>
                    {stage.stage}
                  </span>
                  <h3 style={{color: '#1a237e'}}>{stage.title}</h3>
                  <span className="badge badge-info">{stage.estimatedTime}</span>
                </div>
                <p style={{color: '#666', fontSize: '14px', marginBottom: '10px'}}>🎯 {stage.objective}</p>
                <div style={{marginBottom: '8px'}}>
                  <strong style={{fontSize: '13px'}}>知识点：</strong>
                  {stage.topics && stage.topics.map((t, j) => (
                    <span key={j} className="badge badge-info" style={{marginRight: '4px'}}>{t}</span>
                  ))}
                </div>
                <div>
                  <strong style={{fontSize: '13px'}}>学习资源：</strong>
                  {stage.resources && stage.resources.map((r, j) => (
                    <span key={j} className="badge badge-success" style={{marginRight: '4px'}}>{r}</span>
                  ))}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default LearningPath;
