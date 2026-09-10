import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../api/client';

const AGENTS = [
  { name: 'ProfileAgent', label: '画像智能体', icon: '🎯', desc: '从对话中提取 6 维学生画像' },
  { name: 'PlannerAgent', label: '规划智能体', icon: '🗺️', desc: '根据画像和知识库生成学习路径' },
  { name: 'ResourceAgent', label: '资源智能体', icon: '📚', desc: '生成讲解、导图、阅读、案例' },
  { name: 'QuizAgent', label: '练习智能体', icon: '✏️', desc: '生成匹配难度的练习题' },
  { name: 'ReviewAgent', label: '审查智能体', icon: '🛡️', desc: '检查防幻觉、难度匹配、完整性' }
];


function Generate() {
  const navigate = useNavigate();
  const [profile, setProfile] = useState(null);
  const [dialogueHistory, setDialogueHistory] = useState([]);
  const [course, setCourse] = useState('');
  const [topic, setTopic] = useState('');
  const [generating, setGenerating] = useState(false);
  const [agentSteps, setAgentSteps] = useState([]);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    const p = sessionStorage.getItem('currentProfile');
    const d = sessionStorage.getItem('dialogueHistory');
    if (p) {
      const profileData = JSON.parse(p);
      setProfile(profileData);
      // 根据画像推断课程和知识点
      inferFromProfile(profileData);
    }
    if (d) setDialogueHistory(JSON.parse(d));
  }, []);

  // 从画像专业推断课程和知识点
  const inferFromProfile = (profileData) => {
    const major = (profileData.major || '').toLowerCase();
    if (major.includes('法律') || major.includes('法学')) {
      setCourse('刑法学'); setTopic('犯罪构成要件');
    } else if (major.includes('医学') || major.includes('临床')) {
      setCourse('病理学'); setTopic('炎症与免疫反应');
    } else if (major.includes('金融') || major.includes('经济')) {
      setCourse('宏观经济学'); setTopic('货币政策工具');
    } else if (major.includes('计算机') || major.includes('软件')) {
      setCourse('数据结构'); setTopic('二叉树遍历算法');
    } else {
      // 通用回退：根据学习目标推断
      const goal = profileData.dimensions?.learningGoal?.details || major;
      if (goal) {
        setCourse(profileData.major ? profileData.major.replace('专业', '') + '核心课程' : '通用学习');
        setTopic('基础知识体系');
      } else {
        setCourse('人工智能导论'); setTopic('机器学习基础');
      }
    }
  };

  const handleGenerate = async () => {
    setGenerating(true);
    setError('');
    setResult(null);
    setAgentSteps(AGENTS.map((a, i) => ({ ...a, status: i === 0 ? 'running' : 'pending', index: i })));
    
    try {
      const res = await api.generate({
        dialogueHistory,
        course,
        topic,
        profileId: profile?.id
      });
      
      if (res.data?.resource) {
        const resource = res.data.resource;
        // 根据 executionSteps 更新 agent 状态
        const steps = (resource.executionSteps || []).map((step, i) => {
          const agentDef = AGENTS.find(a => step.agent.includes(a.name)) || AGENTS[i % AGENTS.length];
          return {
            ...agentDef,
            status: 'success',
            duration: step.duration
          };
        });
        setAgentSteps(steps);
        setResult(resource);
      }
    } catch (err) {
      setError(err.message);
      setAgentSteps([]);
    }
    setGenerating(false);
  };

  return (
    <div>
      <div className="card">
        <h2 className="card-title">⚡ 个性化资源生成</h2>
        {profile ? (
          <div style={{background: '#e3f2fd', padding: '12px 16px', borderRadius: '8px', marginBottom: '16px'}}>
            <strong>当前画像：</strong>{profile.studentName} | {profile.major}
            <span style={{marginLeft: '12px', color: '#1a73e8', cursor: 'pointer', fontSize: '13px'}} onClick={() => navigate('/profile/chat')}>重新对话</span>
          </div>
        ) : (
          <div style={{background: '#fff3e0', padding: '12px 16px', borderRadius: '8px', marginBottom: '16px', fontSize: '14px'}}>
            未检测到学生画像，将使用默认画像生成资源。建议先进行 <span style={{color: '#1a73e8', cursor: 'pointer'}} onClick={() => navigate('/profile/chat')}>画像对话</span>
          </div>
        )}

        <div className="grid-2">
          <div className="form-group">
            <label className="form-label">课程（将根据画像自动填充）</label>
            <input className="input" value={course} onChange={e => setCourse(e.target.value)} placeholder="如：刑法学、病理学、数据结构..." />
          </div>
          <div className="form-group">
            <label className="form-label">知识点（将根据画像自动填充）</label>
            <input className="input" value={topic} onChange={e => setTopic(e.target.value)} placeholder="如：犯罪构成要件、炎症反应、二叉树遍历..." />
          </div>
        </div>

        <button className="btn btn-primary" onClick={handleGenerate} disabled={generating} style={{width: '100%', justifyContent: 'center', padding: '14px'}}>
          {generating ? <><div className="loading-spinner"></div> 智能体协同生成中...</> : '🚀 开始生成学习资源'}
        </button>
      </div>

      {error && <div className="error-msg">❌ {error}</div>}

      {(generating || agentSteps.length > 0) && (
        <div className="card">
          <h3 className="card-title">🤖 多智能体执行进度</h3>
          {agentSteps.map((step, i) => (
            <div key={i} className={`agent-step ${generating && step.status === 'running' ? 'running' : step.status}`}>
              <div className="agent-icon">{step.status === 'success' ? '✓' : step.icon}</div>
              <div className="agent-info">
                <div className="agent-name">{step.label} ({step.name})</div>
                <div className="agent-desc">{step.desc}</div>
              </div>
              {step.duration && <div className="agent-duration">{step.duration}ms</div>}
              {generating && step.status === 'running' && <div className="loading-spinner"></div>}
            </div>
          ))}
        </div>
      )}

      {result && (
        <div className="card">
          <h3 className="card-title">✅ 生成完成</h3>
          <div style={{display: 'flex', gap: '12px', marginBottom: '16px'}}>
            <span className="badge badge-info">课程: {result.course}</span>
            <span className="badge badge-info">知识点: {result.topic}</span>
            <span className="badge badge-info">耗时: {result.totalDuration}ms</span>
            <span className={`badge ${result.review?.passed ? 'badge-success' : 'badge-warning'}`}>
              质量评分: {result.review?.score}
            </span>
          </div>
          
          {result.profile && (
            <div style={{marginBottom: '16px'}}>
              <h4 style={{marginBottom: '8px', color: '#1a237e'}}>学生画像（6维）</h4>
              <div className="grid-2">
                {Object.entries(result.profile.dimensions).map(([key, val]) => (
                  <div key={key} className="profile-dim">
                    <div className="profile-dim-header">
                      <span className="profile-dim-name">{dimensionNames[key] || key}</span>
                      <span className="profile-dim-level">{val.level}</span>
                    </div>
                    <div className="profile-dim-detail">{val.details}</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {result.review && !result.review.passed && result.review.issues.length > 0 && (
            <div style={{background: '#fff3e0', padding: '12px', borderRadius: '8px', marginBottom: '12px'}}>
              <strong>⚠️ 质量检查问题：</strong>
              <ul style={{marginTop: '8px', paddingLeft: '20px'}}>
                {result.review.issues.map((issue, i) => (
                  <li key={i} style={{fontSize: '13px'}}>{issue.message}</li>
                ))}
              </ul>
            </div>
          )}

          <button className="btn btn-primary" onClick={() => navigate(`/resource/${result.id}`)}>
            📖 查看完整资源包
          </button>
        </div>
      )}
    </div>
  );
}

const dimensionNames = {
  knowledgeBase: '知识基础',
  learningGoal: '学习目标',
  cognitiveStyle: '认知风格',
  weakPoints: '薄弱知识点',
  resourcePreference: '资源偏好',
  studyTime: '学习时间'
};

export default Generate;
