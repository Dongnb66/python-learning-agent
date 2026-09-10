import { useState, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../api/client';

const SUGGESTIONS = [
  '我是法律专业的学生，想提升案例分析能力',
  '我是医学专业的学生，希望掌握临床思维',
  '我喜欢通过案例和实践来学习',
  '我偏好视频讲解和图文资料',
  '我每周能投入10-15小时学习',
  '我在理论理解方面比较薄弱'
];

function ProfileChat() {
  const navigate = useNavigate();
  const [messages, setMessages] = useState([
    { role: 'assistant', content: '你好！我是学习画像助手，会通过几轮对话了解你的学习情况，为你构建专属学习画像。\n\n先从最基本的开始——可以告诉我你的专业是什么吗？' }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [generating, setGenerating] = useState(false);
  const messagesEnd = useRef(null);

  useEffect(() => {
    messagesEnd.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const send = async (text) => {
    const content = text || input.trim();
    if (!content || loading) return;
    
    const newMessages = [...messages, { role: 'user', content }];
    setMessages(newMessages);
    setInput('');
    setLoading(true);
    
    try {
      const res = await api.profileChat(newMessages);
      setMessages([...newMessages, { role: 'assistant', content: res.data.reply }]);
    } catch (err) {
      setMessages([...newMessages, { role: 'assistant', content: '抱歉，出了点问题：' + err.message }]);
    }
    setLoading(false);
  };

  const generateProfile = async () => {
    setGenerating(true);
    try {
      const res = await api.generateProfile(messages);
      if (res.data?.profile) {
        // 存储画像到 sessionStorage 供后续使用
        sessionStorage.setItem('currentProfile', JSON.stringify(res.data.profile));
        sessionStorage.setItem('dialogueHistory', JSON.stringify(messages));
        navigate('/generate');
      }
    } catch (err) {
      alert('生成画像失败：' + err.message);
    }
    setGenerating(false);
  };

  return (
    <div>
      <div className="card">
        <h2 className="card-title">💬 对话式学习画像构建</h2>
        <p style={{color: '#666', fontSize: '14px', marginBottom: '16px'}}>
          通过自然语言对话，系统自动提取 6 个维度的学习画像：知识基础、学习目标、认知风格、薄弱点、资源偏好、学习时间
        </p>
      </div>

      <div className="card" style={{padding: 0}}>
        <div className="chat-container">
          <div className="chat-messages">
            {messages.map((msg, i) => (
              <div key={i} className={`chat-msg ${msg.role}`}>
                <div className="chat-bubble">{msg.content}</div>
              </div>
            ))}
            {loading && (
              <div className="chat-msg assistant">
                <div className="chat-bubble"><div className="loading-spinner"></div></div>
              </div>
            )}
            <div ref={messagesEnd} />
          </div>

          <div style={{padding: '8px 16px', display: 'flex', gap: '6px', flexWrap: 'wrap'}}>
            {SUGGESTIONS.map(s => (
              <button key={s} className="btn btn-sm btn-secondary" onClick={() => send(s)} disabled={loading}>
                {s}
              </button>
            ))}
          </div>

          <div className="chat-input-area">
            <input
              className="input"
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && send()}
              placeholder="输入你的回答..."
              disabled={loading}
            />
            <button className="btn btn-primary" onClick={() => send()} disabled={loading || !input.trim()}>
              发送
            </button>
            <button className="btn btn-success" onClick={generateProfile} disabled={generating || messages.length < 6}>
              {generating ? <><div className="loading-spinner"></div> 生成中</> : '✓ 生成画像'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default ProfileChat;
