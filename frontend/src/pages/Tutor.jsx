import { useState, useRef, useEffect, useCallback } from 'react';
import { api } from '../api/client';

const SUGGEST_QUESTIONS = [
  '什么是信息增益？',
  '决策树有哪些剪枝方法？',
  '什么是梯度下降？',
  'CNN 和 RNN 有什么区别？',
  '什么是过拟合？',
  '量子计算的基本原理是什么？'
];

const DEFAULT_MODEL_KEY = 'mock';
const HISTORY_KEY = 'tutor_chat_history';
const WELCOME_MSG = { role: 'assistant', content: '你好！我是智能辅导助手。你可以在下方选择或导入大模型，也可以点击麦克风直接用语音提问。我会基于知识库为你解答。', sources: [], hasAnswer: true };

// ===== History Storage Helpers =====
function loadHistory() {
  try {
    const raw = localStorage.getItem(HISTORY_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch { return []; }
}

function saveHistory(list) {
  try {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(list));
  } catch (e) { console.error('保存对话历史失败:', e); }
}

function genId() {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
}

function formatTime(ts) {
  const d = new Date(ts);
  const now = new Date();
  const diff = now - d;
  if (diff < 60000) return '刚刚';
  if (diff < 3600000) return `${Math.floor(diff / 60000)} 分钟前`;
  if (d.toDateString() === now.toDateString()) return `今天 ${d.getHours().toString().padStart(2,'0')}:${d.getMinutes().toString().padStart(2,'0')}`;
  const y = new Date(now); y.setDate(y.getDate() - 1);
  if (d.toDateString() === y.toDateString()) return '昨天';
  return `${d.getMonth() + 1}/${d.getDate()} ${d.getHours().toString().padStart(2,'0')}:${d.getMinutes().toString().padStart(2,'0')}`;
}

function Tutor() {
  const [messages, setMessages] = useState([WELCOME_MSG]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [modelKey, setModelKey] = useState(DEFAULT_MODEL_KEY);
  const [models, setModels] = useState({ builtIn: [], custom: [] });
  const [showAddModel, setShowAddModel] = useState(false);
  const [recording, setRecording] = useState(false);
  const [recorderError, setRecorderError] = useState('');
  const messagesEnd = useRef(null);
  const recognitionRef = useRef(null);

  // History state
  const [history, setHistory] = useState([]);
  const [currentSessionId, setCurrentSessionId] = useState(null);
  const [showHistory, setShowHistory] = useState(true);

  useEffect(() => {
    const h = loadHistory();
    setHistory(h);
    // Auto-load the most recent session if exists
    if (h.length > 0) {
      const latest = h[0];
      setCurrentSessionId(latest.id);
      setMessages(latest.messages);
    }
  }, []);

  // Auto-save messages to history when they change (skip initial welcome-only state)
  const persistMessages = useCallback((msgs, sessionId) => {
    const sid = sessionId || currentSessionId;
    // Only save if there's at least one user message
    const hasUserMsg = msgs.some(m => m.role === 'user');
    if (!hasUserMsg) return;

    const h = loadHistory();
    let session = h.find(s => s.id === sid);

    if (session) {
      session.messages = msgs;
      session.updatedAt = Date.now();
      // Update title if it's still default
      const firstUserMsg = msgs.find(m => m.role === 'user');
      if (firstUserMsg && (session.title === '新对话' || !session.title)) {
        session.title = firstUserMsg.content.slice(0, 30);
      }
    } else {
      const firstUserMsg = msgs.find(m => m.role === 'user');
      const newSession = {
        id: genId(),
        title: firstUserMsg ? firstUserMsg.content.slice(0, 30) : '新对话',
        messages: msgs,
        createdAt: Date.now(),
        updatedAt: Date.now()
      };
      h.unshift(newSession);
      setCurrentSessionId(newSession.id);
      session = newSession;
    }

    // Sort by updatedAt desc, keep max 50 sessions
    h.sort((a, b) => b.updatedAt - a.updatedAt);
    if (h.length > 50) h.length = 50;
    saveHistory(h);
    setHistory([...h]);
  }, [currentSessionId]);

  useEffect(() => {
    messagesEnd.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    loadModels();
  }, []);

  const loadModels = async () => {
    try {
      const res = await api.getModels();
      setModels(res.data);
    } catch (err) {
      console.error('加载模型列表失败:', err);
    }
  };

  const ask = async (question) => {
    const q = question || input.trim();
    if (!q || loading) return;

    const userMsg = { role: 'user', content: q };
    const newMsgs = [...messages, userMsg];
    setMessages(newMsgs);
    setInput('');
    setLoading(true);

    try {
      const profile = JSON.parse(sessionStorage.getItem('currentProfile') || '{}');
      const res = await api.askTutor(q, profile.id, modelKey);
      const assistantMsg = {
        role: 'assistant',
        content: res.data.answer,
        sources: res.data.sources || [],
        hasAnswer: res.data.hasAnswer,
        latency: res.data.latency,
        model: res.data.model,
        modelId: res.data.modelId
      };
      const finalMsgs = [...newMsgs, assistantMsg];
      setMessages(finalMsgs);
      persistMessages(finalMsgs);
    } catch (err) {
      const errMsg = { role: 'assistant', content: '抱歉，出了点问题：' + err.message, sources: [], hasAnswer: false };
      const finalMsgs = [...newMsgs, errMsg];
      setMessages(finalMsgs);
      persistMessages(finalMsgs);
    }
    setLoading(false);
  };

  const startNewChat = () => {
    setMessages([WELCOME_MSG]);
    setCurrentSessionId(null);
  };

  const loadSession = (session) => {
    setCurrentSessionId(session.id);
    setMessages(session.messages);
  };

  const deleteSession = (e, sessionId) => {
    e.stopPropagation();
    const h = loadHistory().filter(s => s.id !== sessionId);
    saveHistory(h);
    setHistory(h);
    if (currentSessionId === sessionId) {
      setCurrentSessionId(null);
      setMessages([WELCOME_MSG]);
    }
  };

  const clearAllHistory = () => {
    if (!confirm('确定要清空所有对话历史吗？此操作不可撤销。')) return;
    localStorage.removeItem(HISTORY_KEY);
    setHistory([]);
    setCurrentSessionId(null);
    setMessages([WELCOME_MSG]);
  };

  const startVoiceInput = () => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      setRecorderError('当前浏览器不支持语音识别，请使用 Chrome/Edge 浏览器。');
      return;
    }

    setRecorderError('');
    const recognition = new SpeechRecognition();
    recognition.lang = 'zh-CN';
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;

    recognition.onstart = () => setRecording(true);

    recognition.onresult = (event) => {
      const transcript = event.results[0][0].transcript;
      setInput(prev => prev ? prev + ' ' + transcript : transcript);
    };

    recognition.onerror = (event) => {
      setRecording(false);
      if (event.error !== 'aborted') {
        setRecorderError('语音识别失败：' + event.error);
      }
    };

    recognition.onend = () => setRecording(false);

    recognitionRef.current = recognition;
    recognition.start();
  };

  const stopVoiceInput = () => {
    recognitionRef.current?.stop();
    setRecording(false);
  };

  const allModels = [
    ...models.builtIn.map(m => ({ ...m, label: `[内置] ${m.name}` })),
    ...models.custom.map(m => ({ ...m, label: `[自定义] ${m.name}` }))
  ];

  return (
    <div>
      <div className="card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '12px' }}>
          <div>
            <h2 className="card-title" style={{ marginBottom: '8px' }}>❓ 智能辅导</h2>
            <p style={{ color: '#666', fontSize: '14px' }}>
              基于 RAG 知识库的防幻觉问答系统。可切换或导入任意兼容 OpenAI 接口的大模型，也支持语音输入。
            </p>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <button className="btn btn-secondary btn-sm" onClick={() => setShowHistory(!showHistory)}>
              {showHistory ? '📋 隐藏历史' : '📋 显示历史'}
            </button>
            <select
              className="select"
              style={{ width: 'auto', minWidth: '180px' }}
              value={modelKey}
              onChange={e => setModelKey(e.target.value)}
              disabled={loading}
            >
              {allModels.map(m => (
                <option key={m.id} value={m.id}>{m.label}</option>
              ))}
            </select>
            <button className="btn btn-secondary btn-sm" onClick={() => setShowAddModel(true)} disabled={loading}>
              ＋ 导入模型
            </button>
          </div>
        </div>
      </div>

      <div style={{ display: 'flex', gap: '0' }}>
        {/* History Sidebar */}
        {showHistory && (
          <div className="chat-history-sidebar">
            <div className="chat-history-header">
              <span style={{ fontSize: '14px', fontWeight: 600 }}>对话历史</span>
              <button className="btn btn-primary btn-sm" onClick={startNewChat} style={{ whiteSpace: 'nowrap' }}>
                ＋ 新对话
              </button>
            </div>
            <div className="chat-history-list">
              {history.length === 0 ? (
                <div className="chat-history-empty">
                  <div style={{ fontSize: '32px', marginBottom: '8px' }}>💬</div>
                  <div style={{ fontSize: '13px', color: '#999' }}>暂无对话记录</div>
                  <div style={{ fontSize: '12px', color: '#bbb', marginTop: '4px' }}>开始提问后会自动保存</div>
                </div>
              ) : (
                history.map(s => (
                  <div
                    key={s.id}
                    className={`chat-history-item ${currentSessionId === s.id ? 'active' : ''}`}
                    onClick={() => loadSession(s)}
                  >
                    <div className="chat-history-item-title">{s.title || '新对话'}</div>
                    <div className="chat-history-item-meta">
                      <span>{formatTime(s.updatedAt || s.createdAt)}</span>
                      <span>{s.messages.filter(m => m.role === 'user').length} 问</span>
                    </div>
                    <button
                      className="chat-history-item-delete"
                      onClick={(e) => deleteSession(e, s.id)}
                      title="删除"
                    >
                      ✕
                    </button>
                  </div>
                ))
              )}
            </div>
            {history.length > 0 && (
              <div className="chat-history-footer">
                <button className="btn btn-sm btn-danger" style={{ width: '100%' }} onClick={clearAllHistory}>
                  清空全部
                </button>
              </div>
            )}
          </div>
        )}

        {/* Chat Area */}
        <div className="card" style={{ padding: 0, flex: 1, minWidth: 0 }}>
          <div className="chat-container">
            <div className="chat-messages">
              {messages.map((msg, i) => (
                <div key={i} className={`chat-msg ${msg.role}`}>
                  <div className="chat-bubble">
                    <div>{msg.content}</div>
                    {msg.role === 'assistant' && msg.sources && msg.sources.length > 0 && (
                      <div style={{ marginTop: '10px', paddingTop: '8px', borderTop: '1px solid rgba(0,0,0,0.1)' }}>
                        <div style={{ fontSize: '11px', fontWeight: 600, color: '#e65100', marginBottom: '4px' }}>📌 引用来源：</div>
                        {msg.sources.map((src, j) => (
                          <div key={j} style={{ fontSize: '11px', color: '#666', margin: '2px 0' }}>
                            📄 {src.document} → {src.section}
                          </div>
                        ))}
                      </div>
                    )}
                    {msg.role === 'assistant' && msg.hasAnswer === false && (
                      <div style={{ marginTop: '8px', padding: '6px 10px', background: '#fce4ec', borderRadius: '6px', fontSize: '12px', color: '#c62828' }}>
                        ⚠️ 防幻觉提示：当前课程知识库资料不足，无法确认该问题的答案
                      </div>
                    )}
                    {(msg.latency || msg.model) && (
                      <div style={{ marginTop: '6px', fontSize: '11px', color: '#999', display: 'flex', gap: '12px' }}>
                        {msg.latency && <span>⏱️ {msg.latency}ms</span>}
                        {msg.model && <span style={{ color: '#1976d2' }}>🤖 {msg.model}</span>}
                      </div>
                    )}
                  </div>
                </div>
              ))}
              {loading && (
                <div className="chat-msg assistant">
                  <div className="chat-bubble"><div className="loading-spinner"></div></div>
                </div>
              )}
              <div ref={messagesEnd} />
            </div>

            <div style={{ padding: '8px 16px', display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
              {SUGGEST_QUESTIONS.map(q => (
                <button key={q} className="btn btn-sm btn-secondary" onClick={() => ask(q)} disabled={loading}>
                  {q}
                </button>
              ))}
            </div>

            {recorderError && (
              <div className="error-msg" style={{ margin: '0 16px 8px' }}>{recorderError}</div>
            )}

            <div className="chat-input-area">
              <button
                className={`btn ${recording ? 'btn-danger' : 'btn-secondary'}`}
                onClick={recording ? stopVoiceInput : startVoiceInput}
                title={recording ? '停止录音' : '语音输入'}
                style={{ padding: '10px 14px' }}
              >
                {recording ? '⏹ 停止' : '🎤 语音'}
              </button>
              <input
                className="input"
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && ask()}
                placeholder={recording ? '正在聆听，请说话...' : '输入你的问题...'}
                disabled={loading}
              />
              <button className="btn btn-primary" onClick={() => ask()} disabled={loading || !input.trim()}>
                提问
              </button>
            </div>
          </div>
        </div>
      </div>

      {showAddModel && (
        <AddModelModal
          onClose={() => setShowAddModel(false)}
          onSaved={() => { setShowAddModel(false); loadModels(); }}
        />
      )}
    </div>
  );
}

function AddModelModal({ onClose, onSaved }) {
  const [form, setForm] = useState({ id: '', name: '', baseUrl: '', apiKey: '', model: '' });
  const [saving, setSaving] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setSaving(true);
    try {
      await api.addModel(form);
      onSaved();
    } catch (err) {
      alert('导入模型失败：' + err.message);
    }
    setSaving(false);
  };

  return (
    <div style={{
      position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.4)',
      display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1000
    }} onClick={onClose}>
      <div className="card" style={{ width: '480px', maxWidth: '90%' }} onClick={e => e.stopPropagation()}>
        <h3 className="card-title">导入自定义模型</h3>
        <p style={{ fontSize: '13px', color: '#666', marginBottom: '16px' }}>
          支持任意兼容 OpenAI Chat Completions 接口的大模型。API Key 仅保存在服务端本地，不会被前端直接读取。
        </p>
        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label className="form-label">模型 ID（英文，用于选择）</label>
            <input className="input" value={form.id} onChange={e => setForm({ ...form, id: e.target.value })} placeholder="例如：qwen-local" required />
          </div>
          <div className="form-group">
            <label className="form-label">显示名称</label>
            <input className="input" value={form.name} onChange={e => setForm({ ...form, name: e.target.value })} placeholder="例如：本地 Qwen" required />
          </div>
          <div className="form-group">
            <label className="form-label">Base URL</label>
            <input className="input" value={form.baseUrl} onChange={e => setForm({ ...form, baseUrl: e.target.value })} placeholder="例如：http://localhost:11434/v1 或 https://api.xxx.com/v1" required />
          </div>
          <div className="form-group">
            <label className="form-label">API Key</label>
            <input className="input" type="password" value={form.apiKey} onChange={e => setForm({ ...form, apiKey: e.target.value })} placeholder="sk-..." required />
          </div>
          <div className="form-group">
            <label className="form-label">Model 名称</label>
            <input className="input" value={form.model} onChange={e => setForm({ ...form, model: e.target.value })} placeholder="例如：qwen2.5:7b 或 gpt-4o-mini" required />
          </div>
          <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '10px', marginTop: '20px' }}>
            <button type="button" className="btn btn-secondary" onClick={onClose}>取消</button>
            <button type="submit" className="btn btn-primary" disabled={saving}>{saving ? '保存中...' : '导入'}</button>
          </div>
        </form>
      </div>
    </div>
  );
}

export default Tutor;
