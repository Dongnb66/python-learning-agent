const BASE = '/api';
const REQUEST_TIMEOUT = 120000; // 120 秒超时（多智能体 DeepSeek 链路需要较长时间）

async function request(url, options = {}) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT);

  try {
    const res = await fetch(`${BASE}${url}`, {
      headers: { 'Content-Type': 'application/json' },
      signal: controller.signal,
      ...options
    });
    clearTimeout(timeoutId);

    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: `HTTP ${res.status}` }));
      throw new Error(err.error || `HTTP ${res.status}`);
    }

    return res.json();
  } catch (err) {
    clearTimeout(timeoutId);
    if (err.name === 'AbortError') {
      throw new Error('请求超时，请检查后端服务是否正常运行');
    }
    throw err;
  }
}

export const api = {
  // Profile
  profileChat: (messages) => request('/profile/chat', {
    method: 'POST', body: JSON.stringify({ messages })
  }),
  generateProfile: (dialogueHistory) => request('/profile/generate', {
    method: 'POST', body: JSON.stringify({ dialogueHistory })
  }),
  getProfiles: () => request('/profile'),
  getProfile: (id) => request(`/profile/${id}`),

  // Generate
  generate: (data) => request('/generate', {
    method: 'POST', body: JSON.stringify(data)
  }),

  // Resource
  getResources: () => request('/resource'),
  getResource: (id) => request(`/resource/${id}`),

  // Path
  getPath: (profileId) => request(`/path/${profileId}`),

  // Tutor
  askTutor: (question, profileId, modelKey) => request('/tutor/ask', {
    method: 'POST', body: JSON.stringify({ question, profileId, modelKey })
  }),

  // Models
  getModels: () => request('/models'),
  addModel: (model) => request('/models', {
    method: 'POST', body: JSON.stringify(model)
  }),
  deleteModel: (id) => request(`/models/${id}`, { method: 'DELETE' }),

  // Knowledge
  getKnowledge: () => request('/knowledge'),
  getKnowledgeDoc: (id) => request(`/knowledge/${id}`),
  uploadKnowledge: (title, content) => request('/knowledge/upload', {
    method: 'POST', body: JSON.stringify({ title, content })
  }),
  updateKnowledge: (id, title, content) => request(`/knowledge/${id}`, {
    method: 'PUT', body: JSON.stringify({ title, content })
  }),

  // Analytics
  getAnalytics: () => request('/analytics'),

  // Efficacy (前后测效果实证)
  getEfficacyQuestions: (topic) => request(`/efficacy/questions?topic=${encodeURIComponent(topic || '机器学习')}`),
  submitEfficacy: (data) => request('/efficacy/submit', {
    method: 'POST', body: JSON.stringify(data)
  }),
  getEfficacySummary: () => request('/efficacy/summary'),

  // Health
  health: () => request('/health')
};
