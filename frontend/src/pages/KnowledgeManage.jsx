import { useState, useEffect } from 'react';
import { api } from '../api/client';

function KnowledgeManage() {
  const [docs, setDocs] = useState([]);
  const [selectedDoc, setSelectedDoc] = useState(null);
  const [loading, setLoading] = useState(true);
  const [editMode, setEditMode] = useState(false);
  const [editTitle, setEditTitle] = useState('');
  const [editContent, setEditContent] = useState('');
  const [uploadMode, setUploadMode] = useState(false);
  const [newTitle, setNewTitle] = useState('');
  const [newContent, setNewContent] = useState('');
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    loadDocs();
  }, []);

  const loadDocs = async () => {
    setLoading(true);
    try {
      const res = await api.getKnowledge();
      setDocs(res.data?.documents || []);
    } catch (err) {
      console.error(err);
    }
    setLoading(false);
  };

  const selectDoc = async (id) => {
    try {
      const res = await api.getKnowledgeDoc(id);
      setSelectedDoc(res.data?.document);
      setEditMode(false);
    } catch (err) {
      console.error(err);
    }
  };

  const startEdit = () => {
    setEditTitle(selectedDoc.title);
    setEditContent(selectedDoc.content);
    setEditMode(true);
  };

  const saveEdit = async () => {
    setSaving(true);
    try {
      await api.updateKnowledge(selectedDoc.id, editTitle, editContent);
      setEditMode(false);
      await loadDocs();
      await selectDoc(selectedDoc.id);
    } catch (err) {
      alert('保存失败：' + err.message);
    }
    setSaving(false);
  };

  const uploadDoc = async () => {
    if (!newTitle.trim() || !newContent.trim()) {
      alert('标题和内容不能为空');
      return;
    }
    setSaving(true);
    try {
      await api.uploadKnowledge(newTitle, newContent);
      setNewTitle('');
      setNewContent('');
      setUploadMode(false);
      await loadDocs();
    } catch (err) {
      alert('上传失败：' + err.message);
    }
    setSaving(false);
  };

  return (
    <div>
      <div className="card">
        <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center'}}>
          <h2 className="card-title" style={{marginBottom: 0}}>📚 课程知识库管理</h2>
          <button className="btn btn-primary" onClick={() => setUploadMode(!uploadMode)}>
            {uploadMode ? '取消' : '➕ 新建文档'}
          </button>
        </div>
      </div>

      {uploadMode && (
        <div className="card">
          <h3 style={{marginBottom: '16px', color: '#1a237e'}}>新建知识库文档</h3>
          <div className="form-group">
            <label className="form-label">文档标题</label>
            <input className="input" value={newTitle} onChange={e => setNewTitle(e.target.value)} placeholder="如：深度学习基础" />
          </div>
          <div className="form-group">
            <label className="form-label">文档内容（Markdown 格式）</label>
            <textarea className="textarea" rows="12" value={newContent} onChange={e => setNewContent(e.target.value)} placeholder="# 标题&#10;&#10;正文内容..." />
          </div>
          <button className="btn btn-success" onClick={uploadDoc} disabled={saving}>
            {saving ? <><div className="loading-spinner"></div> 保存中</> : '✓ 保存文档'}
          </button>
        </div>
      )}

      <div className="grid-2">
        <div className="card">
          <h3 style={{marginBottom: '12px', color: '#1a237e'}}>文档列表</h3>
          {loading ? (
            <div className="loading-overlay"><div className="loading-spinner"></div></div>
          ) : docs.length === 0 ? (
            <div className="empty-state"><p>暂无文档</p></div>
          ) : (
            docs.map(doc => (
              <div
                key={doc.id}
                className="resource-card"
                style={{cursor: 'pointer', borderLeft: selectedDoc?.id === doc.id ? '3px solid #1a73e8' : '1px solid #e8eaed'}}
                onClick={() => selectDoc(doc.id)}
              >
                <div className="resource-card-header">
                  <span className="resource-card-icon">📄</span>
                  <span className="resource-card-title">{doc.title}</span>
                </div>
                <p style={{fontSize: '12px', color: '#999'}}>
                  {doc.sectionCount} 个章节 | {doc.contentLength} 字符
                </p>
                <p style={{fontSize: '12px', color: '#666', marginTop: '4px'}}>{doc.preview}...</p>
              </div>
            ))
          )}
        </div>

        <div className="card">
          {selectedDoc ? (
            editMode ? (
              <div>
                <h3 style={{marginBottom: '12px', color: '#1a237e'}}>编辑文档</h3>
                <div className="form-group">
                  <label className="form-label">标题</label>
                  <input className="input" value={editTitle} onChange={e => setEditTitle(e.target.value)} />
                </div>
                <div className="form-group">
                  <label className="form-label">内容</label>
                  <textarea className="textarea" rows="20" value={editContent} onChange={e => setEditContent(e.target.value)} />
                </div>
                <div style={{display: 'flex', gap: '8px'}}>
                  <button className="btn btn-success" onClick={saveEdit} disabled={saving}>
                    {saving ? '保存中...' : '✓ 保存'}
                  </button>
                  <button className="btn btn-secondary" onClick={() => setEditMode(false)}>取消</button>
                </div>
              </div>
            ) : (
              <div>
                <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px'}}>
                  <h3 style={{color: '#1a237e'}}>{selectedDoc.title}</h3>
                  <button className="btn btn-sm btn-secondary" onClick={startEdit}>✏️ 编辑</button>
                </div>
                <div style={{background: '#f8f9fa', padding: '16px', borderRadius: '8px', maxHeight: '500px', overflowY: 'auto'}}>
                  <pre style={{whiteSpace: 'pre-wrap', fontFamily: 'inherit', fontSize: '13px', lineHeight: '1.6'}}>{selectedDoc.content}</pre>
                </div>
              </div>
            )
          ) : (
            <div className="empty-state">
              <div className="empty-state-icon">📄</div>
              <p>选择左侧文档查看详情</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default KnowledgeManage;
