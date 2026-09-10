import { useState, useEffect } from 'react';
import { api } from '../api/client';

function Analytics() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.getAnalytics().then(res => {
      setData(res.data?.analytics);
      setLoading(false);
    }).catch(() => setLoading(false));
  }, []);

  if (loading) return <div className="loading-overlay"><div className="loading-spinner"></div><span className="loading-text">加载分析数据中...</span></div>;
  if (!data) return <div className="error-msg">加载失败</div>;

  return (
    <div>
      <div className="card">
        <h2 className="card-title">📊 学习效果与系统分析</h2>
      </div>

      <div className="grid-4">
        <div className="card" style={{textAlign: 'center'}}>
          <div style={{fontSize: '32px', fontWeight: 700, color: '#1a73e8'}}>{data.totalProfiles}</div>
          <div style={{fontSize: '13px', color: '#666'}}>学生画像</div>
        </div>
        <div className="card" style={{textAlign: 'center'}}>
          <div style={{fontSize: '32px', fontWeight: 700, color: '#34a853'}}>{data.totalResources}</div>
          <div style={{fontSize: '13px', color: '#666'}}>生成资源</div>
        </div>
        <div className="card" style={{textAlign: 'center'}}>
          <div style={{fontSize: '32px', fontWeight: 700, color: '#ff9800'}}>{data.totalLogs}</div>
          <div style={{fontSize: '13px', color: '#666'}}>系统日志</div>
        </div>
        <div className="card" style={{textAlign: 'center'}}>
          <div style={{fontSize: '32px', fontWeight: 700, color: '#9c27b0'}}>{data.totalDocs}</div>
          <div style={{fontSize: '13px', color: '#666'}}>知识库文档</div>
        </div>
      </div>

      <div className="grid-2">
        <div className="card">
          <h3 style={{marginBottom: '12px', color: '#1a237e'}}>日志类型分布</h3>
          {Object.keys(data.logsByType).length === 0 ? (
            <div className="empty-state"><p>暂无日志数据</p></div>
          ) : (
            Object.entries(data.logsByType).map(([type, count]) => {
              const maxCount = Math.max(...Object.values(data.logsByType));
              const percent = (count / maxCount) * 100;
              const labels = {
                profile_chat: '画像对话',
                qa: '智能辅导',
                generate: '资源生成'
              };
              return (
                <div key={type} style={{marginBottom: '12px'}}>
                  <div style={{display: 'flex', justifyContent: 'space-between', fontSize: '13px', marginBottom: '4px'}}>
                    <span>{labels[type] || type}</span>
                    <span style={{fontWeight: 600}}>{count}</span>
                  </div>
                  <div style={{height: '8px', background: '#f0f0f0', borderRadius: '4px', overflow: 'hidden'}}>
                    <div style={{height: '100%', width: `${percent}%`, background: '#1a73e8', borderRadius: '4px'}}></div>
                  </div>
                </div>
              );
            })
          )}
          <div style={{marginTop: '16px', padding: '12px', background: '#e3f2fd', borderRadius: '8px', fontSize: '13px'}}>
            <strong>平均响应时间：</strong>{data.avgLatency}ms
          </div>
        </div>

        <div className="card">
          <h3 style={{marginBottom: '12px', color: '#1a237e'}}>最近生成记录</h3>
          {data.recentResources.length === 0 ? (
            <div className="empty-state"><p>暂无生成记录</p></div>
          ) : (
            <table className="data-table">
              <thead>
                <tr>
                  <th>课程</th>
                  <th>知识点</th>
                  <th>评分</th>
                  <th>时间</th>
                </tr>
              </thead>
              <tbody>
                {data.recentResources.slice(0, 5).map((r, i) => (
                  <tr key={i}>
                    <td>{r.course}</td>
                    <td>{r.topic}</td>
                    <td>{r.review?.score || '-'}</td>
                    <td style={{fontSize: '12px'}}>{new Date(r.createdAt).toLocaleString('zh-CN')}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>

      <div className="card">
        <h3 style={{marginBottom: '12px', color: '#1a237e'}}>最近系统日志</h3>
        {data.recentLogs.length === 0 ? (
          <div className="empty-state"><p>暂无日志</p></div>
        ) : (
          <table className="data-table">
            <thead>
              <tr>
                <th>类型</th>
                <th>输入摘要</th>
                <th>耗时</th>
                <th>时间</th>
              </tr>
            </thead>
            <tbody>
              {data.recentLogs.slice(0, 10).map((log, i) => (
                <tr key={i}>
                  <td>
                    <span className={`badge ${log.type === 'qa' ? 'badge-info' : log.type === 'generate' ? 'badge-success' : 'badge-warning'}`}>
                      {log.type === 'qa' ? '问答' : log.type === 'generate' ? '生成' : log.type === 'profile_chat' ? '画像' : log.type}
                    </span>
                  </td>
                  <td style={{fontSize: '12px', maxWidth: '300px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap'}}>
                    {log.input || log.topic || '-'}
                  </td>
                  <td>{log.latency ? `${log.latency}ms` : log.totalDuration ? `${log.totalDuration}ms` : '-'}</td>
                  <td style={{fontSize: '12px'}}>{new Date(log.timestamp).toLocaleString('zh-CN')}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}

export default Analytics;
