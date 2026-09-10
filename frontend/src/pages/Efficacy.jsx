import { useState, useEffect } from 'react';
import { api } from '../api/client';

const TOPICS = ['机器学习', '决策树', '神经网络'];

function Bar({ label, value, color }) {
  return (
    <div style={{ marginBottom: '8px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', marginBottom: '3px' }}>
        <span>{label}</span>
        <span style={{ fontWeight: 600 }}>{value}</span>
      </div>
      <div style={{ height: '14px', background: '#f0f0f0', borderRadius: '7px', overflow: 'hidden' }}>
        <div style={{ height: '100%', width: `${value}%`, background: color, borderRadius: '7px', transition: 'width .6s' }} />
      </div>
    </div>
  );
}

function Efficacy() {
  const [summary, setSummary] = useState(null);
  const [topic, setTopic] = useState('机器学习');
  const [studentName, setStudentName] = useState('');
  const [studentId, setStudentId] = useState('');
  const [phase, setPhase] = useState('pre');
  const [questions, setQuestions] = useState([]);
  const [answers, setAnswers] = useState({});
  const [loadingQ, setLoadingQ] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [result, setResult] = useState(null);

  const loadSummary = () => {
    api.getEfficacySummary().then(res => setSummary(res.data)).catch(() => setSummary(null));
  };

  useEffect(() => { loadSummary(); }, []);

  const loadQuestions = (t) => {
    setLoadingQ(true);
    setQuestions([]);
    setAnswers({});
    setResult(null);
    api.getEfficacyQuestions(t).then(res => {
      setQuestions(res.data.questions || []);
      setLoadingQ(false);
    }).catch(() => setLoadingQ(false));
  };

  useEffect(() => { loadQuestions(topic); }, []);

  const onTopicChange = (t) => {
    setTopic(t);
    loadQuestions(t);
  };

  const choose = (qid, idx) => setAnswers(prev => ({ ...prev, [qid]: idx }));

  const submit = async () => {
    if (Object.keys(answers).length < questions.length) {
      alert('请答完所有题目再提交');
      return;
    }
    setSubmitting(true);
    try {
      const res = await api.submitEfficacy({
        studentId: studentId || `stu-${Date.now()}`,
        studentName: studentName || '同学',
        topic, phase, answers
      });
      setResult(res.data);
      loadSummary();
    } catch (e) {
      alert('提交失败：' + e.message);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div>
      <div className="card">
        <h2 className="card-title">📈 学习效果实证</h2>
        <p style={{ fontSize: '13px', color: '#666', marginTop: '6px' }}>
          学生使用系统前后分别完成知识点诊断测验，用数据量化「系统是否真的帮学生提升了成绩」——这是我们区别于普通资源生成工具的核心证据。
        </p>
      </div>

      {/* 汇总看板 */}
      {summary && (
        <div className="grid-4" style={{ marginTop: '16px' }}>
          <div className="card" style={{ textAlign: 'center' }}>
            <div style={{ fontSize: '30px', fontWeight: 700, color: '#1a73e8' }}>{summary.totalStudents}</div>
            <div style={{ fontSize: '13px', color: '#666' }}>参与实证学生</div>
          </div>
          <div className="card" style={{ textAlign: 'center' }}>
            <div style={{ fontSize: '30px', fontWeight: 700, color: '#ff9800' }}>{summary.avgPre}</div>
            <div style={{ fontSize: '13px', color: '#666' }}>学前平均</div>
          </div>
          <div className="card" style={{ textAlign: 'center' }}>
            <div style={{ fontSize: '30px', fontWeight: 700, color: '#34a853' }}>{summary.avgPost}</div>
            <div style={{ fontSize: '13px', color: '#666' }}>学后平均</div>
          </div>
          <div className="card" style={{ textAlign: 'center' }}>
            <div style={{ fontSize: '30px', fontWeight: 700, color: '#e53935' }}>+{summary.avgImprovement}</div>
            <div style={{ fontSize: '13px', color: '#666' }}>平均提升</div>
          </div>
        </div>
      )}

      {/* 各知识点对比 */}
      {summary && summary.perTopic && summary.perTopic.length > 0 && (
        <div className="card" style={{ marginTop: '16px' }}>
          <h3 style={{ marginBottom: '14px', color: '#1a237e' }}>各知识点 学前 vs 学后 平均分</h3>
          {summary.perTopic.map(t => (
            <div key={t.topic} style={{ marginBottom: '18px' }}>
              <div style={{ fontWeight: 600, marginBottom: '6px' }}>{t.topic}　<span style={{ color: '#e53935' }}>↑ {t.improvement} 分</span></div>
              <Bar label="学前" value={t.avgPre} color="#ffb74d" />
              <Bar label="学后" value={t.avgPost} color="#66bb6a" />
            </div>
          ))}
        </div>
      )}

      {/* 做测试 */}
      <div className="card" style={{ marginTop: '16px' }}>
        <h3 style={{ marginBottom: '12px', color: '#1a237e' }}>① 完成一次诊断测验</h3>
        <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap', alignItems: 'center', marginBottom: '14px' }}>
          <select value={topic} onChange={e => onTopicChange(e.target.value)} style={selStyle}>
            {TOPICS.map(t => <option key={t} value={t}>{t}</option>)}
          </select>
          <input placeholder="你的姓名（选填）" value={studentName} onChange={e => setStudentName(e.target.value)} style={inpStyle} />
          <div style={{ display: 'flex', gap: '6px' }}>
            <button onClick={() => setPhase('pre')} style={phase === 'pre' ? phaseOn : phaseOff}>学前测</button>
            <button onClick={() => setPhase('post')} style={phase === 'post' ? phaseOn : phaseOff}>学后测</button>
          </div>
        </div>

        {loadingQ && <div className="loading-text">加载题目...</div>}
        {questions.map((q, i) => (
          <div key={q.id} style={{ marginBottom: '14px', padding: '12px', background: '#fafafa', borderRadius: '8px' }}>
            <div style={{ fontWeight: 600, marginBottom: '8px' }}>{i + 1}. {q.question} <span style={{ fontSize: '11px', color: '#999' }}>[{q.knowledgePoint}]</span></div>
            {q.options.map((opt, idx) => (
              <label key={idx} style={{ display: 'block', padding: '4px 8px', cursor: 'pointer', borderRadius: '6px', background: answers[q.id] === idx ? '#e3f2fd' : 'transparent' }}>
                <input type="radio" name={q.id} checked={answers[q.id] === idx} onChange={() => choose(q.id, idx)} style={{ marginRight: '8px' }} />
                {opt}
              </label>
            ))}
          </div>
        ))}

        {questions.length > 0 && (
          <button onClick={submit} disabled={submitting} style={btnStyle}>
            {submitting ? '判分中...' : `提交${phase === 'pre' ? '学前测' : '学后测'}`}
          </button>
        )}

        {result && (
          <div style={{ marginTop: '16px', padding: '16px', background: '#e8f5e9', borderRadius: '8px' }}>
            <div style={{ fontSize: '20px', fontWeight: 700, color: '#2e7d32' }}>
              本次得分：{result.score} 分（{result.correct}/{result.total}）
            </div>
            {result.improvement !== null && (
              <div style={{ marginTop: '6px', fontSize: '15px', color: '#e53935', fontWeight: 600 }}>
                相比另一阶段，提升 {result.improvement > 0 ? '+' : ''}{result.improvement} 分
              </div>
            )}
            <div style={{ marginTop: '10px' }}>
              {result.detail.map(d => (
                <div key={d.id} style={{ fontSize: '12px', marginBottom: '4px', color: d.correct ? '#2e7d32' : '#c62828' }}>
                  {d.correct ? '✓' : '✗'} {d.knowledgePoint}：{d.explanation}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* 样例学生 */}
      {summary && summary.sampleStudents && summary.sampleStudents.length > 0 && (
        <div className="card" style={{ marginTop: '16px' }}>
          <h3 style={{ marginBottom: '12px', color: '#1a237e' }}>实证样例（部分学生）</h3>
          <table className="data-table">
            <thead>
              <tr><th>学生</th><th>知识点</th><th>学前</th><th>学后</th><th>提升</th></tr>
            </thead>
            <tbody>
              {summary.sampleStudents.slice(0, 8).map((s, i) => (
                <tr key={i}>
                  <td>{s.studentName}</td>
                  <td>{s.topic}</td>
                  <td>{s.pre}</td>
                  <td>{s.post}</td>
                  <td style={{ color: '#e53935', fontWeight: 600 }}>+{s.improvement}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

const selStyle = { padding: '8px 12px', borderRadius: '8px', border: '1px solid #ddd', fontSize: '14px' };
const inpStyle = { padding: '8px 12px', borderRadius: '8px', border: '1px solid #ddd', fontSize: '14px' };
const phaseOn = { padding: '8px 14px', borderRadius: '8px', border: 'none', background: '#1a73e8', color: '#fff', cursor: 'pointer', fontWeight: 600 };
const phaseOff = { padding: '8px 14px', borderRadius: '8px', border: '1px solid #ddd', background: '#fff', color: '#333', cursor: 'pointer' };
const btnStyle = { padding: '10px 24px', borderRadius: '8px', border: 'none', background: 'linear-gradient(90deg,#ff7e3f,#ff5252)', color: '#fff', fontSize: '15px', fontWeight: 600, cursor: 'pointer' };

export default Efficacy;
