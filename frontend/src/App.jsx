import { Routes, Route, NavLink, Navigate } from 'react-router-dom';
import Home from './pages/Home';
import ProfileChat from './pages/ProfileChat';
import Generate from './pages/Generate';
import ResourceDetail from './pages/ResourceDetail';
import LearningPath from './pages/LearningPath';
import Tutor from './pages/Tutor';
import KnowledgeManage from './pages/KnowledgeManage';
import Analytics from './pages/Analytics';
import Efficacy from './pages/Efficacy';

function App() {
  return (
    <div className="app-layout">
      <aside className="sidebar">
        <div className="sidebar-logo">🤖 <span>学习</span>智能体系统</div>
        <div className="sidebar-section">学生端</div>
        <NavLink to="/" className={({isActive}) => isActive ? 'sidebar-link active' : 'sidebar-link'}>🏠 首页</NavLink>
        <NavLink to="/profile/chat" className={({isActive}) => isActive ? 'sidebar-link active' : 'sidebar-link'}>💬 画像对话</NavLink>
        <NavLink to="/generate" className={({isActive}) => isActive ? 'sidebar-link active' : 'sidebar-link'}>⚡ 资源生成</NavLink>
        <NavLink to="/path" className={({isActive}) => isActive ? 'sidebar-link active' : 'sidebar-link'}>🗺️ 学习路径</NavLink>
        <NavLink to="/tutor" className={({isActive}) => isActive ? 'sidebar-link active' : 'sidebar-link'}>❓ 智能辅导</NavLink>
        <NavLink to="/efficacy" className={({isActive}) => isActive ? 'sidebar-link active' : 'sidebar-link'}>📈 效果实证</NavLink>
        <div className="sidebar-section">管理端</div>
        <NavLink to="/admin/knowledge" className={({isActive}) => isActive ? 'sidebar-link active' : 'sidebar-link'}>📚 知识库管理</NavLink>
        <NavLink to="/admin/analytics" className={({isActive}) => isActive ? 'sidebar-link active' : 'sidebar-link'}>📊 学习效果</NavLink>
      </aside>
      <main className="main-content">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/profile/chat" element={<ProfileChat />} />
          <Route path="/generate" element={<Generate />} />
          <Route path="/resource/:id" element={<ResourceDetail />} />
          <Route path="/path" element={<LearningPath />} />
          <Route path="/tutor" element={<Tutor />} />
        <Route path="/efficacy" element={<Efficacy />} />
          <Route path="/admin" element={<Navigate to="/admin/knowledge" />} />
          <Route path="/admin/knowledge" element={<KnowledgeManage />} />
          <Route path="/admin/analytics" element={<Analytics />} />
        </Routes>
      </main>
    </div>
  );
}

export default App;
