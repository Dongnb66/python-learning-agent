// 思维导图组件
import React from 'react';

function MindMapView({ data }) {
  if (!data) return null;
  
  const renderNode = (node, isRoot = false) => (
    <div key={node.name}>
      <span className={`mindmap-node ${isRoot ? 'mindmap-root' : ''}`}>{node.name}</span>
      {node.children && node.children.length > 0 && (
        <div className="mindmap-children">
          {node.children.map(child => renderNode(child))}
        </div>
      )}
    </div>
  );
  
  return <div>{renderNode(data, true)}</div>;
}

export default MindMapView;
