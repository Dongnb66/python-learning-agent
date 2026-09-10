// 简单的 Markdown 渲染器
import React from 'react';

function Markdown({ content }) {
  if (!content) return null;
  
  const lines = content.split('\n');
  const elements = [];
  let inList = false;
  let listBuffer = [];
  let inCode = false;
  let codeBuffer = [];
  
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    
    // 代码块
    if (line.startsWith('```')) {
      if (inCode) {
        elements.push(
          <div key={`code-${i}`} className="code-block">
            <pre>{codeBuffer.join('\n')}</pre>
          </div>
        );
        codeBuffer = [];
        inCode = false;
      } else {
        inCode = true;
      }
      continue;
    }
    if (inCode) {
      codeBuffer.push(line);
      continue;
    }
    
    // 标题
    if (line.startsWith('### ')) {
      if (inList) { inList = false; }
      elements.push(<h3 key={i}>{renderInline(line.slice(4))}</h3>);
    } else if (line.startsWith('## ')) {
      if (inList) { inList = false; }
      elements.push(<h2 key={i}>{renderInline(line.slice(3))}</h2>);
    } else if (line.startsWith('# ')) {
      if (inList) { inList = false; }
      elements.push(<h1 key={i}>{renderInline(line.slice(2))}</h1>);
    } else if (line.startsWith('- ') || line.startsWith('* ')) {
      if (!inList) {
        inList = true;
        listBuffer = [];
      }
      listBuffer.push(<li key={i}>{renderInline(line.slice(2))}</li>);
    } else if (line.match(/^\d+\.\s/)) {
      if (!inList) {
        inList = true;
        listBuffer = [];
      }
      listBuffer.push(<li key={i}>{renderInline(line.replace(/^\d+\.\s/, ''))}</li>);
    } else if (line.trim() === '') {
      if (inList) {
        elements.push(<ul key={`ul-${i}`}>{listBuffer}</ul>);
        listBuffer = [];
        inList = false;
      }
    } else {
      if (inList) {
        elements.push(<ul key={`ul-${i}`}>{listBuffer}</ul>);
        listBuffer = [];
        inList = false;
      }
      elements.push(<p key={i}>{renderInline(line)}</p>);
    }
  }

  if (inList && listBuffer.length > 0) {
    elements.push(<ul key="ul-final">{listBuffer}</ul>);
  }
  
  if (inCode && codeBuffer.length > 0) {
    elements.push(
      <div key="code-final" className="code-block">
        <pre>{codeBuffer.join('\n')}</pre>
      </div>
    );
  }
  
  return <div className="markdown-body">{elements}</div>;
}

function renderInline(text) {
  // 处理行内代码 `code`
  const parts = [];
  let remaining = text;
  let key = 0;
  
  while (remaining.length > 0) {
    const codeMatch = remaining.match(/`([^`]+)`/);
    if (codeMatch) {
      const before = remaining.substring(0, codeMatch.index);
      if (before) parts.push(before);
      parts.push(<code key={key++}>{codeMatch[1]}</code>);
      remaining = remaining.substring(codeMatch.index + codeMatch[0].length);
    } else {
      parts.push(remaining);
      break;
    }
  }
  
  return parts.length > 1 ? parts : text;
}

export default Markdown;
