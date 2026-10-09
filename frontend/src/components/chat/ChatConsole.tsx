import { useState, useRef, useEffect } from 'react';
import { Upload, Send, Loader2 } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { useAppStore } from '../../store/useAppStore';
import { ChatMessage } from '../../types';
import { StepProgressIndicator } from '../workspace/StepProgressIndicator';
import { ContextVariablesCompact } from '../workspace/ContextVariablesPanel';

// ============================================================================
// ChatConsole 组件 - 中间对话中枢 - 固定高度版本
// ============================================================================

export const ChatConsole = () => {
  const [inputValue, setInputValue] = useState('');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // 从 Store 获取状态
  const { agentState, currentStep, logs, messages, addMessage } = useAppStore();


  // 判断是否正在运行
  const isRunning =
    agentState === 'phase0_running' ||
    agentState === 'phase1_running';

  // 自动滚动到底部
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  // 当日志或消息更新时自动滚动
  useEffect(() => {
    scrollToBottom();
  }, [logs, messages, currentStep]);

  // 处理文件选择
  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
    }
  };

  // 处理发送消息
  const handleSendMessage = () => {
    if (!inputValue.trim() && !selectedFile) return;

    // 创建用户消息
    const userMessage: ChatMessage = {
      id: `msg-${Date.now()}`,
      timestamp: new Date().toISOString(),
      role: 'user',
      content: inputValue.trim(),
      file: selectedFile
        ? {
            name: selectedFile.name,
            size: selectedFile.size,
          }
        : undefined,
    };

    // 添加到 Store
    addMessage(userMessage);

    // TODO: 发送到后端 API
    console.log('发送消息:', userMessage);

    // 模拟 Agent 回复（实际应从 WebSocket 或 API 接收）
    setTimeout(() => {
      const agentMessage: ChatMessage = {
        id: `msg-${Date.now()}`,
        timestamp: new Date().toISOString(),
        role: 'agent',
        content: `收到您的消息：**${inputValue.trim()}**\n\n这是一个示例回复，支持 **Markdown** 格式：\n\n- 列表项 1\n- 列表项 2\n\n\`\`\`python\nprint("Hello, MetaboAgent!")\n\`\`\``,
      };
      addMessage(agentMessage);
    }, 1000);

    // 清空输入
    setInputValue('');
    setSelectedFile(null);
  };

  // 处理回车发送
  const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <div 
      style={{
        width: '100%',
        height: '100vh',
        display: 'flex',
        flexDirection: 'column',
        backgroundColor: 'white',
        position: 'relative',
      }}
    >
      {/* ====================================================================
          顶部标题区 - 固定高度 80px
      ==================================================================== */}
      <div 
        style={{
          height: '80px',
          flexShrink: 0,
          padding: '1.5rem',
          borderBottom: '1px solid #e5e7eb',
          backgroundColor: 'white',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div>
            <h2 style={{ fontSize: '1.125rem', fontWeight: 600, color: '#111827', marginBottom: '0.25rem' }}>
              分析控制台
            </h2>
            <p style={{ fontSize: '0.875rem', color: '#6b7280' }}>
              {isRunning ? (
                <span style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <Loader2 size={14} className="animate-spin text-blue-600" />
                  <span style={{ color: '#2563eb' }}>正在执行分析...</span>
                </span>
              ) : (
                '实时查看分析进度和日志'
              )}
            </p>
          </div>

          {/* 状态指示器 */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <div
              style={{
                width: '8px',
                height: '8px',
                borderRadius: '50%',
                backgroundColor:
                  agentState === 'idle'
                    ? '#9ca3af'
                    : agentState === 'error'
                    ? '#ef4444'
                    : agentState === 'completed'
                    ? '#10b981'
                    : '#3b82f6',
              }}
            />
            <span style={{ fontSize: '0.875rem', color: '#4b5563' }}>
              {agentState === 'idle' && '空闲'}
              {agentState === 'phase0_running' && 'Phase 0'}
              {agentState === 'phase1_running' && 'Phase 1'}
              {agentState === 'completed' && '已完成'}
              {agentState === 'error' && '错误'}
            </span>
          </div>
        </div>
      </div>

      {/* ====================================================================
          中间消息列表区 - 固定高度，使用 calc()
      ==================================================================== */}
      <div 
        style={{
          height: 'calc(100vh - 80px - 120px)', // 总高度 - 顶部 - 底部
          overflowY: 'auto',
          padding: '1.5rem',
          backgroundColor: '#f9fafb',
        }}
      >
        {/* 欢迎消息 */}
        {messages.length === 0 && logs.length === 0 && (
          <div style={{ textAlign: 'center', padding: '3rem 0' }}>
            <div style={{ 
              display: 'inline-block', 
              padding: '1rem', 
              backgroundColor: '#dbeafe', 
              borderRadius: '50%', 
              marginBottom: '1rem' 
            }}>
              <Upload size={32} style={{ color: '#2563eb' }} />
            </div>
            <h3 style={{ fontSize: '1.125rem', fontWeight: 600, color: '#111827', marginBottom: '0.5rem' }}>
              欢迎使用 MetaboAgent
            </h3>
            <p style={{ fontSize: '0.875rem', color: '#6b7280', marginBottom: '1rem' }}>
              上传 CSV 数据文件开始分析，或输入指令与 Agent 交互
            </p>
          </div>
        )}

        {/* 聊天消息列表 */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          {messages.map((message) => (
            <ChatMessageBubble key={message.id} message={message} />
          ))}

          {/* 日志消息列表 */}
          {logs.map((log) => (
            <LogMessageBubble key={log.id} log={log} />
          ))}

          {/* Step 进度指示器 - 当正在运行时显示 */}
          {isRunning && <StepProgressIndicator />}
          
          {/* Context 变量紧凑展示 */}
          {isRunning && (
            <div style={{ padding: '0.75rem', backgroundColor: 'white', border: '1px solid #e5e7eb', borderRadius: '0.5rem' }}>
              <div style={{ fontSize: '0.75rem', fontWeight: 600, color: '#374151', marginBottom: '0.5rem' }}>
                当前 Context 变量
              </div>
              <ContextVariablesCompact />
            </div>
          )}

          {/* 自动滚动锚点 */}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* ====================================================================
          底部输入区 - 固定高度 120px
      ==================================================================== */}
      <div 
        style={{
          height: '120px',
          flexShrink: 0,
          padding: '1rem 1.5rem',
          borderTop: '1px solid #e5e7eb',
          backgroundColor: 'white',
        }}
      >
        {/* 已选择文件提示 */}
        {selectedFile && (
          <div style={{ 
            marginBottom: '0.75rem', 
            padding: '0.5rem', 
            backgroundColor: '#dbeafe', 
            border: '1px solid #93c5fd', 
            borderRadius: '0.5rem', 
            display: 'flex', 
            alignItems: 'center', 
            justifyContent: 'space-between' 
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Upload size={16} style={{ color: '#2563eb' }} />
              <span style={{ fontSize: '0.875rem', color: '#1e3a8a' }}>{selectedFile.name}</span>
              <span style={{ fontSize: '0.75rem', color: '#2563eb' }}>
                ({(selectedFile.size / 1024).toFixed(2)} KB)
              </span>
            </div>
            <button
              onClick={() => setSelectedFile(null)}
              style={{ fontSize: '0.75rem', color: '#2563eb', background: 'none', border: 'none', cursor: 'pointer' }}
            >
              移除
            </button>
          </div>
        )}

        {/* 输入框和按钮 */}
        <div style={{ display: 'flex', gap: '0.5rem' }}>
          {/* 上传 CSV 按钮 */}
          <input
            ref={fileInputRef}
            type="file"
            accept=".csv,.xlsx,.xls"
            onChange={handleFileSelect}
            style={{ display: 'none' }}
          />
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={isRunning}
            style={{
              padding: '0.5rem 1rem',
              backgroundColor: '#f3f4f6',
              color: '#374151',
              border: 'none',
              borderRadius: '0.5rem',
              cursor: isRunning ? 'not-allowed' : 'pointer',
              opacity: isRunning ? 0.5 : 1,
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
              fontSize: '0.875rem',
              fontWeight: 500,
            }}
            title="上传 CSV 数据"
          >
            <Upload size={18} />
            <span>上传数据</span>
          </button>

          {/* 文本输入框 */}
          <input
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyPress={handleKeyPress}
            disabled={isRunning}
            placeholder={
              isRunning ? '分析进行中，请稍候...' : '输入指令或问题...'
            }
            style={{
              flex: 1,
              padding: '0.5rem 1rem',
              border: '1px solid #d1d5db',
              borderRadius: '0.5rem',
              fontSize: '0.875rem',
              backgroundColor: isRunning ? '#f3f4f6' : 'white',
              cursor: isRunning ? 'not-allowed' : 'text',
            }}
          />

          {/* 发送按钮 */}
          <button
            onClick={handleSendMessage}
            disabled={isRunning || (!inputValue.trim() && !selectedFile)}
            style={{
              padding: '0.5rem 1.5rem',
              backgroundColor: '#2563eb',
              color: 'white',
              border: 'none',
              borderRadius: '0.5rem',
              cursor: (isRunning || (!inputValue.trim() && !selectedFile)) ? 'not-allowed' : 'pointer',
              opacity: (isRunning || (!inputValue.trim() && !selectedFile)) ? 0.5 : 1,
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
              fontWeight: 500,
            }}
          >
            <Send size={18} />
            <span>发送</span>
          </button>
        </div>

        {/* 提示文本 */}
        <div style={{ marginTop: '0.5rem', fontSize: '0.75rem', color: '#6b7280' }}>
          按 Enter 发送消息 • 支持 CSV、Excel 格式数据文件
        </div>
      </div>
    </div>
  );
};

// ============================================================================
// ChatMessageBubble 组件 - ChatGPT 风格消息气泡
// ============================================================================

interface ChatMessageBubbleProps {
  message: ChatMessage;
}

const ChatMessageBubble = ({ message }: ChatMessageBubbleProps) => {
  const isUser = message.role === 'user';

  // 格式化时间戳
  const formatTime = (timestamp: string) => {
    try {
      const date = new Date(timestamp);
      return date.toLocaleTimeString('zh-CN', {
        hour: '2-digit',
        minute: '2-digit',
      });
    } catch {
      return '';
    }
  };

  if (isUser) {
    // 用户消息 - 右侧，蓝色背景
    return (
      <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
        <div style={{ maxWidth: '80%' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end', gap: '0.5rem', marginBottom: '0.25rem' }}>
            <span style={{ fontSize: '0.75rem', color: '#6b7280' }}>{formatTime(message.timestamp)}</span>
            <span style={{ fontSize: '0.75rem', fontWeight: 500, color: '#374151' }}>我</span>
          </div>
          
          {/* 文件附件 */}
          {message.file && (
            <div style={{ 
              marginBottom: '0.5rem', 
              padding: '0.5rem', 
              backgroundColor: '#dbeafe', 
              border: '1px solid #93c5fd', 
              borderRadius: '0.5rem', 
              display: 'flex', 
              alignItems: 'center', 
              gap: '0.5rem', 
              justifyContent: 'flex-end' 
            }}>
              <Upload size={14} style={{ color: '#2563eb' }} />
              <span style={{ fontSize: '0.75rem', color: '#1e3a8a' }}>
                {message.file.name} ({(message.file.size / 1024).toFixed(2)} KB)
              </span>
            </div>
          )}
          
          {/* 消息内容 */}
          <div style={{ 
            backgroundColor: '#dbeafe', 
            color: '#111827', 
            borderRadius: '1rem', 
            padding: '0.75rem 1rem' 
          }}>
            <div style={{ fontSize: '0.875rem', whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
              {message.content}
            </div>
          </div>
        </div>
      </div>
    );
  } else {
    // Agent/System 消息 - 左侧，浅灰色背景
    return (
      <div style={{ display: 'flex', justifyContent: 'flex-start' }}>
        <div style={{ maxWidth: '80%' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.25rem' }}>
            <span style={{ fontSize: '0.75rem', fontWeight: 500, color: '#374151' }}>
              {message.role === 'system' ? '系统' : 'MetaboAgent'}
            </span>
            <span style={{ fontSize: '0.75rem', color: '#6b7280' }}>{formatTime(message.timestamp)}</span>
          </div>
          
          {/* 消息内容 */}
          <div style={{ 
            backgroundColor: '#f3f4f6', 
            color: '#111827', 
            borderRadius: '1rem', 
            padding: '0.75rem 1rem' 
          }}>
            <div style={{ fontSize: '0.875rem' }}>
              <ReactMarkdown
                className="prose prose-sm max-w-none"
                components={{
                  p: ({ children }) => (
                    <p style={{ marginBottom: '0.5rem', color: '#111827' }}>{children}</p>
                  ),
                  strong: ({ children }) => (
                    <strong style={{ fontWeight: 600, color: '#111827' }}>{children}</strong>
                  ),
                  code: ({ children }) => (
                    <code style={{ 
                      padding: '0.125rem 0.375rem', 
                      backgroundColor: '#e5e7eb', 
                      color: '#111827', 
                      borderRadius: '0.25rem', 
                      fontSize: '0.75rem', 
                      fontFamily: 'monospace' 
                    }}>
                      {children}
                    </code>
                  ),
                  pre: ({ children }) => (
                    <pre style={{ 
                      marginTop: '0.5rem', 
                      marginBottom: '0.5rem', 
                      padding: '0.75rem', 
                      backgroundColor: '#1f2937', 
                      color: '#f3f4f6', 
                      borderRadius: '0.5rem', 
                      overflowX: 'auto', 
                      fontSize: '0.75rem' 
                    }}>
                      {children}
                    </pre>
                  ),
                  ul: ({ children }) => (
                    <ul style={{ 
                      listStyleType: 'disc', 
                      listStylePosition: 'inside', 
                      marginBottom: '0.5rem', 
                      color: '#111827' 
                    }}>
                      {children}
                    </ul>
                  ),
                  ol: ({ children }) => (
                    <ol style={{ 
                      listStyleType: 'decimal', 
                      listStylePosition: 'inside', 
                      marginBottom: '0.5rem', 
                      color: '#111827' 
                    }}>
                      {children}
                    </ol>
                  ),
                  li: ({ children }) => (
                    <li style={{ color: '#111827' }}>{children}</li>
                  ),
                }}
              >
                {message.content}
              </ReactMarkdown>
            </div>
          </div>
        </div>
      </div>
    );
  }
};

// ============================================================================
// LogMessageBubble 组件 - 日志消息气泡
// ============================================================================

interface LogMessageBubbleProps {
  log: {
    id: string;
    timestamp: string;
    type: 'stdout' | 'stderr' | 'system' | 'warning';
    content: string;
  };
}

const LogMessageBubble = ({ log }: LogMessageBubbleProps) => {
  // 根据类型设置样式
  const getMessageStyle = () => {
    switch (log.type) {
      case 'system':
        return {
          container: { backgroundColor: '#f3f4f6', border: '1px solid #e5e7eb' },
          label: { color: '#4b5563' },
          content: { color: '#1f2937' },
        };
      case 'stderr':
        return {
          container: { backgroundColor: '#fef2f2', border: '1px solid #fecaca' },
          label: { color: '#dc2626' },
          content: { color: '#7f1d1d' },
        };
      case 'stdout':
      default:
        return {
          container: { backgroundColor: '#f0fdf4', border: '1px solid #bbf7d0' },
          label: { color: '#16a34a' },
          content: { color: '#14532d' },
        };
    }
  };

  const style = getMessageStyle();

  // 格式化时间戳
  const formatTime = (timestamp: string) => {
    try {
      const date = new Date(timestamp);
      return date.toLocaleTimeString('zh-CN', {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
      });
    } catch {
      return '';
    }
  };

  return (
    <div style={{ 
      padding: '1rem', 
      borderRadius: '0.5rem', 
      ...style.container 
    }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
        <span style={{ fontSize: '0.75rem', fontWeight: 500, ...style.label }}>
          {log.type === 'system' && '系统'}
          {log.type === 'stdout' && 'Agent'}
          {log.type === 'stderr' && '错误'}
        </span>
        <span style={{ fontSize: '0.75rem', color: '#6b7280' }}>{formatTime(log.timestamp)}</span>
      </div>
      <div style={{ fontSize: '0.875rem', whiteSpace: 'pre-wrap', ...style.content }}>
        {log.content}
      </div>
    </div>
  );
};
