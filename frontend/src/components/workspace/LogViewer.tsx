import { useRef, useEffect } from 'react';
import { Trash2, Terminal } from 'lucide-react';
import { useAppStore } from '../../store/useAppStore';

// ============================================================================
// LogViewer 组件 - 终端风格日志查看器
// ============================================================================

export const LogViewer = () => {
  const logEndRef = useRef<HTMLDivElement>(null);
  
  // 从 Store 获取日志
  const { logs, clearLogs } = useAppStore();

  // 自动滚动到底部
  const scrollToBottom = () => {
    logEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  // 当日志更新时自动滚动
  useEffect(() => {
    scrollToBottom();
  }, [logs]);

  // 清空日志
  const handleClearLogs = () => {
    if (window.confirm('确定要清空所有日志吗？')) {
      clearLogs();
    }
  };

  // 格式化时间戳
  const formatTimestamp = (timestamp: string) => {
    try {
      const date = new Date(timestamp);
      const base = date.toLocaleTimeString('zh-CN', {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
      });
      return `${base}.${String(date.getMilliseconds()).padStart(3, '0')}`;
    } catch {
      return timestamp;
    }
  };

  return (
    <div className="bg-white rounded-xl shadow-lg overflow-hidden border border-gray-200 flex flex-col h-[600px] w-full max-w-4xl">
      {/* ====================================================================
          顶部工具栏
      ==================================================================== */}
      <div className="flex items-center justify-between px-4 py-2 bg-gray-900 border-b border-gray-700 shrink-0">
        <div className="flex items-center gap-2">
          <Terminal size={16} className="text-green-400" />
          <span className="text-xs font-semibold text-white">执行日志</span>
        </div>
        
        <button
          onClick={handleClearLogs}
          className="flex items-center gap-1 px-2 py-1 text-xs text-gray-400 hover:text-red-400 hover:bg-gray-800 rounded transition-colors"
          title="清空日志"
        >
          <Trash2 size={14} />
          <span>清空</span>
        </button>
      </div>

      {/* ====================================================================
          日志内容区 - 固定高度可滚动
      ==================================================================== */}
      <div className="flex-1 overflow-y-auto p-4 space-y-1 min-h-0 bg-black">
        {/* 空状态提示 */}
        {logs.length === 0 && (
          <div className="text-gray-500 text-center py-8">
            <Terminal size={32} className="mx-auto mb-2 opacity-50" />
            <p className="text-xs">等待日志输出...</p>
          </div>
        )}

        {/* 日志列表 */}
        {logs.map((log) => (
          <div
            key={log.id}
            className="flex gap-2 hover:bg-gray-900 px-2 py-1 rounded text-xs leading-relaxed"
          >
            {/* 时间戳 */}
            <span className="text-gray-600 shrink-0 w-24">
              [{formatTimestamp(log.timestamp)}]
            </span>

            {/* 日志类型标签 */}
            <span
              className={`font-semibold shrink-0 w-16 ${
                log.type === 'stderr'
                  ? 'text-red-400'
                  : log.type === 'system'
                  ? 'text-blue-400'
                  : log.type === 'warning'
                  ? 'text-yellow-400'
                  : 'text-green-400'
              }`}
            >
              {log.type === 'stderr' && '[ERROR]'}
              {log.type === 'system' && '[SYS]'}
              {log.type === 'warning' && '[WARN]'}
              {log.type === 'stdout' && '[INFO]'}
            </span>

            {/* 日志内容 */}
            <span
              className={`flex-1 whitespace-pre-wrap break-words ${
                log.type === 'stderr' 
                  ? 'text-red-400' 
                  : log.type === 'warning'
                  ? 'text-yellow-400'
                  : 'text-green-300'
              }`}
            >
              {log.content}
            </span>
          </div>
        ))}

        {/* 自动滚动锚点 */}
        <div ref={logEndRef} />
      </div>

      {/* ====================================================================
          底部状态栏
      ==================================================================== */}
      <div className="px-4 py-2 bg-gray-900 border-t border-gray-700 text-xs text-gray-500 shrink-0">
        <div className="flex items-center justify-between">
          <span className="text-gray-400">共 {logs.length} 条日志</span>
          <div className="flex gap-4">
            <span className="text-yellow-400">
              警告: {logs.filter((log) => log.type === 'warning').length}
            </span>
            <span className="text-red-400">
              错误: {logs.filter((log) => log.type === 'stderr').length}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};
