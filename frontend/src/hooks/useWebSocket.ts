import { useEffect, useRef, useCallback, useState } from 'react';
import { useAppStore } from '../store/useAppStore';
import { 
  AgentState, 
  LogMessage, 
  StepProgressMessage, 
  ContextUpdateMessage,
  DataSummaryMessage 
} from '../types';

// ============================================================================
// useWebSocket Hook - WebSocket 连接和消息处理
// ============================================================================

interface UseWebSocketOptions {
  url: string;
  autoConnect?: boolean;
  onOpen?: () => void;
  onClose?: () => void;
  onError?: (error: Event) => void;
}

export const useWebSocket = (options: UseWebSocketOptions) => {
  const { url, autoConnect = false, onOpen, onClose, onError } = options;
  
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  
  // 从 Store 获取 actions
  const {
    setAgentState,
    setCurrentStepDetails,
    addStepToHistory,
    updateContextVariables,
    setDataSummary,
    addLog,
    setCurrentStep,
  } = useAppStore();

  // 处理 WebSocket 消息
  const handleMessage = useCallback((event: MessageEvent) => {
    try {
      const message = JSON.parse(event.data);
      console.log('[WebSocket] 收到消息:', message);

      switch (message.type) {
        // Agent 状态更新
        case 'agent_state':
          setAgentState(message.state as AgentState);
          break;

        // 日志消息
        case 'log_message':
          if (message.log) {
            const log: LogMessage = {
              id: message.log.id || `log-${Date.now()}`,
              timestamp: message.log.timestamp || new Date().toISOString(),
              type: message.log.type,
              content: message.log.content,
            };
            addLog(log);
          }
          break;

        // 阶段完成
        case 'stage_complete':
          setCurrentStep(`${message.stage} - ${message.status}`);
          addLog({
            id: `stage-${Date.now()}`,
            timestamp: message.timestamp || new Date().toISOString(),
            type: 'system',
            content: `✓ ${message.stage} 完成`,
          });
          break;

        // Step 进度更新
        case 'step_progress':
          const stepMsg = message as StepProgressMessage;
          setCurrentStepDetails(
            stepMsg.stage_id,
            stepMsg.step_id,
            stepMsg.step_name
          );
          
          // 添加到历史记录
          addStepToHistory(
            stepMsg.stage_id,
            stepMsg.step_id,
            stepMsg.step_name,
            stepMsg.status as 'running' | 'completed' | 'failed'
          );
          
          // 添加日志
          if (stepMsg.status === 'running') {
            addLog({
              id: `step-${Date.now()}`,
              timestamp: stepMsg.timestamp,
              type: 'system',
              content: `▶ Stage ${stepMsg.stage_id}, Step ${stepMsg.step_id}: ${stepMsg.step_name}`,
            });
          } else if (stepMsg.status === 'completed') {
            addLog({
              id: `step-${Date.now()}`,
              timestamp: stepMsg.timestamp,
              type: 'system',
              content: `✓ Stage ${stepMsg.stage_id}, Step ${stepMsg.step_id} 完成`,
            });
          }
          break;

        // Context 变量更新
        case 'context_update':
          const contextMsg = message as ContextUpdateMessage;
          updateContextVariables(contextMsg.variables);
          
          // 添加日志
          const varNames = Object.keys(contextMsg.variables).join(', ');
          addLog({
            id: `context-${Date.now()}`,
            timestamp: contextMsg.timestamp,
            type: 'system',
            content: `📝 Context 变量更新: ${varNames}`,
          });
          break;

        // 数据摘要
        case 'data_summary':
          const summaryMsg = message as DataSummaryMessage;
          setDataSummary({
            ...summaryMsg.summary,
            target_column: summaryMsg.summary.target_column || null,
          });
          
          // 添加日志
          addLog({
            id: `summary-${Date.now()}`,
            timestamp: summaryMsg.timestamp,
            type: 'system',
            content: `📊 数据摘要: ${summaryMsg.summary.n_rows} 行 × ${summaryMsg.summary.n_cols} 列`,
          });
          break;

        // 错误消息
        case 'error':
          addLog({
            id: `error-${Date.now()}`,
            timestamp: new Date().toISOString(),
            type: 'stderr',
            content: `❌ 错误: ${message.message}`,
          });
          setAgentState('error');
          break;

        default:
          console.warn('[WebSocket] 未知消息类型:', message.type);
      }
    } catch (error) {
      console.error('[WebSocket] 消息解析失败:', error);
    }
  }, [
    setAgentState,
    setCurrentStepDetails,
    addStepToHistory,
    updateContextVariables,
    setDataSummary,
    addLog,
    setCurrentStep,
  ]);

  // 连接 WebSocket
  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      console.log('[WebSocket] 已连接，跳过重复连接');
      return;
    }

    console.log('[WebSocket] 正在连接:', url);
    
    const ws = new WebSocket(url);
    
    ws.onopen = () => {
      console.log('[WebSocket] 连接成功');
      setIsConnected(true);
      onOpen?.();
    };
    
    ws.onmessage = handleMessage;
    
    ws.onclose = () => {
      console.log('[WebSocket] 连接关闭');
      setIsConnected(false);
      onClose?.();
      
      // 自动重连（5秒后）
      reconnectTimeoutRef.current = setTimeout(() => {
        console.log('[WebSocket] 尝试重连...');
        connect();
      }, 5000);
    };
    
    ws.onerror = (error) => {
      console.error('[WebSocket] 连接错误:', error);
      setIsConnected(false);
      onError?.(error);
    };
    
    wsRef.current = ws;
  }, [url, handleMessage, onOpen, onClose, onError]);

  // 断开连接
  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
  }, []);

  // 发送消息
  const sendMessage = useCallback((message: any) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message));
      console.log('[WebSocket] 发送消息:', message);
    } else {
      console.error('[WebSocket] 连接未打开，无法发送消息');
    }
  }, []);

  // 启动分析
  const startAnalysis = useCallback((diseaseName: string) => {
    sendMessage({
      action: 'start_analysis',
      disease_name: diseaseName,
    });
  }, [sendMessage]);

  // 自动连接
  useEffect(() => {
    if (autoConnect) {
      connect();
    }

    return () => {
      disconnect();
    };
  }, [autoConnect, connect, disconnect]);

  return {
    connect,
    disconnect,
    sendMessage,
    startAnalysis,
    isConnected,
  };
};
