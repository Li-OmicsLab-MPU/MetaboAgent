import { useAppStore } from '../store/useAppStore';
import { WebSocketMessage, LogMessage, AgentState } from '../types';

// ============================================================================
// WebSocket 配置
// ============================================================================

const WS_BASE_URL = import.meta.env.VITE_WS_BASE_URL || 'ws://localhost:8000';
const MAX_RECONNECT_ATTEMPTS = 3;
const RECONNECT_DELAY = 3000; // 3 秒

// ============================================================================
// WebSocket 消息类型定义
// ============================================================================

interface LogMessagePayload {
  id: string;
  timestamp: string;
  type: 'stdout' | 'stderr' | 'system';
  content: string;
}

interface ProgressUpdatePayload {
  task_id: string;
  phase: 'phase0' | 'phase1';
  step: string;
  progress: number;
  message: string;
  agent_state?: AgentState;
}

interface StateChangePayload {
  task_id: string;
  old_state: AgentState;
  new_state: AgentState;
  timestamp: string;
}

interface Phase0ResultPayload {
  disease_name: string;
  confirmed_biomarkers: any[];
  target_pathways: string[];
  [key: string]: any;
}

interface Phase1ResultPayload {
  feature_selection: any;
  training: any;
  [key: string]: any;
}

// ============================================================================
// WebSocket 管理器类
// ============================================================================

class WebSocketManager {
  private ws: WebSocket | null = null;
  private projectId: string | null = null;
  private reconnectAttempts = 0;
  private reconnectTimer: NodeJS.Timeout | null = null;
  private isManualClose = false;

  /**
   * 连接到 WebSocket 服务器
   * @param projectId 项目 ID
   */
  connect(projectId: string): void {
    // 如果已经连接，先断开
    if (this.ws?.readyState === WebSocket.OPEN) {
      console.log('[WebSocket] 已存在连接，先断开旧连接');
      this.disconnect();
    }

    this.projectId = projectId;
    this.isManualClose = false;
    this.reconnectAttempts = 0;

    this.createConnection();
  }

  /**
   * 创建 WebSocket 连接
   */
  private createConnection(): void {
    if (!this.projectId) {
      console.error('[WebSocket] 缺少 projectId，无法连接');
      return;
    }

    try {
      const url = `${WS_BASE_URL}/ws/${this.projectId}`;
      console.log(`[WebSocket] 正在连接: ${url}`);

      this.ws = new WebSocket(url);

      // 连接成功
      this.ws.onopen = () => {
        console.log('[WebSocket] 连接成功');
        this.reconnectAttempts = 0;
        
        // 可选：发送初始化消息
        this.sendMessage({
          type: 'client_connected',
          timestamp: new Date().toISOString(),
        });
      };

      // 接收消息
      this.ws.onmessage = (event) => {
        this.handleMessage(event.data);
      };

      // 连接错误
      this.ws.onerror = (error) => {
        console.error('[WebSocket] 连接错误:', error);
      };

      // 连接关闭
      this.ws.onclose = (event) => {
        console.log(`[WebSocket] 连接关闭 (code: ${event.code}, reason: ${event.reason})`);
        
        // 如果不是手动关闭，尝试重连
        if (!this.isManualClose) {
          this.attemptReconnect();
        }
      };
    } catch (error) {
      console.error('[WebSocket] 创建连接失败:', error);
      this.attemptReconnect();
    }
  }

  /**
   * 尝试重新连接
   */
  private attemptReconnect(): void {
    if (this.reconnectAttempts >= MAX_RECONNECT_ATTEMPTS) {
      console.error(`[WebSocket] 已达到最大重连次数 (${MAX_RECONNECT_ATTEMPTS})，停止重连`);
      
      // 更新状态为错误
      const store = useAppStore.getState();
      store.setAgentState('error');
      store.addLog({
        id: `error-${Date.now()}`,
        timestamp: new Date().toISOString(),
        type: 'system',
        content: `WebSocket 连接失败，已尝试重连 ${MAX_RECONNECT_ATTEMPTS} 次`,
      });
      
      return;
    }

    this.reconnectAttempts++;
    console.log(
      `[WebSocket] 将在 ${RECONNECT_DELAY / 1000} 秒后尝试第 ${this.reconnectAttempts} 次重连...`
    );

    // 清除之前的定时器
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
    }

    // 设置重连定时器
    this.reconnectTimer = setTimeout(() => {
      console.log(`[WebSocket] 开始第 ${this.reconnectAttempts} 次重连`);
      this.createConnection();
    }, RECONNECT_DELAY);
  }

  /**
   * 处理接收到的消息
   * @param data 消息数据
   */
  private handleMessage(data: string): void {
    try {
      const message: WebSocketMessage = JSON.parse(data);
      console.log('[WebSocket] 收到消息:', message.type);

      const store = useAppStore.getState();

      switch (message.type) {
        case 'log':
        case 'log_message':
          this.handleLogMessage(message.data as LogMessagePayload, store);
          break;

        case 'progress':
        case 'progress_update':
          this.handleProgressUpdate(message.data as ProgressUpdatePayload, store);
          break;

        case 'state_change':
          this.handleStateChange(message.data as StateChangePayload, store);
          break;

        case 'phase0_result':
          this.handlePhase0Result(message.data as Phase0ResultPayload, store);
          break;

        case 'phase1_result':
          this.handlePhase1Result(message.data as Phase1ResultPayload, store);
          break;

        case 'error':
          this.handleError(message.data, store);
          break;

        default:
          console.warn('[WebSocket] 未知消息类型:', message.type);
      }
    } catch (error) {
      console.error('[WebSocket] 解析消息失败:', error);
      console.error('[WebSocket] 原始数据:', data);
      
      // 不让系统崩溃，记录错误日志
      try {
        const store = useAppStore.getState();
        store.addLog({
          id: `parse-error-${Date.now()}`,
          timestamp: new Date().toISOString(),
          type: 'system',
          content: `解析 WebSocket 消息失败: ${error instanceof Error ? error.message : '未知错误'}`,
        });
      } catch (logError) {
        console.error('[WebSocket] 记录错误日志失败:', logError);
      }
    }
  }

  /**
   * 处理日志消息
   */
  private handleLogMessage(data: LogMessagePayload, store: ReturnType<typeof useAppStore.getState>): void {
    try {
      const log: LogMessage = {
        id: data.id || `log-${Date.now()}`,
        timestamp: data.timestamp || new Date().toISOString(),
        type: data.type || 'stdout',
        content: data.content || '',
      };

      store.addLog(log);
    } catch (error) {
      console.error('[WebSocket] 处理日志消息失败:', error);
    }
  }

  /**
   * 处理进度更新
   */
  private handleProgressUpdate(data: ProgressUpdatePayload, store: ReturnType<typeof useAppStore.getState>): void {
    try {
      // 更新当前步骤
      if (data.step) {
        store.setCurrentStep(data.step);
      }

      // 更新 Agent 状态
      if (data.agent_state) {
        store.setAgentState(data.agent_state);
      }

      // 更新任务进度
      if (data.task_id && store.currentTask?.id === data.task_id) {
        store.updateTask(data.task_id, {
          progress: data.progress || 0,
          current_phase: data.phase,
          current_step: data.step,
        });
      }

      // 添加进度日志
      if (data.message) {
        store.addLog({
          id: `progress-${Date.now()}`,
          timestamp: new Date().toISOString(),
          type: 'system',
          content: data.message,
        });
      }
    } catch (error) {
      console.error('[WebSocket] 处理进度更新失败:', error);
    }
  }

  /**
   * 处理状态变更
   */
  private handleStateChange(data: StateChangePayload, store: ReturnType<typeof useAppStore.getState>): void {
    try {
      store.setAgentState(data.new_state);

      store.addLog({
        id: `state-${Date.now()}`,
        timestamp: data.timestamp || new Date().toISOString(),
        type: 'system',
        content: `状态变更: ${data.old_state} -> ${data.new_state}`,
      });
    } catch (error) {
      console.error('[WebSocket] 处理状态变更失败:', error);
    }
  }

  /**
   * 处理 Phase 0 结果
   */
  private handlePhase0Result(data: Phase0ResultPayload, store: ReturnType<typeof useAppStore.getState>): void {
    try {
      store.setPhase0Result(data);

      store.addLog({
        id: `phase0-result-${Date.now()}`,
        timestamp: new Date().toISOString(),
        type: 'system',
        content: `Phase 0 完成: 发现 ${data.confirmed_biomarkers?.length || 0} 个生物标志物，${data.target_pathways?.length || 0} 条通路`,
      });
    } catch (error) {
      console.error('[WebSocket] 处理 Phase 0 结果失败:', error);
    }
  }

  /**
   * 处理 Phase 1 结果
   */
  private handlePhase1Result(data: Phase1ResultPayload, store: ReturnType<typeof useAppStore.getState>): void {
    try {
      store.setPhase1Result(data);

      store.addLog({
        id: `phase1-result-${Date.now()}`,
        timestamp: new Date().toISOString(),
        type: 'system',
        content: `Phase 1 完成: 最佳模型 ${data.training?.best_model || 'N/A'}，准确率 ${data.training?.accuracy || 0}`,
      });
    } catch (error) {
      console.error('[WebSocket] 处理 Phase 1 结果失败:', error);
    }
  }

  /**
   * 处理错误消息
   */
  private handleError(data: any, store: ReturnType<typeof useAppStore.getState>): void {
    try {
      store.setAgentState('error');

      store.addLog({
        id: `error-${Date.now()}`,
        timestamp: new Date().toISOString(),
        type: 'stderr',
        content: typeof data === 'string' ? data : JSON.stringify(data),
      });
    } catch (error) {
      console.error('[WebSocket] 处理错误消息失败:', error);
    }
  }

  /**
   * 发送消息到服务器
   * @param data 要发送的数据
   */
  sendMessage(data: any): void {
    try {
      if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
        console.error('[WebSocket] 连接未打开，无法发送消息');
        return;
      }

      const message = typeof data === 'string' ? data : JSON.stringify(data);
      this.ws.send(message);
      console.log('[WebSocket] 消息已发送:', data);
    } catch (error) {
      console.error('[WebSocket] 发送消息失败:', error);
    }
  }

  /**
   * 断开连接
   */
  disconnect(): void {
    console.log('[WebSocket] 手动断开连接');
    
    this.isManualClose = true;
    
    // 清除重连定时器
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }

    // 关闭 WebSocket 连接
    if (this.ws) {
      this.ws.close(1000, 'Client disconnect');
      this.ws = null;
    }

    this.projectId = null;
    this.reconnectAttempts = 0;
  }

  /**
   * 获取当前连接状态
   */
  getConnectionState(): number {
    return this.ws?.readyState ?? WebSocket.CLOSED;
  }

  /**
   * 检查是否已连接
   */
  isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }
}

// ============================================================================
// 导出单例实例
// ============================================================================

export const wsManager = new WebSocketManager();
