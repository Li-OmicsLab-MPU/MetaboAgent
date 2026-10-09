import { create } from 'zustand';
import {
  User,
  AnalysisTask,
  AgentState,
  LogMessage,
  ChatMessage,
  Phase0Result,
  Phase1Result,
} from '../types';

// ============================================================================
// State 接口定义
// ============================================================================

interface AnalysisResults {
  phase0?: Phase0Result;
  phase1?: Phase1Result;
}

interface AppState {
  // -------------------------------------------------------------------------
  // Agent 运行状态
  // -------------------------------------------------------------------------
  agentState: AgentState;
  currentStep: string;
  logs: LogMessage[];
  analysisResults: AnalysisResults;

  // -------------------------------------------------------------------------
  // 详细状态（Step 级别）
  // -------------------------------------------------------------------------
  currentStageId: string | null;      // 当前 Stage ID
  currentStepId: string | null;       // 当前 Step ID
  currentStepName: string | null;     // 当前 Step 名称
  
  // -------------------------------------------------------------------------
  // Step 历史记录（用于时间线显示）
  // -------------------------------------------------------------------------
  stepHistory: Array<{
    stageId: string;
    stepId: string;
    stepName: string;
    status: 'running' | 'completed' | 'failed';
    timestamp: string;
  }>;
  
  // -------------------------------------------------------------------------
  // Context 变量（Phase 1 动态变量）
  // -------------------------------------------------------------------------
  contextVariables: Record<string, any>;
  
  // -------------------------------------------------------------------------
  // 数据摘要
  // -------------------------------------------------------------------------
  dataSummary: {
    n_rows: number;
    n_cols: number;
    columns: string[];
    target_column: string | null;
  } | null;

  // -------------------------------------------------------------------------
  // 聊天消息状态
  // -------------------------------------------------------------------------
  messages: ChatMessage[];

  // -------------------------------------------------------------------------
  // Agent 状态管理 Actions
  // -------------------------------------------------------------------------
  setAgentState: (state: AgentState) => void;
  setCurrentStep: (step: string) => void;
  setCurrentStepDetails: (stageId: string, stepId: string, stepName: string) => void;
  addStepToHistory: (stageId: string, stepId: string, stepName: string, status: 'running' | 'completed' | 'failed') => void;
  updateContextVariables: (variables: Record<string, any>) => void;
  setDataSummary: (summary: { n_rows: number; n_cols: number; columns: string[]; target_column: string | null }) => void;
  addLog: (log: LogMessage) => void;
  clearLogs: () => void;
  addMessage: (message: ChatMessage) => void;
  clearMessages: () => void;
  setPhase0Result: (data: Phase0Result) => void;
  setPhase1Result: (data: Phase1Result) => void;
  resetStore: () => void;

  // -------------------------------------------------------------------------
  // 用户状态
  // -------------------------------------------------------------------------
  user: User | null;
  isAuthenticated: boolean;
  setUser: (user: User | null) => void;
  logout: () => void;

  // -------------------------------------------------------------------------
  // 任务状态
  // -------------------------------------------------------------------------
  tasks: AnalysisTask[];
  currentTask: AnalysisTask | null;
  setTasks: (tasks: AnalysisTask[]) => void;
  setCurrentTask: (task: AnalysisTask | null) => void;
  updateTask: (taskId: string, updates: Partial<AnalysisTask>) => void;

  // -------------------------------------------------------------------------
  // UI 状态
  // -------------------------------------------------------------------------
  sidebarOpen: boolean;
  toggleSidebar: () => void;
  loading: boolean;
  setLoading: (loading: boolean) => void;
}

// ============================================================================
// 初始状态
// ============================================================================

const initialAnalysisState = {
  agentState: 'idle' as AgentState,
  currentStep: '',
  logs: [] as LogMessage[],
  messages: [] as ChatMessage[],
  analysisResults: {} as AnalysisResults,
  // 详细状态
  currentStageId: null,
  currentStepId: null,
  currentStepName: null,
  // Step 历史记录
  stepHistory: [] as Array<{
    stageId: string;
    stepId: string;
    stepName: string;
    status: 'running' | 'completed' | 'failed';
    timestamp: string;
  }>,
  // Context 变量
  contextVariables: {},
  // 数据摘要
  dataSummary: null,
};

// ============================================================================
// Zustand Store 创建
// ============================================================================

export const useAppStore = create<AppState>((set) => ({
  // -------------------------------------------------------------------------
  // Agent 运行状态初始值
  // -------------------------------------------------------------------------
  ...initialAnalysisState,

  // -------------------------------------------------------------------------
  // Agent 状态管理 Actions
  // -------------------------------------------------------------------------
  setAgentState: (agentState) =>
    set(() => ({
      agentState,
    })),

  setCurrentStep: (currentStep) =>
    set(() => ({
      currentStep,
    })),

  setCurrentStepDetails: (stageId, stepId, stepName) =>
    set(() => ({
      currentStageId: stageId,
      currentStepId: stepId,
      currentStepName: stepName,
      currentStep: `Stage ${stageId}: ${stepName}`,
    })),

  addStepToHistory: (stageId, stepId, stepName, status) =>
    set((state) => {
      // 检查是否已存在相同的 step
      const existingIndex = state.stepHistory.findIndex(
        (s) => s.stageId === stageId && s.stepId === stepId
      );
      
      if (existingIndex >= 0) {
        // 更新现有 step 的状态
        const newHistory = [...state.stepHistory];
        newHistory[existingIndex] = {
          ...newHistory[existingIndex],
          status,
          timestamp: new Date().toISOString(),
        };
        return { stepHistory: newHistory };
      } else {
        // 添加新 step
        return {
          stepHistory: [
            ...state.stepHistory,
            {
              stageId,
              stepId,
              stepName,
              status,
              timestamp: new Date().toISOString(),
            },
          ],
        };
      }
    }),

  updateContextVariables: (variables) =>
    set((state) => ({
      contextVariables: {
        ...state.contextVariables,
        ...variables,
      },
    })),

  setDataSummary: (summary) =>
    set(() => ({
      dataSummary: summary,
    })),

  addLog: (log) =>
    set((state) => ({
      logs: [...state.logs, log],
    })),

  clearLogs: () =>
    set(() => ({
      logs: [],
    })),

  addMessage: (message) =>
    set((state) => ({
      messages: [...state.messages, message],
    })),

  clearMessages: () =>
    set(() => ({
      messages: [],
    })),

  setPhase0Result: (data) =>
    set((state) => ({
      analysisResults: {
        ...state.analysisResults,
        phase0: data,
      },
    })),

  setPhase1Result: (data) =>
    set((state) => ({
      analysisResults: {
        ...state.analysisResults,
        phase1: data,
      },
    })),

  resetStore: () =>
    set(() => ({
      ...initialAnalysisState,
    })),

  // -------------------------------------------------------------------------
  // 用户状态
  // -------------------------------------------------------------------------
  user: null,
  isAuthenticated: false,

  setUser: (user) =>
    set(() => ({
      user,
      isAuthenticated: !!user,
    })),

  logout: () => {
    localStorage.removeItem('token');
    set(() => ({
      user: null,
      isAuthenticated: false,
    }));
  },

  // -------------------------------------------------------------------------
  // 任务状态
  // -------------------------------------------------------------------------
  tasks: [],
  currentTask: null,

  setTasks: (tasks) =>
    set(() => ({
      tasks,
    })),

  setCurrentTask: (task) =>
    set(() => ({
      currentTask: task,
    })),

  updateTask: (taskId, updates) =>
    set((state) => ({
      tasks: state.tasks.map((task) =>
        task.id === taskId ? { ...task, ...updates } : task
      ),
      currentTask:
        state.currentTask?.id === taskId
          ? { ...state.currentTask, ...updates }
          : state.currentTask,
    })),

  // -------------------------------------------------------------------------
  // UI 状态
  // -------------------------------------------------------------------------
  sidebarOpen: true,
  toggleSidebar: () =>
    set((state) => ({
      sidebarOpen: !state.sidebarOpen,
    })),

  loading: false,
  setLoading: (loading) =>
    set(() => ({
      loading,
    })),
}));
