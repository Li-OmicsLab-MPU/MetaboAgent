import { useState } from 'react';
import { ChatConsole } from '../components/chat/ChatConsole';
import { LogViewer } from '../components/workspace/LogViewer';
import { StepProgressTimeline } from '../components/workspace/StepProgressIndicator';
import { ContextVariablesPanel } from '../components/workspace/ContextVariablesPanel';
import { AnalysisConfigForm } from '../components/analysis/AnalysisConfigForm';
import { Phase0ResultsPanel } from '../components/results/Phase0ResultsPanel';
import { useWebSocket } from '../hooks/useWebSocket';
import { useAppStore } from '../store/useAppStore';

export const MainLayout = () => {
  const [activeTab, setActiveTab] = useState<'phase0' | 'phase1' | 'logs'>('phase0');
  
  const { agentState, analysisResults } = useAppStore();
  
  // WebSocket 连接
  const { sendMessage, isConnected } = useWebSocket({
    url: 'ws://localhost:8000/ws/analyze',
    autoConnect: true,
  });

  const handleStartAnalysis = (config: any) => {
    // 发送配置到后端
    sendMessage({
      action: 'start_analysis',
      disease_name: config.disease_name,
      data_path: config.data_path,
      target_column: config.target_column,
      max_candidates: config.max_candidates,
      use_cache: config.use_cache,
      max_steps: config.max_steps,
    });
  };

  const isRunning = agentState === 'phase0_running' || agentState === 'phase1_running';

  return (
    <div className="flex h-screen overflow-hidden bg-gradient-to-br from-gray-50 via-blue-50 to-purple-50">
      {/* 左侧：配置和进度 - 30% */}
      <aside className="w-[30%] p-6 overflow-y-auto border-r border-gray-200 bg-white/50 backdrop-blur-sm">
        {/* Logo 和标题 */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 via-purple-600 to-pink-600 bg-clip-text text-transparent mb-2">
            MetaboAgent
          </h1>
          <p className="text-gray-600 text-sm">代谢组学智能分析平台</p>
          <div className="flex items-center gap-2 mt-2">
            <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'} animate-pulse`} />
            <span className="text-xs text-gray-500">
              {isConnected ? 'WebSocket 已连接' : 'WebSocket 未连接'}
            </span>
          </div>
        </div>

        {/* 分析配置表单 */}
        <div className="mb-6">
          <AnalysisConfigForm
            onStartAnalysis={handleStartAnalysis}
            isRunning={isRunning}
            isConnected={isConnected}
          />
        </div>

        {/* 执行进度 */}
        {isRunning && (
          <div className="space-y-4">
            <StepProgressTimeline />
            <ContextVariablesPanel />
          </div>
        )}
      </aside>

      {/* 中间：聊天控制台 - 35% */}
      <main 
        className="w-[35%] flex flex-col overflow-hidden bg-white border-r border-gray-200"
        style={{ width: '35%', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}
      >
        <ChatConsole />
      </main>

      {/* 右侧：结果展示 - 35% */}
      <section className="w-[35%] flex flex-col bg-white/50 backdrop-blur-sm">
        {/* 标签页切换 */}
        <div className="px-6 py-4 border-b border-gray-200 bg-white">
          <div className="flex gap-2">
            <TabButton
              active={activeTab === 'phase0'}
              onClick={() => setActiveTab('phase0')}
              label="Phase 0 结果"
            />
            <TabButton
              active={activeTab === 'phase1'}
              onClick={() => setActiveTab('phase1')}
              label="Phase 1 结果"
            />
            <TabButton
              active={activeTab === 'logs'}
              onClick={() => setActiveTab('logs')}
              label="执行日志"
            />
          </div>
        </div>

        {/* 内容区域 - 固定高度，不允许无限增长 */}
        <div className="flex-1 p-6 overflow-hidden">
          {activeTab === 'phase0' && (
            <div className="h-full overflow-y-auto">
              <Phase0ResultsPanel result={analysisResults.phase0 || null} />
            </div>
          )}
          {activeTab === 'phase1' && (
            <div className="h-full overflow-y-auto">
              <Phase1ResultsPlaceholder result={analysisResults.phase1 || null} />
            </div>
          )}
          {activeTab === 'logs' && (
            <div className="h-full flex items-start justify-center">
              <LogViewer />
            </div>
          )}
        </div>
      </section>
    </div>
  );
};

// ============================================================================
// TabButton 组件 - 标签页按钮
// ============================================================================

interface TabButtonProps {
  active: boolean;
  onClick: () => void;
  label: string;
}

const TabButton = ({ active, onClick, label }: TabButtonProps) => {
  return (
    <button
      onClick={onClick}
      className={`px-4 py-2 rounded-lg font-medium text-sm transition-all duration-200 ${
        active
          ? 'bg-gradient-to-r from-blue-600 to-purple-600 text-white shadow-lg transform scale-105'
          : 'bg-gray-100 text-gray-600 hover:bg-gray-200 hover:text-gray-900'
      }`}
    >
      {label}
    </button>
  );
};

// ============================================================================
// Phase1ResultsPlaceholder 组件 - Phase 1 结果占位符
// ============================================================================

interface Phase1ResultsPlaceholderProps {
  result: any;
}

const Phase1ResultsPlaceholder = ({ result }: Phase1ResultsPlaceholderProps) => {
  if (!result) {
    return (
      <div className="bg-white rounded-xl shadow-lg p-12 border border-gray-200 text-center">
        <div className="w-16 h-16 mx-auto mb-4 bg-gradient-to-br from-purple-100 to-pink-100 rounded-full flex items-center justify-center">
          <svg className="w-8 h-8 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
          </svg>
        </div>
        <h3 className="text-lg font-semibold text-gray-900 mb-2">等待 Phase 1 结果</h3>
        <p className="text-sm text-gray-500">Phase 1 分析完成后，结果将在此处显示</p>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-200">
      <h3 className="text-lg font-bold text-gray-900 mb-4">Phase 1 结果</h3>
      <div className="space-y-4">
        <div className="p-4 bg-purple-50 rounded-lg">
          <div className="text-sm text-gray-600 mb-1">特征选择方法</div>
          <div className="text-xl font-bold text-purple-600">
            {result.feature_selection?.strategy || 'N/A'}
          </div>
        </div>
        <div className="p-4 bg-green-50 rounded-lg">
          <div className="text-sm text-gray-600 mb-1">最佳模型</div>
          <div className="text-xl font-bold text-green-600">
            {result.training?.best_model || 'N/A'}
          </div>
        </div>
      </div>
    </div>
  );
};
