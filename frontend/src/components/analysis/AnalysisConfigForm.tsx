import { useState } from 'react';
import { Play, Settings, Database, Target } from 'lucide-react';

// ============================================================================
// AnalysisConfigForm 组件 - 分析配置表单
// ============================================================================

interface AnalysisConfig {
  disease_name: string;
  data_path: string;
  target_column: string;
  max_candidates: number;
  use_cache: boolean;
  max_steps: number;
}

interface AnalysisConfigFormProps {
  onStartAnalysis: (config: AnalysisConfig) => void;
  isRunning: boolean;
  isConnected: boolean;
}

export const AnalysisConfigForm = ({ onStartAnalysis, isRunning, isConnected }: AnalysisConfigFormProps) => {
  const [config, setConfig] = useState<AnalysisConfig>({
    disease_name: "Crohn's Disease",
    data_path: "../data/test_data2_for_agent.csv",
    target_column: "Group",
    max_candidates: 50,
    use_cache: true,
    max_steps: 100,
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    // 检查 WebSocket 连接状态
    if (!isConnected) {
      alert('⚠️ WebSocket 未连接\n\n请确保后端服务已启动：\ncd backend && bash start.sh');
      return;
    }
    
    onStartAnalysis(config);
  };

  return (
    <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-200">
      <div className="flex items-center gap-3 mb-6">
        <div className="p-2 bg-blue-100 rounded-lg">
          <Settings size={24} className="text-blue-600" />
        </div>
        <div>
          <h2 className="text-xl font-bold text-gray-900">分析配置</h2>
          <p className="text-sm text-gray-500">配置 Phase 0 和 Phase 1 分析参数</p>
        </div>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Phase 0 配置 */}
        <div className="space-y-4">
          <div className="flex items-center gap-2 text-sm font-semibold text-gray-700 mb-3">
            <Database size={16} className="text-blue-600" />
            <span>Phase 0: 先验知识提取</span>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              疾病名称
            </label>
            <input
              type="text"
              value={config.disease_name}
              onChange={(e) => setConfig({ ...config, disease_name: e.target.value })}
              disabled={isRunning}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-100 disabled:cursor-not-allowed"
              placeholder="例如：Crohn's Disease"
            />
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                最大候选数
              </label>
              <input
                type="number"
                value={config.max_candidates}
                onChange={(e) => setConfig({ ...config, max_candidates: parseInt(e.target.value) })}
                disabled={isRunning}
                min={10}
                max={200}
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-100"
              />
            </div>

            <div>
              <label className="flex items-center gap-2 text-sm font-medium text-gray-700 mb-2">
                <input
                  type="checkbox"
                  checked={config.use_cache}
                  onChange={(e) => setConfig({ ...config, use_cache: e.target.checked })}
                  disabled={isRunning}
                  className="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
                />
                使用缓存
              </label>
              <p className="text-xs text-gray-500 mt-1">
                启用后将使用之前的 Phase 0 结果（如果存在）
              </p>
            </div>
          </div>
        </div>

        {/* Phase 1 配置 */}
        <div className="space-y-4 pt-4 border-t border-gray-200">
          <div className="flex items-center gap-2 text-sm font-semibold text-gray-700 mb-3">
            <Target size={16} className="text-green-600" />
            <span>Phase 1: 数据分析与建模</span>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              数据文件路径
            </label>
            <input
              type="text"
              value={config.data_path}
              onChange={(e) => setConfig({ ...config, data_path: e.target.value })}
              disabled={isRunning}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-100 font-mono text-sm"
              placeholder="../data/test_data2_for_agent.csv"
            />
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                目标列名
              </label>
              <input
                type="text"
                value={config.target_column}
                onChange={(e) => setConfig({ ...config, target_column: e.target.value })}
                disabled={isRunning}
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-100"
                placeholder="Group"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                最大步骤数
              </label>
              <input
                type="number"
                value={config.max_steps}
                onChange={(e) => setConfig({ ...config, max_steps: parseInt(e.target.value) })}
                disabled={isRunning}
                min={10}
                max={200}
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-100"
              />
            </div>
          </div>
        </div>

        {/* 提交按钮 */}
        <button
          type="submit"
          disabled={isRunning || !isConnected}
          className={`w-full py-3 px-6 font-semibold rounded-lg shadow-lg transition-all duration-200 flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed ${
            !isConnected
              ? 'bg-gradient-to-r from-red-500 to-red-600 text-white'
              : isRunning
              ? 'bg-gradient-to-r from-gray-400 to-gray-500 text-white'
              : 'bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 text-white'
          }`}
        >
          {!isConnected ? (
            <>
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
              </svg>
              <span>后端未连接</span>
            </>
          ) : isRunning ? (
            <>
              <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
              <span>分析进行中...</span>
            </>
          ) : (
            <>
              <Play size={20} />
              <span>开始分析</span>
            </>
          )}
        </button>

        {/* 提示信息 */}
        {!isConnected ? (
          <div className="text-xs text-red-600 text-center bg-red-50 p-2 rounded border border-red-200">
            ⚠️ 后端服务未启动，请运行：<code className="bg-red-100 px-1 rounded">cd backend && bash start.sh</code>
          </div>
        ) : !isRunning ? (
          <div className="text-xs text-gray-500 text-center">
            点击"开始分析"将执行完整的 Phase 0 + Phase 1 流程
          </div>
        ) : null}
      </form>
    </div>
  );
};
