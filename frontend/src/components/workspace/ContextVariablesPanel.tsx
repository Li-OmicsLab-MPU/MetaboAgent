import { Settings, CheckCircle2, XCircle, Info } from 'lucide-react';
import { useAppStore } from '../../store/useAppStore';

// ============================================================================
// ContextVariablesPanel 组件 - Context 变量展示面板
// ============================================================================

export const ContextVariablesPanel = () => {
  const { contextVariables, agentState } = useAppStore();

  // 判断是否有 Context 变量
  const hasVariables = Object.keys(contextVariables).length > 0;
  const isRunning = agentState === 'phase1_running';

  if (!hasVariables && !isRunning) {
    return null;
  }

  return (
    <div className="p-4 bg-white border border-gray-200 rounded-lg">
      <div className="flex items-center gap-2 mb-4">
        <Settings size={16} className="text-gray-600" />
        <h3 className="text-sm font-semibold text-gray-900">Context 变量</h3>
        <span className="text-xs text-gray-500">
          (Phase 1 动态变量)
        </span>
      </div>

      {!hasVariables ? (
        <div className="text-xs text-gray-500 text-center py-4">
          等待 Context 变量更新...
        </div>
      ) : (
        <div className="space-y-3">
          {/* is_balanced */}
          {contextVariables.is_balanced !== undefined && (
            <ContextVariableItem
              label="数据平衡性"
              value={contextVariables.is_balanced}
              type="boolean"
              description="数据集中各组样本数量是否平衡"
            />
          )}

          {/* run_ml_pipeline */}
          {contextVariables.run_ml_pipeline !== undefined && (
            <ContextVariableItem
              label="ML 流程"
              value={contextVariables.run_ml_pipeline}
              type="boolean"
              description="是否执行机器学习流程"
            />
          )}

          {/* run_traditional_stats */}
          {contextVariables.run_traditional_stats !== undefined && (
            <ContextVariableItem
              label="传统统计"
              value={contextVariables.run_traditional_stats}
              type="boolean"
              description="是否执行传统统计分析"
            />
          )}

          {/* feature_selection_method */}
          {contextVariables.feature_selection_method && (
            <ContextVariableItem
              label="特征选择方法"
              value={contextVariables.feature_selection_method}
              type="string"
              description="当前使用的特征选择算法"
            />
          )}

          {/* protected_columns */}
          {contextVariables.protected_columns && (
            <ContextVariableItem
              label="受保护列"
              value={contextVariables.protected_columns}
              type="array"
              description="不参与特征选择的列"
            />
          )}

          {/* 其他动态变量 */}
          {Object.entries(contextVariables)
            .filter(([key]) => 
              !['is_balanced', 'run_ml_pipeline', 'run_traditional_stats', 
                'feature_selection_method', 'protected_columns'].includes(key)
            )
            .map(([key, value]) => (
              <ContextVariableItem
                key={key}
                label={key}
                value={value}
                type={typeof value}
              />
            ))}
        </div>
      )}
    </div>
  );
};

// ============================================================================
// ContextVariableItem 组件 - 单个 Context 变量展示
// ============================================================================

interface ContextVariableItemProps {
  label: string;
  value: any;
  type: 'boolean' | 'string' | 'number' | 'array' | string;
  description?: string;
}

const ContextVariableItem = ({ 
  label, 
  value, 
  type, 
  description 
}: ContextVariableItemProps) => {
  // 渲染值
  const renderValue = () => {
    if (type === 'boolean') {
      return (
        <div className="flex items-center gap-2">
          {value ? (
            <>
              <CheckCircle2 size={14} className="text-green-600" />
              <span className="text-sm font-medium text-green-700">True</span>
            </>
          ) : (
            <>
              <XCircle size={14} className="text-red-600" />
              <span className="text-sm font-medium text-red-700">False</span>
            </>
          )}
        </div>
      );
    }

    if (type === 'array') {
      return (
        <div className="text-sm text-gray-700">
          [{Array.isArray(value) ? value.join(', ') : String(value)}]
        </div>
      );
    }

    if (type === 'string') {
      return (
        <div className="text-sm font-mono text-blue-700 bg-blue-50 px-2 py-1 rounded">
          {String(value)}
        </div>
      );
    }

    return (
      <div className="text-sm text-gray-700">
        {String(value)}
      </div>
    );
  };

  return (
    <div className="p-3 bg-gray-50 rounded-lg border border-gray-200">
      <div className="flex items-start justify-between gap-2 mb-1">
        <div className="flex items-center gap-1.5">
          <span className="text-xs font-semibold text-gray-700">
            {label}
          </span>
          {description && (
            <div className="group relative">
              <Info size={12} className="text-gray-400 cursor-help" />
              <div className="absolute left-0 bottom-full mb-2 hidden group-hover:block z-10">
                <div className="bg-gray-900 text-white text-xs rounded px-2 py-1 whitespace-nowrap">
                  {description}
                </div>
              </div>
            </div>
          )}
        </div>
        <span className="text-xs text-gray-500 font-mono">
          {type}
        </span>
      </div>
      
      <div className="mt-2">
        {renderValue()}
      </div>
    </div>
  );
};

// ============================================================================
// ContextVariablesCompact 组件 - 紧凑版 Context 变量展示
// ============================================================================

export const ContextVariablesCompact = () => {
  const { contextVariables } = useAppStore();

  const hasVariables = Object.keys(contextVariables).length > 0;

  if (!hasVariables) {
    return null;
  }

  return (
    <div className="flex flex-wrap gap-2">
      {contextVariables.is_balanced !== undefined && (
        <span className={`inline-flex items-center gap-1 px-2 py-1 rounded text-xs font-medium ${
          contextVariables.is_balanced 
            ? 'bg-green-100 text-green-700' 
            : 'bg-red-100 text-red-700'
        }`}>
          {contextVariables.is_balanced ? '✓' : '✗'} 数据平衡
        </span>
      )}

      {contextVariables.run_ml_pipeline !== undefined && (
        <span className={`inline-flex items-center gap-1 px-2 py-1 rounded text-xs font-medium ${
          contextVariables.run_ml_pipeline 
            ? 'bg-blue-100 text-blue-700' 
            : 'bg-gray-100 text-gray-700'
        }`}>
          {contextVariables.run_ml_pipeline ? '✓' : '✗'} ML 流程
        </span>
      )}

      {contextVariables.feature_selection_method && (
        <span className="inline-flex items-center gap-1 px-2 py-1 rounded text-xs font-medium bg-purple-100 text-purple-700">
          特征选择: {contextVariables.feature_selection_method}
        </span>
      )}
    </div>
  );
};
