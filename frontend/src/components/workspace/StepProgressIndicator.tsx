import { CheckCircle2, Circle, Loader2 } from 'lucide-react';
import { useAppStore } from '../../store/useAppStore';

// ============================================================================
// StepProgressIndicator 组件 - Step 级别进度指示器
// ============================================================================

export const StepProgressIndicator = () => {
  const { 
    agentState, 
    currentStageId, 
    currentStepId, 
    currentStepName 
  } = useAppStore();

  // 判断是否正在运行
  const isRunning = agentState === 'phase1_running';

  if (!isRunning || !currentStageId) {
    return null;
  }

  return (
    <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
      <div className="flex items-start gap-3">
        <Loader2 size={20} className="animate-spin text-blue-600 mt-0.5 shrink-0" />
        
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <span className="text-sm font-semibold text-blue-900">
              Stage {currentStageId}
            </span>
            {currentStepId && (
              <>
                <span className="text-blue-400">›</span>
                <span className="text-sm font-medium text-blue-700">
                  Step {currentStepId}
                </span>
              </>
            )}
          </div>
          
          {currentStepName && (
            <div className="text-xs text-blue-700 mt-1">
              {currentStepName}
            </div>
          )}
          
          {/* 进度条（可选） */}
          <div className="mt-3">
            <div className="h-1.5 bg-blue-200 rounded-full overflow-hidden">
              <div 
                className="h-full bg-blue-600 rounded-full transition-all duration-500 animate-pulse"
                style={{ width: '60%' }}
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

// ============================================================================
// StepProgressTimeline 组件 - Step 时间线（从 Store 读取真实数据）
// ============================================================================

export const StepProgressTimeline = () => {
  const { 
    agentState, 
    currentStageId, 
    currentStepId,
    stepHistory,
  } = useAppStore();

  const isRunning = agentState === 'phase0_running' || agentState === 'phase1_running';

  if (!isRunning && stepHistory.length === 0) {
    return null;
  }

  // 按 Stage 分组 steps
  const stageMap = new Map<string, Array<{
    stepId: string;
    stepName: string;
    status: 'running' | 'completed' | 'failed';
    timestamp: string;
  }>>();

  stepHistory.forEach((step) => {
    if (!stageMap.has(step.stageId)) {
      stageMap.set(step.stageId, []);
    }
    stageMap.get(step.stageId)!.push({
      stepId: step.stepId,
      stepName: step.stepName,
      status: step.status,
      timestamp: step.timestamp,
    });
  });

  // 转换为数组并排序
  const stages = Array.from(stageMap.entries())
    .map(([stageId, steps]) => ({
      stageId,
      steps: steps.sort((a, b) => a.stepId.localeCompare(b.stepId)),
    }))
    .sort((a, b) => a.stageId.localeCompare(b.stageId));

  return (
    <div className="p-4 bg-white border border-gray-200 rounded-lg">
      <h3 className="text-sm font-semibold text-gray-900 mb-4">执行进度</h3>
      
      {stages.length === 0 && (
        <div className="text-sm text-gray-500 text-center py-4">
          等待执行开始...
        </div>
      )}
      
      <div className="space-y-4">
        {stages.map((stage) => {
          const isCurrentStage = stage.stageId === currentStageId;
          const stageCompleted = stage.steps.every(s => s.status === 'completed');
          const stageHasRunning = stage.steps.some(s => s.status === 'running');
          
          return (
            <div key={stage.stageId} className="relative">
              {/* Stage 标题 */}
              <div className="flex items-center gap-2 mb-2">
                {stageCompleted ? (
                  <CheckCircle2 size={16} className="text-green-600" />
                ) : stageHasRunning ? (
                  <Loader2 size={16} className="animate-spin text-blue-600" />
                ) : (
                  <Circle size={16} className="text-gray-400" />
                )}
                
                <span className={`text-sm font-medium ${
                  isCurrentStage ? 'text-blue-900' : 
                  stageCompleted ? 'text-green-900' : 
                  'text-gray-600'
                }`}>
                  Stage {stage.stageId}
                </span>
              </div>
              
              {/* Steps 列表 */}
              <div className="ml-6 pl-4 border-l-2 border-gray-200 space-y-2">
                {stage.steps.map((step) => {
                  const isCurrentStep = step.stepId === currentStepId;
                  
                  return (
                    <div 
                      key={step.stepId}
                      className={`flex items-center gap-2 text-xs ${
                        isCurrentStep ? 'text-blue-700 font-medium' : 
                        step.status === 'completed' ? 'text-green-700' :
                        step.status === 'failed' ? 'text-red-700' :
                        'text-gray-500'
                      }`}
                    >
                      {step.status === 'completed' && (
                        <CheckCircle2 size={12} className="text-green-600" />
                      )}
                      {step.status === 'running' && (
                        <Loader2 size={12} className="animate-spin text-blue-600" />
                      )}
                      {step.status === 'failed' && (
                        <Circle size={12} className="text-red-600" />
                      )}
                      
                      <span>Step {step.stepId}: {step.stepName}</span>
                    </div>
                  );
                })}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
