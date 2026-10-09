import { useMemo } from 'react';
import ReactECharts from 'echarts-for-react';
import { useAppStore } from '../../store/useAppStore';
import { BarChart3 } from 'lucide-react';

// ============================================================================
// Mock 数据 - 用于预览效果
// ============================================================================

const MOCK_LEADERBOARD = [
  {
    model: 'WeightedEnsemble_L2',
    score_val: 0.9245,
    pred_time_val: 0.0234,
    accuracy: 0.9156,
  },
  {
    model: 'LightGBM_BAG_L1',
    score_val: 0.9123,
    pred_time_val: 0.0189,
    accuracy: 0.9034,
  },
  {
    model: 'XGBoost_BAG_L1',
    score_val: 0.9087,
    pred_time_val: 0.0256,
    accuracy: 0.8998,
  },
  {
    model: 'CatBoost_BAG_L1',
    score_val: 0.9012,
    pred_time_val: 0.0312,
    accuracy: 0.8923,
  },
  {
    model: 'RandomForest_BAG_L1',
    score_val: 0.8956,
    pred_time_val: 0.0145,
    accuracy: 0.8867,
  },
  {
    model: 'NeuralNetTorch_BAG_L1',
    score_val: 0.8834,
    pred_time_val: 0.0423,
    accuracy: 0.8745,
  },
  {
    model: 'ExtraTrees_BAG_L1',
    score_val: 0.8789,
    pred_time_val: 0.0167,
    accuracy: 0.8701,
  },
];

// ============================================================================
// ChartsPanel 组件 - 模型性能可视化
// ============================================================================

export const ChartsPanel = () => {
  // 从 Store 获取分析结果
  const { analysisResults } = useAppStore();

  // 获取 leaderboard 数据，如果没有则使用 Mock 数据
  const leaderboardData = useMemo(() => {
    const realData = analysisResults.phase1?.training?.leaderboard;
    return realData && realData.length > 0 ? realData : MOCK_LEADERBOARD;
  }, [analysisResults]);

  // 准备图表数据
  const chartData = useMemo(() => {
    // 按 score_val 降序排序
    const sortedData = [...leaderboardData].sort(
      (a, b) => b.score_val - a.score_val
    );

    return {
      modelNames: sortedData.map((item) => item.model),
      scores: sortedData.map((item) => item.score_val),
      accuracies: sortedData.map((item) => ('accuracy' in item ? item.accuracy : item.score_val) || item.score_val),
      predTimes: sortedData.map((item) => item.pred_time_val),
    };
  }, [leaderboardData]);

  // ECharts 配置
  const chartOption = useMemo(() => {
    return {
      title: {
        text: '模型性能天梯榜 (AutoGluon)',
        left: 'center',
        top: 10,
        textStyle: {
          fontSize: 16,
          fontWeight: 'bold',
          color: '#1f2937',
        },
      },
      tooltip: {
        trigger: 'axis',
        axisPointer: {
          type: 'shadow',
        },
        formatter: (params: any) => {
          const dataIndex = params[0].dataIndex;
          const modelName = chartData.modelNames[dataIndex];
          const rocAuc = chartData.scores[dataIndex];
          const accuracy = chartData.accuracies[dataIndex];
          const predTime = chartData.predTimes[dataIndex];

          return `
            <div style="padding: 8px;">
              <div style="font-weight: bold; margin-bottom: 8px;">${modelName}</div>
              <div style="margin-bottom: 4px;">
                <span style="color: #3b82f6;">● ROC-AUC:</span> ${rocAuc.toFixed(4)}
              </div>
              <div style="margin-bottom: 4px;">
                <span style="color: #10b981;">● Accuracy:</span> ${accuracy.toFixed(4)}
              </div>
              <div>
                <span style="color: #f59e0b;">● Pred Time:</span> ${predTime.toFixed(4)}s
              </div>
            </div>
          `;
        },
      },
      grid: {
        left: '20%',
        right: '10%',
        top: 60,
        bottom: 40,
        containLabel: true,
      },
      xAxis: {
        type: 'value',
        name: 'ROC-AUC Score',
        nameLocation: 'middle',
        nameGap: 30,
        nameTextStyle: {
          fontSize: 12,
          color: '#6b7280',
        },
        min: 0,
        max: 1,
        axisLabel: {
          formatter: '{value}',
          color: '#6b7280',
        },
        splitLine: {
          lineStyle: {
            color: '#e5e7eb',
          },
        },
      },
      yAxis: {
        type: 'category',
        data: chartData.modelNames,
        axisLabel: {
          color: '#374151',
          fontSize: 11,
          formatter: (value: string) => {
            // 缩短过长的模型名称
            return value.length > 25 ? value.substring(0, 22) + '...' : value;
          },
        },
        axisTick: {
          show: false,
        },
        axisLine: {
          lineStyle: {
            color: '#e5e7eb',
          },
        },
      },
      series: [
        {
          name: 'ROC-AUC',
          type: 'bar',
          data: chartData.scores,
          itemStyle: {
            color: {
              type: 'linear',
              x: 0,
              y: 0,
              x2: 1,
              y2: 0,
              colorStops: [
                { offset: 0, color: '#3b82f6' },
                { offset: 1, color: '#60a5fa' },
              ],
            },
            borderRadius: [0, 4, 4, 0],
          },
          label: {
            show: true,
            position: 'right',
            formatter: '{c}',
            fontSize: 11,
            color: '#374151',
          },
          barMaxWidth: 30,
        },
      ],
      animationDuration: 1000,
      animationEasing: 'cubicOut',
    };
  }, [chartData]);

  return (
    <div className="h-full flex flex-col bg-white rounded-lg shadow">
      {/* ====================================================================
          顶部标题栏
      ==================================================================== */}
      <div className="px-6 py-4 border-b border-gray-200">
        <div className="flex items-center gap-2">
          <BarChart3 size={20} className="text-blue-600" />
          <h3 className="text-lg font-semibold text-gray-900">
            模型性能对比
          </h3>
        </div>
        <p className="text-sm text-gray-500 mt-1">
          {leaderboardData === MOCK_LEADERBOARD
            ? '预览数据（Mock）'
            : `共 ${leaderboardData.length} 个模型`}
        </p>
      </div>

      {/* ====================================================================
          图表区域
      ==================================================================== */}
      <div className="flex-1 p-6">
        {leaderboardData.length > 0 ? (
          <ReactECharts
            option={chartOption}
            style={{ height: '100%', width: '100%' }}
            opts={{ renderer: 'canvas' }}
          />
        ) : (
          <div className="h-full flex items-center justify-center text-gray-500">
            <div className="text-center">
              <BarChart3 size={48} className="mx-auto mb-4 opacity-50" />
              <p className="text-sm">暂无模型数据</p>
              <p className="text-xs mt-2">完成 Phase 1 分析后将显示结果</p>
            </div>
          </div>
        )}
      </div>

      {/* ====================================================================
          底部统计信息
      ==================================================================== */}
      {leaderboardData.length > 0 && (
        <div className="px-6 py-4 border-t border-gray-200 bg-gray-50">
          <div className="grid grid-cols-3 gap-4 text-center">
            <div>
              <div className="text-xs text-gray-600">最佳模型</div>
              <div className="text-sm font-semibold text-gray-900 mt-1 truncate">
                {chartData.modelNames[0]}
              </div>
            </div>
            <div>
              <div className="text-xs text-gray-600">最高 ROC-AUC</div>
              <div className="text-sm font-semibold text-blue-600 mt-1">
                {chartData.scores[0].toFixed(4)}
              </div>
            </div>
            <div>
              <div className="text-xs text-gray-600">平均预测时间</div>
              <div className="text-sm font-semibold text-orange-600 mt-1">
                {(
                  chartData.predTimes.reduce((a, b) => a + b, 0) /
                  chartData.predTimes.length
                ).toFixed(4)}
                s
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
