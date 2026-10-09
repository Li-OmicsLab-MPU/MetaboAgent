import { Database, TrendingUp, Activity, Clock } from 'lucide-react';
import { Phase0Result } from '../../types';

// ============================================================================
// Phase0ResultsPanel 组件 - Phase 0 结果展示
// ============================================================================

interface Phase0ResultsPanelProps {
  result: Phase0Result | null;
}

export const Phase0ResultsPanel = ({ result }: Phase0ResultsPanelProps) => {
  if (!result) {
    return (
      <div className="bg-white rounded-xl shadow-lg p-8 border border-gray-200 text-center">
        <Database size={48} className="mx-auto text-gray-400 mb-4" />
        <p className="text-gray-500">等待 Phase 0 结果...</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* 统计卡片 */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <StatCard
          icon={<Database size={24} className="text-blue-600" />}
          label="疾病"
          value={result.disease_name}
          bgColor="bg-blue-50"
        />
        <StatCard
          icon={<TrendingUp size={24} className="text-green-600" />}
          label="生物标志物"
          value={result.confirmed_biomarkers.length}
          bgColor="bg-green-50"
        />
        <StatCard
          icon={<Activity size={24} className="text-purple-600" />}
          label="富集通路"
          value={result.target_pathways.length}
          bgColor="bg-purple-50"
        />
        <StatCard
          icon={<Clock size={24} className="text-orange-600" />}
          label="执行时间"
          value={`${result.execution_time?.toFixed(1) || 0}s`}
          bgColor="bg-orange-50"
        />
      </div>

      {/* 生物标志物列表 */}
      <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-200">
        <h3 className="text-lg font-bold text-gray-900 mb-4 flex items-center gap-2">
          <TrendingUp size={20} className="text-green-600" />
          确认的生物标志物
        </h3>
        
        {result.confirmed_biomarkers.length > 0 ? (
          <div className="overflow-hidden border border-gray-200 rounded-lg">
            <table className="w-full text-sm">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-4 py-3 text-left font-semibold text-gray-700">#</th>
                  <th className="px-4 py-3 text-left font-semibold text-gray-700">代谢物名称</th>
                  <th className="px-4 py-3 text-left font-semibold text-gray-700">HMDB ID</th>
                  <th className="px-4 py-3 text-left font-semibold text-gray-700">置信度</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200">
                {result.confirmed_biomarkers.slice(0, 10).map((biomarker, index) => (
                  <tr key={index} className="hover:bg-gray-50 transition-colors">
                    <td className="px-4 py-3 text-gray-600">{index + 1}</td>
                    <td className="px-4 py-3 font-medium text-gray-900">{biomarker.name}</td>
                    <td className="px-4 py-3 text-gray-600 font-mono text-xs">{biomarker.hmdb_id}</td>
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-2">
                        <div className="flex-1 bg-gray-200 rounded-full h-2 max-w-[100px]">
                          <div
                            className="bg-green-600 h-2 rounded-full"
                            style={{ width: `${biomarker.confidence_score * 100}%` }}
                          />
                        </div>
                        <span className="text-xs text-gray-600">
                          {(biomarker.confidence_score * 100).toFixed(0)}%
                        </span>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="text-gray-500 text-center py-4">暂无生物标志物数据</p>
        )}
        
        {result.confirmed_biomarkers.length > 10 && (
          <p className="text-xs text-gray-500 mt-3 text-center">
            显示前 10 个，共 {result.confirmed_biomarkers.length} 个生物标志物
          </p>
        )}
      </div>

      {/* 富集通路列表 */}
      <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-200">
        <h3 className="text-lg font-bold text-gray-900 mb-4 flex items-center gap-2">
          <Activity size={20} className="text-purple-600" />
          富集通路（Top 10）
        </h3>
        
        {result.target_pathways.length > 0 ? (
          <div className="space-y-2">
            {result.target_pathways.slice(0, 10).map((pathway, index) => (
              <div
                key={index}
                className="flex items-center gap-3 p-3 bg-purple-50 rounded-lg border border-purple-200 hover:bg-purple-100 transition-colors"
              >
                <div className="flex-shrink-0 w-8 h-8 bg-purple-600 text-white rounded-full flex items-center justify-center font-bold text-sm">
                  {index + 1}
                </div>
                <div className="flex-1 text-sm text-gray-900 font-medium">
                  {pathway}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-gray-500 text-center py-4">暂无富集通路数据</p>
        )}
        
        {result.target_pathways.length > 10 && (
          <p className="text-xs text-gray-500 mt-3 text-center">
            显示前 10 条，共 {result.target_pathways.length} 条富集通路
          </p>
        )}
      </div>
    </div>
  );
};

// ============================================================================
// StatCard 组件 - 统计卡片
// ============================================================================

interface StatCardProps {
  icon: React.ReactNode;
  label: string;
  value: string | number;
  bgColor: string;
}

const StatCard = ({ icon, label, value, bgColor }: StatCardProps) => {
  return (
    <div className="bg-white rounded-xl shadow-lg p-4 border border-gray-200">
      <div className={`inline-flex p-2 rounded-lg ${bgColor} mb-3`}>
        {icon}
      </div>
      <div className="text-2xl font-bold text-gray-900 mb-1">
        {value}
      </div>
      <div className="text-sm text-gray-600">
        {label}
      </div>
    </div>
  );
};
