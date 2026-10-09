import { Upload, Activity, BarChart3, FileText } from 'lucide-react';

export const HomePage = () => {
  return (
    <div className="max-w-6xl mx-auto">
      <div className="mb-8">
        <h2 className="text-3xl font-bold text-gray-900 mb-2">
          欢迎使用 MetaboAgent 平台
        </h2>
        <p className="text-gray-600">
          智能代谢组学分析平台，基于 AI Agent 的自动化分析流程
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
        <div className="card hover:shadow-lg transition-shadow cursor-pointer">
          <Upload className="text-primary-600 mb-4" size={32} />
          <h3 className="text-xl font-semibold mb-2">上传数据</h3>
          <p className="text-gray-600">
            上传代谢组学数据文件，支持 CSV、Excel 等格式
          </p>
        </div>

        <div className="card hover:shadow-lg transition-shadow cursor-pointer">
          <Activity className="text-primary-600 mb-4" size={32} />
          <h3 className="text-xl font-semibold mb-2">自动分析</h3>
          <p className="text-gray-600">
            AI Agent 自动执行差异分析、通路富集等分析流程
          </p>
        </div>

        <div className="card hover:shadow-lg transition-shadow cursor-pointer">
          <BarChart3 className="text-primary-600 mb-4" size={32} />
          <h3 className="text-xl font-semibold mb-2">可视化结果</h3>
          <p className="text-gray-600">
            交互式图表展示分析结果，支持多种可视化方式
          </p>
        </div>
      </div>

      <div className="card">
        <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
          <FileText size={24} />
          快速开始
        </h3>
        <ol className="space-y-3 text-gray-700">
          <li className="flex items-start gap-2">
            <span className="font-semibold text-primary-600">1.</span>
            <span>准备您的代谢组学数据文件（CSV 或 Excel 格式）</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="font-semibold text-primary-600">2.</span>
            <span>在"分析任务"页面上传数据并配置分析参数</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="font-semibold text-primary-600">3.</span>
            <span>启动分析任务，实时查看分析进度</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="font-semibold text-primary-600">4.</span>
            <span>在"结果查看"页面浏览和下载分析结果</span>
          </li>
        </ol>
      </div>
    </div>
  );
};
