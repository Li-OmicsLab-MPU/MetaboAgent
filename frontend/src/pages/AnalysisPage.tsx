import { useState } from 'react';
import { Upload, Play, FileText } from 'lucide-react';
import { apiService } from '../services/api';
import { LoadingSpinner } from '../components/LoadingSpinner';
import { useAppStore } from '../store/useAppStore';

export const AnalysisPage = () => {
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const { setLoading } = useAppStore();

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleUpload = async () => {
    if (!file) return;

    setUploading(true);
    try {
      const response = await apiService.uploadFile(
        '/api/upload',
        file,
        setUploadProgress
      );
      console.log('Upload success:', response);
      alert('文件上传成功！');
    } catch (error) {
      console.error('Upload failed:', error);
      alert('文件上传失败');
    } finally {
      setUploading(false);
      setUploadProgress(0);
    }
  };

  const handleStartAnalysis = async () => {
    setLoading(true);
    try {
      const response = await apiService.post('/api/analysis/start', {
        phase: 'phase0',
        // 其他参数...
      });
      console.log('Analysis started:', response);
      alert('分析任务已启动！');
    } catch (error) {
      console.error('Failed to start analysis:', error);
      alert('启动分析失败');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto">
      <h2 className="text-3xl font-bold text-gray-900 mb-6">分析任务</h2>

      {/* 文件上传 */}
      <div className="card mb-6">
        <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
          <Upload size={24} />
          上传数据文件
        </h3>
        
        <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center">
          <input
            type="file"
            id="file-upload"
            className="hidden"
            accept=".csv,.xlsx,.xls"
            onChange={handleFileChange}
          />
          <label
            htmlFor="file-upload"
            className="cursor-pointer flex flex-col items-center"
          >
            <FileText size={48} className="text-gray-400 mb-4" />
            <span className="text-gray-600 mb-2">
              点击选择文件或拖拽文件到此处
            </span>
            <span className="text-sm text-gray-500">
              支持 CSV、Excel 格式
            </span>
          </label>
        </div>

        {file && (
          <div className="mt-4 p-4 bg-gray-50 rounded-lg">
            <p className="text-sm text-gray-700">
              已选择文件: <span className="font-medium">{file.name}</span>
            </p>
            <p className="text-sm text-gray-500">
              大小: {(file.size / 1024).toFixed(2)} KB
            </p>
          </div>
        )}

        {uploading && (
          <div className="mt-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-gray-600">上传进度</span>
              <span className="text-sm font-medium">{uploadProgress}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div
                className="bg-primary-600 h-2 rounded-full transition-all"
                style={{ width: `${uploadProgress}%` }}
              />
            </div>
          </div>
        )}

        <button
          onClick={handleUpload}
          disabled={!file || uploading}
          className="btn-primary mt-4 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {uploading ? <LoadingSpinner size={20} /> : '上传文件'}
        </button>
      </div>

      {/* 分析配置 */}
      <div className="card mb-6">
        <h3 className="text-xl font-semibold mb-4">分析配置</h3>
        
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              分析阶段
            </label>
            <select className="input-field">
              <option value="phase0">Phase 0 - 差异分析</option>
              <option value="phase1">Phase 1 - 机器学习</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              分组列名
            </label>
            <input
              type="text"
              className="input-field"
              placeholder="例如: Group"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              显著性阈值 (p-value)
            </label>
            <input
              type="number"
              className="input-field"
              placeholder="0.05"
              step="0.01"
              min="0"
              max="1"
            />
          </div>
        </div>
      </div>

      {/* 启动分析 */}
      <button
        onClick={handleStartAnalysis}
        className="btn-primary w-full flex items-center justify-center gap-2"
      >
        <Play size={20} />
        启动分析
      </button>
    </div>
  );
};
