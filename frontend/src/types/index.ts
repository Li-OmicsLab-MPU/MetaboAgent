// ============================================================================
// MetaboAgent Frontend TypeScript 接口定义
// ============================================================================

// ----------------------------------------------------------------------------
// 通用类型
// ----------------------------------------------------------------------------

export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  message?: string;
  error?: string;
}

export interface User {
  id: string;
  username: string;
  email: string;
  role: 'admin' | 'user';
}

// ----------------------------------------------------------------------------
// Agent 状态类型
// ----------------------------------------------------------------------------

export type AgentState =
  | 'idle'
  | 'phase0_running'
  | 'phase1_running'
  | 'completed'
  | 'error'
  | 'paused'
  | 'cancelled';

// 详细的 Agent 状态（包含 Step 级别信息）
export interface DetailedAgentState {
  phase: 'phase0' | 'phase1';
  status: 'idle' | 'running' | 'completed' | 'error';
  current_stage_id?: string;  // Phase 1 专用，如 "0", "1", "5", "6"
  current_step_id?: string;    // Phase 1 专用，如 "1.5.1", "5.2.1"
  current_step_name?: string;  // 当前步骤名称
  progress_percentage?: number; // 进度百分比
}

// Context 变量（Phase 1 动态变量）
export interface ContextVariables {
  is_balanced?: boolean;              // 数据是否平衡
  run_ml_pipeline?: boolean;          // 是否运行 ML 流程
  run_traditional_stats?: boolean;    // 是否运行传统统计
  feature_selection_method?: string;  // 特征选择方法
  protected_columns?: string[];       // 受保护的列
  [key: string]: any;                 // 其他动态变量
}

// 数据摘要（Phase 1 数据信息）
export interface DataSummary {
  n_rows: number;                     // 行数
  n_cols: number;                     // 列数
  columns: string[];                  // 列名列表
  target_column?: string;             // 目标列名
  missing_values?: Record<string, number>; // 缺失值统计
}

// ----------------------------------------------------------------------------
// 日志消息类型
// ----------------------------------------------------------------------------

export interface LogMessage {
  id: string;
  timestamp: string;
  type: 'stdout' | 'stderr' | 'system' | 'warning';
  content: string;
}

// ----------------------------------------------------------------------------
// 聊天消息类型
// ----------------------------------------------------------------------------

export interface ChatMessage {
  id: string;
  timestamp: string;
  role: 'user' | 'agent' | 'system';
  content: string;
  file?: {
    name: string;
    size: number;
  };
}

// ----------------------------------------------------------------------------
// Phase 0 相关接口
// ----------------------------------------------------------------------------

export interface ConfirmedBiomarker {
  name: string;
  hmdb_id: string;
  confidence_score: number;
  fold_change?: number;
  p_value?: number;
  adjusted_p_value?: number;
}

export interface Phase0Result {
  disease_name: string;
  confirmed_biomarkers: ConfirmedBiomarker[];
  target_pathways: string[];
  differential_metabolites_count?: number;
  enrichment_results?: EnrichmentResult[];
  analysis_strategy?: string;
  execution_time?: number;
}

export interface EnrichmentResult {
  pathway_id: string;
  pathway_name: string;
  p_value: number;
  adjusted_p_value: number;
  matched_metabolites: string[];
  pathway_size: number;
  overlap_size: number;
}

// ----------------------------------------------------------------------------
// 数据质量报告接口
// ----------------------------------------------------------------------------

export interface DataQualityReport {
  n_rows: number;
  n_cols: number;
  missing_rate: number;
  outliers_detected: number;
  duplicate_rows?: number;
  column_types?: Record<string, string>;
  summary_statistics?: Record<string, any>;
}

// ----------------------------------------------------------------------------
// Phase 1 相关接口
// ----------------------------------------------------------------------------

export interface FeatureSelectionResult {
  strategy: string;
  n_features_selected: number;
  selected_features: string[];
  consensus_strategy?: string;
  feature_importance_scores?: Record<string, number>;
  selection_methods_used?: string[];
  execution_time?: number;
}

export interface LeaderboardEntry {
  model: string;
  score_val: number;
  pred_time_val: number;
  fit_time?: number;
  pred_time_test?: number;
  stack_level?: number;
}

export interface TrainingResult {
  best_model: string;
  accuracy: number;
  roc_auc: number;
  training_time: number;
  leaderboard: LeaderboardEntry[];
  best_model_score?: number;
  n_models_trained?: number;
  cross_validation_scores?: number[];
  feature_importance?: Record<string, number>;
}

export interface Phase1Result {
  feature_selection: FeatureSelectionResult;
  training: TrainingResult;
  model_path?: string;
  predictions?: number[];
  test_accuracy?: number;
  confusion_matrix?: number[][];
}

// ----------------------------------------------------------------------------
// 分析任务接口
// ----------------------------------------------------------------------------

export interface AnalysisTask {
  id: string;
  name: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  progress: number;
  current_phase?: 'phase0' | 'phase1';
  current_step?: string;
  createdAt: string;
  updatedAt: string;
  startedAt?: string;
  completedAt?: string;
  error_message?: string;
}

export interface AnalysisConfig {
  disease_name?: string;
  group_column: string;
  control_group?: string;
  case_group?: string;
  p_value_threshold?: number;
  fold_change_threshold?: number;
  feature_selection_strategy?: string;
  model_preset?: string;
  time_limit?: number;
}

export interface AnalysisResult {
  taskId: string;
  phase: 'phase0' | 'phase1';
  phase0_result?: Phase0Result;
  phase1_result?: Phase1Result;
  data_quality?: DataQualityReport;
  logs: LogMessage[];
  visualizations?: Visualization[];
  timestamp: string;
  execution_time?: number;
}

// ----------------------------------------------------------------------------
// 可视化接口
// ----------------------------------------------------------------------------

export interface Visualization {
  id: string;
  type: 'volcano_plot' | 'pathway_bar' | 'heatmap' | 'roc_curve' | 'feature_importance' | 'confusion_matrix';
  title: string;
  data: any;
  config?: Record<string, any>;
}

// ----------------------------------------------------------------------------
// 代谢物数据接口
// ----------------------------------------------------------------------------

export interface MetaboliteData {
  id: string;
  name: string;
  hmdb_id?: string;
  formula?: string;
  mass?: number;
  fold_change?: number;
  p_value?: number;
  adjusted_p_value?: number;
  group_mean_control?: number;
  group_mean_case?: number;
  [key: string]: any;
}

// ----------------------------------------------------------------------------
// WebSocket 消息接口
// ----------------------------------------------------------------------------

export interface WebSocketMessage {
  type:
    | 'progress'
    | 'result'
    | 'error'
    | 'log'
    | 'log_message'
    | 'progress_update'
    | 'state_change'
    | 'step_progress'
    | 'context_update'
    | 'data_summary'
    | 'phase0_result'
    | 'phase1_result'
    | 'phase2_result'
    | 'phase_result'
    | 'agent_state'
    | 'job_state'
    | 'job_completed'
    | 'job_config';
  data: any;
  timestamp?: string;
  task_id?: string;
  [key: string]: any;
}

export interface ProgressMessage {
  task_id: string;
  phase: 'phase0' | 'phase1';
  step: string;
  progress: number;
  message: string;
  timestamp: string;
}

export interface StateChangeMessage {
  task_id: string;
  old_state: AgentState;
  new_state: AgentState;
  timestamp: string;
}

// Step 进度消息
export interface StepProgressMessage {
  type: 'step_progress';
  stage_id: string;           // Stage ID，如 "0", "1", "5", "6"
  step_id: string;            // Step ID，如 "1.5.1", "5.2.1"
  step_name: string;          // Step 名称
  status: 'running' | 'completed' | 'failed';
  timestamp: string;
}

// Context 变量更新消息
export interface ContextUpdateMessage {
  type: 'context_update';
  variables: ContextVariables;
  timestamp: string;
}

// 数据摘要消息
export interface DataSummaryMessage {
  type: 'data_summary';
  summary: DataSummary;
  timestamp: string;
}

// ----------------------------------------------------------------------------
// 文件上传接口
// ----------------------------------------------------------------------------

export interface UploadedFile {
  id: string;
  filename: string;
  size: number;
  mime_type: string;
  upload_time: string;
  status: 'uploaded' | 'processing' | 'ready' | 'error';
}

export interface FileValidationResult {
  valid: boolean;
  errors?: string[];
  warnings?: string[];
  preview?: {
    columns: string[];
    rows: any[][];
    total_rows: number;
  };
}

// ----------------------------------------------------------------------------
// 历史记录接口
// ----------------------------------------------------------------------------

export interface AnalysisHistory {
  id: string;
  task_name: string;
  disease_name?: string;
  status: 'completed' | 'failed';
  phase0_completed: boolean;
  phase1_completed: boolean;
  n_biomarkers?: number;
  n_pathways?: number;
  best_model?: string;
  best_accuracy?: number;
  created_at: string;
  completed_at?: string;
  execution_time?: number;
}

// ----------------------------------------------------------------------------
// 导出汇总类型
// ----------------------------------------------------------------------------

export interface ExportData {
  task_info: AnalysisTask;
  config: AnalysisConfig;
  results: AnalysisResult;
  export_time: string;
}
