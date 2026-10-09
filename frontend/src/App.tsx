/* Dynamic result payloads are validated by the backend contract and rendered defensively below. */
/* eslint-disable @typescript-eslint/no-explicit-any */
import { useEffect, useMemo, useRef, useState } from 'react';
import {
  Activity,
  Archive,
  Atom,
  BarChart3,
  Beaker,
  CheckCircle2,
  ChevronRight,
  CircleDot,
  Clock,
  Database,
  Download,
  FileSpreadsheet,
  FileText,
  FlaskConical,
  Layers3,
  Loader2,
  Network,
  Play,
  RefreshCw,
  Settings,
  ShieldCheck,
  Upload,
  Workflow,
} from 'lucide-react';
import './App.css';
import { apiService } from './services/api';
import { JobStreamMessage, jobStream } from './services/jobStream';

type ColumnPreview = {
  name: string;
  dtype: string;
  missing_rate: number;
  unique_count: number;
  unique_values?: unknown[];
  suggested_role: 'id' | 'target' | 'feature' | 'metadata' | string;
};

type FileRecord = {
  file_id: string;
  filename: string;
  file_type: string;
  storage_path: string;
  preview: {
    n_rows: number;
    n_columns: number;
    columns: ColumnPreview[];
    head: Record<string, unknown>[];
  };
  demo_id?: string;
  is_demo?: boolean;
  demo_defaults?: {
    id_column?: string;
    group_column?: string;
    positive_class?: string;
    disease_name?: string;
  };
};

type DemoDataset = {
  id: string;
  name: string;
  description: string;
  filename: string;
  available: boolean;
  defaults?: FileRecord['demo_defaults'];
  preview?: { n_rows: number; n_columns: number };
};

type Scenario = {
  key?: string;
  name?: string;
  description?: string;
  intended_use?: string;
  clinical_action?: string;
  [key: string]: unknown;
};

type FigureStyle = {
  key: string;
  name: string;
  description: string;
};

type JobStatus = {
  job_id: string;
  status: 'queued' | 'running' | 'completed' | 'failed' | string;
  current_phase?: string;
  current_step?: string | null;
  current_step_name?: string | null;
  current_step_status?: 'running' | 'completed' | 'failed' | string | null;
  progress?: Record<string, number>;
  error?: string | null;
  stream_url?: string;
  results_url?: string;
  current_activity?: ActivityEvent;
  created_at?: string;
  started_at?: string | null;
  updated_at?: string;
  runtime_estimate?: RuntimeEstimateProfile;
};

type RuntimeEstimatePhase = {
  median_seconds: number;
  low_seconds: number;
  high_seconds: number;
  sample_size: number;
  source: 'historical' | 'baseline' | string;
};

type RuntimeEstimateProfile = {
  phases: Record<string, RuntimeEstimatePhase>;
  median_seconds: number;
  low_seconds: number;
  high_seconds: number;
  completed_run_count: number;
  method: string;
};

type Artifact = {
  name: string;
  phase: string;
  type: string;
  format: string;
  size_bytes?: number;
  download_url: string;
};

type LogEntry = {
  id?: string;
  timestamp?: string;
  level?: string;
  type?: string;
  phase?: string;
  content: string;
  category?: 'milestone' | 'technical' | 'error';
};

type StepProgress = {
  stageId: string;
  stepId: string;
  stepName: string;
  status: 'running' | 'completed' | 'failed' | string;
  timestamp: string;
  stepIndex?: number;
  stepTotal?: number;
  phaseProgress?: number;
};

type ActivityEvent = {
  id: string;
  phase: string;
  title: string;
  detail: string;
  status: 'running' | 'completed' | 'retrying' | 'failed' | string;
  timestamp: string;
  progress?: number;
  stepId?: string;
  stepIndex?: number;
  stepTotal?: number;
  elapsedSeconds?: number;
  startedAt?: string;
};

type StepKey = 'upload' | 'config' | 'run' | 'results';

const phaseOptions = [
  { key: 'phase0', label: 'Phase 0 Prior evidence retrieval', optional: true },
  { key: 'phase1', label: 'Phase 1 Data preprocessing and modeling', optional: false },
  { key: 'phase2', label: 'Phase 2 Biomarker panel optimization', optional: false },
  { key: 'phase3', label: 'Phase 3 Figures and report generation', optional: true },
];

const trackingPhases = [
  { key: 'phase0', index: '01', title: 'Prior evidence', detail: 'Retrieve disease-linked metabolites, biomarkers, and pathways.' },
  { key: 'phase1', index: '02', title: 'Data and baseline models', detail: 'Validate, preprocess, select stable features, and train baseline models.' },
  { key: 'phase2', index: '03', title: 'Panel optimization', detail: 'Search for a compact panel across predictive and biological objectives.' },
  { key: 'phase3', index: '04', title: 'Evidence outputs', detail: 'Render scientific figures and assemble the final report.' },
];

const phase1StepDetails: Record<string, string> = {
  '0.1': 'Validate the uploaded table and identify protected ID and outcome columns.',
  '0.2': 'Standardize metabolite identifiers and map recognized names to HMDB records.',
  '0.3': 'Confirm binary outcome semantics and preserve the selected positive-class direction.',
  '1.1': 'Assess missingness and apply the fixed missing-value policy.',
  '1.2': 'Normalize metabolite intensities to reduce systematic sample-level variation.',
  '1.3': 'Transform skewed features and audit potential outliers.',
  '1.4': 'Verify the integrity of the pre-engineering matrix.',
  '1.5': 'Lock the predefined machine-learning analysis strategy.',
  '1.5.1': 'Generate reaction-ratio features supported by metabolic relationships.',
  '1.5.2': 'Generate controlled metabolite-class aggregate features.',
  '1.5.3': 'Generate pathway scores when metabolite coverage is sufficient.',
  '1.5.4': 'Verify that engineered features were integrated without schema drift.',
  '1.5.5': 'Scale the merged matrix for downstream feature selection.',
  '2.1': 'Run the predefined univariate screening analysis.',
  '3.1': 'Audit unsupervised structure with PCA.',
  '3.2': 'Evaluate supervised structure with leakage-controlled PLS-DA.',
  '4.1': 'Reconcile statistical and model-based candidate features.',
  '5.0': 'Define the candidate universe and inspect class balance.',
  '5.2.1': 'Estimate feature-selection stability using training-only resampling.',
  '5.2.2': 'Audit stability scores and identify the stable core.',
  '5.2.3': 'Validate the stable feature-panel handoff.',
  '5.3': 'Create final training and holdout feature datasets.',
  '6.1': 'Inspect modeling readiness and choose an appropriate evaluation setup.',
  '6.2': 'Train and compare baseline models without holdout leakage.',
  '6.3': 'Save the selected model, leaderboard, and reproducibility artifacts.',
};

const languageLabels: Record<string, string> = {
  en_US: 'English (US)',
};

const scenarioEnglishLabels: Record<string, string> = {
  primary_care: 'Primary care screening',
  er_triage: 'Emergency room triage',
  companion_diagnostics: 'Companion diagnostics',
};

const figureOptions = [
  { key: 'roc', label: 'ROC curve' },
  { key: 'dca', label: 'DCA decision curve' },
  { key: 'calibration', label: 'Calibration curve' },
  { key: 'shap', label: 'SHAP explanation plot' },
  { key: 'rcs', label: 'RCS curve' },
  { key: 'phase2_radar', label: 'Phase 2 four-objective radar chart' },
  { key: 'phase0_prior_evidence_atlas', label: 'Phase 0 prior evidence atlas' },
  { key: 'phase1_stability_landscape', label: 'Phase 1 stability landscape' },
];

const fallbackRuntimeEstimate: RuntimeEstimateProfile = {
  phases: {
    phase0: { median_seconds: 20, low_seconds: 10, high_seconds: 45, sample_size: 0, source: 'baseline' },
    phase1: { median_seconds: 1200, low_seconds: 720, high_seconds: 2400, sample_size: 0, source: 'baseline' },
    phase2: { median_seconds: 650, low_seconds: 420, high_seconds: 1200, sample_size: 0, source: 'baseline' },
    phase3: { median_seconds: 45, low_seconds: 20, high_seconds: 120, sample_size: 0, source: 'baseline' },
  },
  median_seconds: 1915,
  low_seconds: 1170,
  high_seconds: 3765,
  completed_run_count: 0,
  method: 'baseline',
};

type RuntimeEstimateView = {
  lowSeconds: number;
  highSeconds: number;
  elapsedSeconds: number;
  signalAgeSeconds: number;
  finishWindow: string;
  confidenceLabel: string;
  signalLabel: string;
};

function calculateRuntimeEstimate(
  job: JobStatus | null,
  events: ActivityEvent[],
  selectedPhases: string[],
  progressByPhase: Record<string, number>,
  now: number,
): RuntimeEstimateView | null {
  if (!job || job.status === 'completed' || job.status === 'failed') return null;
  const profile = job.runtime_estimate || fallbackRuntimeEstimate;
  const phaseKeys = trackingPhases.map((phase) => phase.key).filter((phase) => selectedPhases.includes(phase));
  const currentPhase = phaseKeys.find((phase) => String(job.current_phase || '').includes(phase))
    || events[events.length - 1]?.phase;
  const runStartValue = job.started_at || job.created_at || events[0]?.startedAt || events[0]?.timestamp;
  const runStart = runStartValue ? Date.parse(runStartValue) : now;
  const elapsedSeconds = Math.max(0, Math.round((now - (Number.isNaN(runStart) ? now : runStart)) / 1000));
  const currentPhaseEvents = events.filter((event) => event.phase === currentPhase);
  const phaseStartCandidates = currentPhaseEvents
    .map((event) => Date.parse(event.startedAt || event.timestamp))
    .filter((value) => !Number.isNaN(value));
  const phaseStart = phaseStartCandidates.length ? Math.min(...phaseStartCandidates) : runStart;
  const phaseElapsed = Math.max(0, Math.round((now - phaseStart) / 1000));

  let lowSeconds = 0;
  let highSeconds = 0;
  for (const phase of phaseKeys) {
    const estimate = profile.phases?.[phase] || fallbackRuntimeEstimate.phases[phase];
    const phaseProgress = Math.max(0, Math.min(100, job.progress?.[phase] ?? progressByPhase[phase] ?? 0));
    if (phaseProgress >= 100) continue;
    const remainingFraction = Math.max(0.03, 1 - phaseProgress / 100);
    if (phase === currentPhase) {
      lowSeconds += Math.max(15, estimate.low_seconds * remainingFraction, estimate.low_seconds - phaseElapsed);
      highSeconds += Math.max(45, estimate.high_seconds * remainingFraction, estimate.high_seconds - phaseElapsed);
    } else {
      lowSeconds += estimate.low_seconds;
      highSeconds += estimate.high_seconds;
    }
  }

  const lastSignalValue = events[events.length - 1]?.timestamp || job.updated_at || runStartValue;
  const lastSignal = lastSignalValue ? Date.parse(lastSignalValue) : now;
  const signalAgeSeconds = Math.max(0, Math.round((now - (Number.isNaN(lastSignal) ? now : lastSignal)) / 1000));
  const signalLabel = signalAgeSeconds < 35
    ? `Live signal received ${formatRelativeSeconds(signalAgeSeconds)} ago`
    : signalAgeSeconds < 120
      ? 'Current operation is still computing; no intervention is needed'
      : 'No recent milestone — the connection monitor remains active';
  const finishLow = new Date(now + lowSeconds * 1000);
  const finishHigh = new Date(now + highSeconds * 1000);
  const finishWindow = `${formatClockTime(finishLow)}–${formatClockTime(finishHigh)}`;
  const sampleCount = profile.completed_run_count || Math.max(0, ...Object.values(profile.phases || {}).map((item) => item.sample_size || 0));
  const confidenceLabel = sampleCount >= 3
    ? `Adaptive range from ${sampleCount} completed runs`
    : 'Initial range; recalibrates as the workflow advances';

  return {
    lowSeconds: Math.max(0, Math.round(lowSeconds)),
    highSeconds: Math.max(0, Math.round(highSeconds)),
    elapsedSeconds,
    signalAgeSeconds,
    finishWindow,
    confidenceLabel,
    signalLabel,
  };
}

function App() {
  const [activeStep, setActiveStep] = useState<StepKey>('upload');
  const [uploading, setUploading] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [fileRecord, setFileRecord] = useState<FileRecord | null>(null);
  const [scenarios, setScenarios] = useState<Scenario[]>([]);
  const [figureStyles, setFigureStyles] = useState<FigureStyle[]>([]);
  const [demoDatasets, setDemoDatasets] = useState<DemoDataset[]>([]);
  const [selectedDemoId, setSelectedDemoId] = useState('');
  const [loadingDemo, setLoadingDemo] = useState(false);
  const [idColumn, setIdColumn] = useState('');
  const [groupColumn, setGroupColumn] = useState('');
  const [positiveClass, setPositiveClass] = useState('');
  const [diseaseName, setDiseaseName] = useState('');
  const [clinicalScenario, setClinicalScenario] = useState('');
  const [figureStyle, setFigureStyle] = useState('nature');
  const [language, setLanguage] = useState('en_US');
  const [selectedPhases, setSelectedPhases] = useState<string[]>(['phase0', 'phase1', 'phase2', 'phase3']);
  const [selectedFigures, setSelectedFigures] = useState<Record<string, boolean>>(
    Object.fromEntries(figureOptions.map((item) => [item.key, true]))
  );
  const [job, setJob] = useState<JobStatus | null>(null);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [results, setResults] = useState<any | null>(null);
  const [artifacts, setArtifacts] = useState<Artifact[]>([]);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState('');
  const [requiredFieldErrors, setRequiredFieldErrors] = useState<string[]>([]);
  const [streamStatus, setStreamStatus] = useState<'idle' | 'connecting' | 'connected' | 'reconnecting' | 'closed' | 'error'>('idle');
  const [phaseProgress, setPhaseProgress] = useState<Record<string, number>>({ phase0: 0, phase1: 0, phase2: 0, phase3: 0 });
  const [, setStepProgress] = useState<StepProgress | null>(null);
  const [activityEvents, setActivityEvents] = useState<ActivityEvent[]>([]);
  const [logLevelFilter, setLogLevelFilter] = useState('all');
  const [logPhaseFilter, setLogPhaseFilter] = useState('all');
  const [clockNow, setClockNow] = useState(Date.now());
  const completedJobsRef = useRef(new Set<string>());

  const columns = fileRecord?.preview.columns ?? [];
  const groupValues = useMemo(() => {
    if (!fileRecord || !groupColumn) return [];
    const values = new Set<string>();
    const targetColumn = fileRecord.preview.columns.find((column) => column.name === groupColumn);
    for (const value of targetColumn?.unique_values || []) {
      if (value !== undefined && value !== null && String(value).trim()) values.add(String(value));
    }
    for (const row of fileRecord.preview.head) {
      const value = row[groupColumn];
      if (value !== undefined && value !== null && String(value).trim()) values.add(String(value));
    }
    return Array.from(values).sort((left, right) => left.localeCompare(right, undefined, { numeric: true }));
  }, [fileRecord, groupColumn]);

  const scenarioLabels = useMemo(() => Object.fromEntries(scenarios.map((item) => {
    const key = String(item.key || item.name || '');
    return [key, scenarioEnglishLabels[key] || humanizeKey(key)];
  })), [scenarios]);
  const figureStyleLabels = useMemo(() => Object.fromEntries(figureStyles.map((item) => [item.key, item.name || humanizeKey(item.key)])), [figureStyles]);

  useEffect(() => {
    loadConfig();
    return () => jobStream.disconnect();
  }, []);

  useEffect(() => {
    setClockNow(Date.now());
    if (!running) return;
    const timer = window.setInterval(() => setClockNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [running]);

  useEffect(() => {
    if (!fileRecord) return;
    const idGuess = fileRecord.preview.columns.find((col) => col.suggested_role === 'id')?.name;
    const targetGuess = fileRecord.preview.columns.find((col) => col.suggested_role === 'target')?.name;
    const defaults = fileRecord.demo_defaults || {};
    setIdColumn(defaults.id_column || idGuess || fileRecord.preview.columns[0]?.name || '');
    setGroupColumn(defaults.group_column || targetGuess || fileRecord.preview.columns[1]?.name || '');
    setDiseaseName(defaults.disease_name || inferDiseaseName(fileRecord.filename));
    setPositiveClass(defaults.positive_class || '');
    setRequiredFieldErrors([]);
  }, [fileRecord]);

  useEffect(() => {
    if (groupValues.length !== 2) {
      setPositiveClass((current) => current && !groupValues.includes(current) ? '' : current);
      return;
    }
    const findPreferred = (candidates: string[]) => groupValues.find((value) => candidates.includes(value.trim().toLowerCase()));
    const inferredPositive = findPreferred(['1', 'case', 'disease', 'positive', 'yes', 'true']) || groupValues[1];
    setPositiveClass(inferredPositive);
  }, [groupValues]);

  const complementaryClass = useMemo(() => {
    if (groupValues.length !== 2 || !positiveClass) return '';
    return groupValues.find((value) => value !== positiveClass) || '';
  }, [groupValues, positiveClass]);

  const loadConfig = async () => {
    try {
      const [scenarioPayload, stylePayload, demoPayload] = await Promise.all([
        apiService.get<{ scenarios?: Scenario[]; active_scenario?: string }>('/api/v1/config/scenarios'),
        apiService.get<{ styles?: FigureStyle[]; default_style?: string }>('/api/v1/config/figure-styles'),
        apiService.get<{ datasets?: DemoDataset[] }>('/api/v1/config/demo-datasets'),
      ]);
      setScenarios(scenarioPayload.scenarios || []);
      setClinicalScenario(scenarioPayload.active_scenario || scenarioPayload.scenarios?.[0]?.key || '');
      setFigureStyles(stylePayload.styles || []);
      setFigureStyle(stylePayload.default_style || 'nature');
      const availableDemos = (demoPayload.datasets || []).filter((item) => item.available);
      setDemoDatasets(availableDemos);
      setSelectedDemoId((current) => current || availableDemos[0]?.id || '');
    } catch (err) {
      console.warn('Configuration loading failed; using defaults.', err);
    }
  };

  const loadDemoDataset = async () => {
    if (!selectedDemoId) return;
    setLoadingDemo(true);
    setError('');
    try {
      const payload = await apiService.post<FileRecord>(`/api/v1/demo-datasets/${selectedDemoId}/load`, {});
      setFileRecord(payload);
      setSelectedFile(null);
      setActiveStep('config');
    } catch (err) {
      setError(sanitizeTrackingText(err instanceof Error ? err.message : 'Demo dataset could not be loaded.'));
    } finally {
      setLoadingDemo(false);
    }
  };

  const uploadFile = async () => {
    if (!selectedFile) return;
    setUploading(true);
    setError('');
    try {
      const payload = await apiService.uploadFile<FileRecord>('/api/v1/files/upload', selectedFile);
      setFileRecord(payload);
      setActiveStep('config');
    } catch (err) {
      setError(sanitizeTrackingText(err instanceof Error ? err.message : 'File upload failed.'));
    } finally {
      setUploading(false);
    }
  };

  const togglePhase = (phase: string) => {
    setSelectedPhases((prev) => {
      if (phase === 'phase1' || phase === 'phase2') return prev;
      return prev.includes(phase) ? prev.filter((item) => item !== phase) : [...prev, phase];
    });
  };

  const createAndRunJob = async () => {
    const missing: string[] = [];
    if (!fileRecord) missing.push('Upload or load a dataset');
    if (!idColumn) missing.push('Sample ID column');
    if (!groupColumn) missing.push('Group / outcome column');
    if (selectedPhases.includes('phase0') && !diseaseName.trim()) missing.push('Disease name / clinical question for Phase 0');
    setRequiredFieldErrors(missing);
    if (missing.length) {
      setError('Complete the highlighted required fields before starting the analysis.');
      setActiveStep('config');
      return;
    }
    if (!fileRecord) return;
    setError('');
    setLogs([]);
    setResults(null);
    setArtifacts([]);
    setPhaseProgress({ phase0: 0, phase1: 0, phase2: 0, phase3: 0 });
    setStepProgress(null);
    setActivityEvents([]);
    setStreamStatus('connecting');
    setRunning(true);
    try {
      const currentFileRecord = fileRecord;
      const phases = Array.from(new Set([...selectedPhases, 'phase1', 'phase2']));
      const created = await apiService.post<JobStatus>('/api/v1/jobs', {
          file_id: currentFileRecord.file_id,
          dataset: {
            id_column: idColumn,
            group_column: groupColumn,
            positive_class: positiveClass || undefined,
          },
          analysis: {
            disease_name: diseaseName.trim(),
            clinical_scenario: clinicalScenario || undefined,
            run_phase0_prior_search: selectedPhases.includes('phase0'),
            phases,
          },
          phase0: {
            max_candidates: 50,
            use_cache: true,
          },
          phase1: {
            max_steps: 100,
          },
          phase2: {
            beam_width: 1,
            k_folds: 5,
            epsilon: 0.005,
            patience: 5,
          },
          phase3: {
            figure_style: figureStyle,
            language,
            output_formats: ['pdf', 'png', 'json', 'html'],
            figures: selectedFigures,
          },
          runtime: {
            memory_enabled: true,
            random_seed: 42,
            save_intermediate: true,
          },
      });
      setJob(created);
      setActiveStep('run');
      connectJobStream(created.job_id);
    } catch (err) {
      setRunning(false);
      setError(sanitizeTrackingText(err instanceof Error ? err.message : 'Task creation failed.'));
    }
  };

  const recordActivity = (event: ActivityEvent) => {
    setActivityEvents((previous) => {
      const identity = event.stepId ? `${event.phase}:${event.stepId}` : `${event.phase}:${event.title}`;
      const existingIndex = previous.findIndex((item) => (
        item.stepId ? `${item.phase}:${item.stepId}` : `${item.phase}:${item.title}`
      ) === identity);
      if (existingIndex >= 0) {
        const next = [...previous];
        next[existingIndex] = {
          ...next[existingIndex],
          ...event,
          id: next[existingIndex].id,
          startedAt: next[existingIndex].startedAt || next[existingIndex].timestamp,
        };
        return next;
      }
      return [...previous, { ...event, startedAt: event.startedAt || event.timestamp }].slice(-80);
    });
  };

  useEffect(() => {
    if (!job?.job_id || !['queued', 'running'].includes(job.status)) return;
    let cancelled = false;
    const pollJobState = async () => {
      try {
      const snapshot = await apiService.get<JobStatus>(`/api/v1/jobs/${job.job_id}`);
        if (cancelled) return;
        setJob(snapshot);
        if (snapshot.progress) setPhaseProgress((previous) => ({ ...previous, ...snapshot.progress }));
        if (snapshot.current_activity) {
          const activity = snapshot.current_activity as ActivityEvent & {
            step_id?: string;
            step_index?: number;
            step_total?: number;
            elapsed_seconds?: number;
          };
          const activityStepId = activity.stepId || activity.step_id;
          recordActivity({
            id: `snapshot_${activity.phase}_${activityStepId || activity.title}`,
            phase: activity.phase || snapshot.current_phase || 'runtime',
            title: sanitizeTrackingText(String(activity.title || snapshot.current_step_name || 'Pipeline activity')),
            detail: sanitizeTrackingText(String(activity.detail || 'The server-side workflow is continuing.')),
            status: activity.status || snapshot.current_step_status || snapshot.status,
            timestamp: activity.timestamp || snapshot.updated_at || new Date().toISOString(),
            progress: snapshot.progress?.[activity.phase],
            stepId: activityStepId,
            stepIndex: activity.stepIndex || activity.step_index,
            stepTotal: activity.stepTotal || activity.step_total,
            elapsedSeconds: activity.elapsedSeconds || activity.elapsed_seconds,
          });
        }
        try {
          const logSnapshot = await apiService.get<{ logs?: LogEntry[] }>(`/api/v1/jobs/${job.job_id}/logs`);
          const incomingLogs = logSnapshot.logs;
          if (!cancelled && Array.isArray(incomingLogs)) {
            setLogs((previous) => {
              const logKey = (entry: LogEntry) => entry.id || [entry.timestamp, entry.level || entry.type, entry.phase, entry.content].join('|');
              const known = new Set(previous.map(logKey));
              const additions = incomingLogs
                .filter((entry) => entry?.content && !known.has(logKey(entry)))
                .map((entry) => {
                  const content = sanitizeTrackingText(entry.content);
                  const level = entry.type || entry.level || 'info';
                  const category = level === 'stderr' || level === 'error'
                    ? 'error'
                    : /complete|ready|created|connected|result|winner/i.test(content)
                      ? 'milestone'
                      : 'technical';
                  return { ...entry, content, category } as LogEntry;
                });
              return additions.length ? [...previous, ...additions].slice(-500) : previous;
            });
          }
        } catch {
          // Status polling remains useful even if a transient log request fails.
        }
        if (snapshot.status === 'completed') {
          await completeJob(snapshot.job_id);
        } else if (snapshot.status === 'failed') {
          setRunning(false);
          setError(sanitizeTrackingText(snapshot.error || 'The analysis task failed.'));
        }
      } catch {
        // The live stream owns transient connectivity errors. Polling retries silently.
      }
    };
    void pollJobState();
    const timer = window.setInterval(pollJobState, 5000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [job?.job_id, job?.status]);

  const connectJobStream = (jobId: string) => {
    jobStream.connect(jobId, {
      onMessage: async (message: JobStreamMessage) => {
      if (message.type === 'job_snapshot' && message.job) {
        const snapshot = message.job as JobStatus;
        setJob(snapshot);
        if (snapshot.status === 'completed') {
          await completeJob(snapshot.job_id);
        } else if (snapshot.status === 'failed') {
          setRunning(false);
          setError(sanitizeTrackingText(snapshot.error || 'The analysis task failed.'));
        }
      }
      if (message.type === 'job_state' || message.type === 'agent_state') {
        setJob((prev) => prev ? {
          ...prev,
          status: message.status || prev.status,
          current_phase: message.current_phase || message.state || prev.current_phase,
          current_step: message.current_step || prev.current_step,
          progress: message.progress || prev.progress,
        } : prev);
        if (message.progress) setPhaseProgress((prev) => ({ ...prev, ...message.progress }));
      }
      if (message.type === 'step_progress') {
        const nextStep: StepProgress = {
          stageId: String(message.stage_id || ''),
          stepId: String(message.step_id || message.current_step || ''),
          stepName: String(message.step_name || 'Phase 1 analysis step'),
          status: String(message.status || 'running'),
          timestamp: message.timestamp || new Date().toISOString(),
          stepIndex: Number(message.step_index || 0) || undefined,
          stepTotal: Number(message.step_total || 0) || undefined,
          phaseProgress: Number(message.phase_progress || 0) || undefined,
        };
        setStepProgress(nextStep);
        recordActivity({
          id: `step_${nextStep.stepId}_${nextStep.timestamp}`,
          phase: 'phase1',
          title: nextStep.stepName,
          detail: phase1StepDetails[nextStep.stepId] || 'Running the next operation in the fixed Phase 1 analysis protocol.',
          status: nextStep.status,
          timestamp: nextStep.timestamp,
          progress: nextStep.phaseProgress,
          stepId: nextStep.stepId,
          stepIndex: nextStep.stepIndex,
          stepTotal: nextStep.stepTotal,
        });
        if (nextStep.phaseProgress !== undefined) {
          setPhaseProgress((prev) => ({ ...prev, phase1: nextStep.phaseProgress || prev.phase1 }));
        }
        setJob((prev) => prev ? {
          ...prev,
          current_step: nextStep.stepId,
          current_step_name: nextStep.stepName,
          current_step_status: nextStep.status,
          current_phase: message.current_phase || prev.current_phase,
        } : prev);
      }
      if (message.type === 'activity_update') {
        const phase = String(message.phase || message.current_phase || 'runtime');
        const phaseValue = Number(message.phase_progress || 0) || undefined;
        const event: ActivityEvent = {
          id: `activity_${phase}_${String(message.step_id || message.title || Date.now())}`,
          phase,
          title: sanitizeTrackingText(String(message.title || 'Pipeline activity')),
          detail: sanitizeTrackingText(String(message.detail || 'The workflow is continuing.')),
          status: String(message.status || 'running'),
          timestamp: message.timestamp || new Date().toISOString(),
          progress: phaseValue,
          stepId: message.step_id ? String(message.step_id) : undefined,
          stepIndex: Number(message.step_index || 0) || undefined,
          stepTotal: Number(message.step_total || 0) || undefined,
          elapsedSeconds: Number(message.elapsed_seconds || 0) || undefined,
        };
        recordActivity(event);
        if (phaseValue !== undefined) {
          setPhaseProgress((prev) => ({ ...prev, [phase]: Math.max(prev[phase] || 0, phaseValue) }));
        }
        if (message.progress) {
          setJob((prev) => prev ? { ...prev, progress: message.progress } : prev);
        }
      }
      if (message.type === 'job_config') {
        addLog({ content: `Run configuration created: ${String(message.runtime_config_path || '')}`, type: 'system', timestamp: message.timestamp, category: 'milestone' });
      }
      if (message.type === 'log_message') {
        addLog({
          id: message.log?.id,
          timestamp: message.log?.timestamp || message.timestamp,
          type: message.log?.type || message.log?.level || 'system',
          content: message.log?.content || JSON.stringify(message),
          phase: String(message.log?.phase || message.phase || message.current_phase || ''),
        });
      }
      if (message.type?.endsWith('_result') || message.type === 'phase_result') {
        addLog({ content: `${formatPhaseName(String(message.phase || message.type).replace('_result', ''))} results are ready.`, type: 'system', timestamp: message.timestamp, category: 'milestone' });
      }
      if (message.type === 'job_completed') {
        recordActivity({ id: `completed_${Date.now()}`, phase: 'completed', title: 'Analysis complete', detail: 'All selected phases completed. Results and artifacts are ready.', status: 'completed', timestamp: message.timestamp || new Date().toISOString(), progress: 100 });
        addLog({ content: 'Analysis completed. Loading results and artifacts.', type: 'system', timestamp: message.timestamp, category: 'milestone' });
        await completeJob(jobId);
      }
      if (message.type === 'error') {
        const errorMessage = sanitizeTrackingText(typeof message.message === 'string' ? message.message : 'The analysis task failed.');
        setRunning(false);
        setJob((prev) => prev ? { ...prev, status: 'failed', current_phase: 'failed', error: errorMessage } : prev);
        addLog({ content: errorMessage, type: 'stderr', timestamp: message.timestamp });
        setError(errorMessage);
      }
      },
      onOpen: () => {
        setStreamStatus('connected');
        addLog({ content: 'Live task stream connected.', type: 'system', category: 'milestone' });
      },
      onError: (message) => {
        setStreamStatus('reconnecting');
        addLog({ content: `${message} Status polling remains active.`, type: 'warning', category: 'technical' });
      },
      onClose: () => {
        if (running && job?.status !== 'completed') {
          setStreamStatus('reconnecting');
          addLog({ content: 'The live stream paused. Server-side execution and status polling continue.', type: 'warning', category: 'technical' });
        } else {
          setStreamStatus('closed');
        }
      },
    });
  };

  const addLog = (log: LogEntry) => {
    const content = sanitizeTrackingText(log.content);
    const level = log.type || log.level || 'info';
    const category = log.category || (level === 'stderr' || level === 'error'
      ? 'error'
      : /complete|ready|created|connected|result|winner/i.test(content)
        ? 'milestone'
        : 'technical');
    setLogs((prev) => [...prev, {
      ...log,
      content,
      category,
      phase: log.phase || job?.current_phase || undefined,
      id: log.id || `log_${Date.now()}_${prev.length}`,
    }].slice(-500));
  };

  const loadJobOutputs = async (jobId = job?.job_id) => {
    if (!jobId) return;
    const [resultPayload, artifactPayload] = await Promise.all([
      apiService.get<Record<string, unknown>>(`/api/v1/jobs/${jobId}/results`),
      apiService.get<{ artifacts?: Artifact[] }>(`/api/v1/jobs/${jobId}/artifacts`),
    ]);
    setResults(resultPayload);
    setArtifacts(artifactPayload.artifacts || []);
  };

  const completeJob = async (jobId: string) => {
    if (!jobId || completedJobsRef.current.has(jobId)) return;
    completedJobsRef.current.add(jobId);
    setRunning(false);
    setJob((prev) => prev ? {
      ...prev,
      status: 'completed',
      current_phase: 'completed',
      progress: { ...(prev.progress || {}), overall: 100 },
    } : prev);
    setPhaseProgress({ phase0: 100, phase1: 100, phase2: 100, phase3: 100 });
    try {
      await loadJobOutputs(jobId);
    } catch (err) {
      setError(sanitizeTrackingText(err instanceof Error ? err.message : 'Analysis completed, but the result package could not be loaded.'));
    } finally {
      setActiveStep('results');
      jobStream.disconnect();
    }
  };

  const progress = job?.progress?.overall ?? (job?.status === 'completed' ? 100 : Math.round(
    Object.values(phaseProgress).reduce((sum, value) => sum + value, 0) / Object.keys(phaseProgress).length,
  ));
  const visibleLogs = logs.filter((log) =>
    (logLevelFilter === 'all' || (log.type || log.level || 'info') === logLevelFilter)
    && (logPhaseFilter === 'all' || log.phase === logPhaseFilter),
  );
  const selectedTrackingPhases = trackingPhases.filter((phase) => selectedPhases.includes(phase.key));
  const currentActivity = [...activityEvents].reverse()[0];
  const activityHistory = [...activityEvents].reverse().filter((event) => event.id !== currentActivity?.id).slice(0, 14);
  const completedActivityCount = activityEvents.filter((event) => event.status === 'completed').length;
  const runtimeEstimate = useMemo(
    () => calculateRuntimeEstimate(job, activityEvents, selectedPhases, phaseProgress, clockNow),
    [job, activityEvents, selectedPhases, phaseProgress, clockNow],
  );
  const runtimeEstimatePhases = job?.runtime_estimate?.phases || fallbackRuntimeEstimate.phases;

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark" aria-hidden="true">
            <span className="brand-orbit orbit-a" />
            <span className="brand-orbit orbit-b" />
            <span className="brand-core">M</span>
          </div>
          <div>
            <span className="brand-kicker">METABOLOMICS · AI</span>
            <h1>MetaboAgent</h1>
            <p>Evidence-guided discovery</p>
          </div>
        </div>

        <div className="sidebar-section-label">ANALYSIS WORKFLOW</div>
        <nav className="step-nav">
          <StepButton step="upload" activeStep={activeStep} icon={<Database size={17} />} index="01" title="Dataset intake" meta="Dataset intake" onClick={setActiveStep} done={!!fileRecord} />
          <StepButton step="config" activeStep={activeStep} icon={<Settings size={17} />} index="02" title="Study design" meta="Analysis protocol" onClick={setActiveStep} disabled={!fileRecord} done={!!job} />
          <StepButton step="run" activeStep={activeStep} icon={<Activity size={17} />} index="03" title="Execution trace" meta="Live provenance" onClick={setActiveStep} disabled={!job} done={job?.status === 'completed'} />
          <StepButton step="results" activeStep={activeStep} icon={<BarChart3 size={17} />} index="04" title="Evidence results" meta="Evidence workspace" onClick={setActiveStep} disabled={!results && !artifacts.length} />
        </nav>

        <div className="progress-card">
          <div className="progress-card-title"><Workflow size={15} /><span>PIPELINE STATUS</span></div>
          <div className="progress-label"><span>Overall progress</span><strong>{progress}%</strong></div>
          <div className="progress-bar"><div style={{ width: `${progress}%` }} /></div>
          <p>{job ? `${formatRunStatus(job.status)} · ${formatPhaseName(job.current_phase || 'queued')}` : 'No active run'}</p>
        </div>

        <div className="protocol-card">
          <div className="protocol-title"><ShieldCheck size={15} />REPRODUCIBILITY</div>
          <dl>
            <div><dt>Seed</dt><dd>42</dd></div>
            <div><dt>Memory</dt><dd>Enabled</dd></div>
            <div><dt>Artifacts</dt><dd>Versioned</dd></div>
          </dl>
        </div>
      </aside>

      <main className="main-panel">
        <header className="topbar">
          <div className="topbar-copy">
            <div className="workspace-kicker"><CircleDot size={13} /> METABOLIC DISCOVERY WORKSPACE <span>v1.0</span></div>
            <h2>Clinical Metabolomics Biomarker Discovery</h2>
            <p>Evidence-guided discovery and validation of clinically relevant metabolic biomarkers.</p>
            <div className="science-tags">
              <span><Atom size={13} />Metabolite-centric</span>
              <span><Network size={13} />Pathway-aware</span>
              <span><ShieldCheck size={13} />Reproducible</span>
            </div>
          </div>
          <div className="topbar-actions">
            <span className="system-status"><i /> Pipeline ready</span>
            <button className="secondary-button" onClick={loadConfig}><RefreshCw size={15} />Sync configuration</button>
          </div>
        </header>

        {error && <div className="error-banner">{error}</div>}

        {activeStep === 'upload' && (
          <section className="panel-grid two-columns upload-workspace">
            <div className="card upload-card">
              <div className="card-heading">
                <span className="section-number">01</span>
                <div><div className="card-kicker">DATASET INTAKE</div><h3><FileSpreadsheet size={19} />Upload metabolomics data</h3><p>Establish the data contract and perform an initial audit of structure, missingness, and variable roles.</p></div>
              </div>
              <label className="drop-zone">
                <input type="file" accept=".csv,.xlsx,.xls" onChange={(event) => setSelectedFile(event.target.files?.[0] || null)} />
                <div className="drop-icon"><Upload size={25} /></div>
                <small>CSV · XLSX · XLS</small>
                <strong>{selectedFile ? selectedFile.name : 'Choose a metabolomics data file'}</strong>
                <span>Click to choose a file. The system will inspect field types, missingness patterns, and the first 20 rows.</span>
              </label>
              <button className="primary-button" disabled={!selectedFile || uploading} onClick={uploadFile}>
                {uploading ? <Loader2 className="spin" size={18} /> : <Upload size={18} />}
                {uploading ? 'Establishing the data contract...' : 'Upload and audit dataset'}
              </button>
              <div className="data-contract">
                <span><ShieldCheck size={14} />Format integrity</span>
                <span><Layers3 size={14} />Variable-role detection</span>
                <span><Beaker size={14} />Metabolite field precheck</span>
              </div>
            </div>

            <PreviewCard fileRecord={fileRecord} />
          </section>
        )}

        {activeStep === 'upload' && demoDatasets.length > 0 && <section className="demo-dataset-card" aria-label="Demo dataset">
          <div className="demo-dataset-copy">
            <div className="card-kicker">DEMO DATASET</div>
            <strong>Run a complete example</strong>
            <p>Use the server-hosted liver-cancer example to explore the full workflow without uploading a local file.</p>
          </div>
          <div className="demo-dataset-controls">
            <label htmlFor="demo-dataset-select">Example dataset</label>
            <select id="demo-dataset-select" value={selectedDemoId} onChange={(event) => setSelectedDemoId(event.target.value)}>
              {demoDatasets.map((dataset) => <option key={dataset.id} value={dataset.id}>{dataset.name}{dataset.preview ? ` · ${dataset.preview.n_rows} samples` : ''}</option>)}
            </select>
            <button className="secondary-button" disabled={!selectedDemoId || loadingDemo} onClick={loadDemoDataset}>{loadingDemo ? <Loader2 className="spin" size={15} /> : <Database size={15} />}Load demo dataset</button>
          </div>
        </section>}

        {activeStep === 'config' && (
          <section className="panel-grid">
            <div className="card">
              <div className="card-heading compact"><span className="section-number">02A</span><div><div className="card-kicker">DATA CONTRACT</div><h3><Settings size={19} />Variable roles and group definition</h3><p>Define the sample identifier, outcome variable, and binary class label to keep the analysis semantics stable.</p></div></div>
              <div className="form-grid">
                <SelectField label="Sample ID column" badge="Required" error={requiredFieldErrors.includes('Sample ID column')} description="A unique identifier for each sample row." value={idColumn} onChange={setIdColumn} options={columns.map((col) => col.name)} />
                <SelectField label="Group / outcome column" badge="Required" error={requiredFieldErrors.includes('Group / outcome column')} description="The target variable used for supervised modeling." value={groupColumn} onChange={setGroupColumn} options={columns.map((col) => col.name)} />
                <SelectField label="Case / event group (positive class)" badge="Auto-detected" description="The positive-class direction used for ROC, DCA, and model evaluation; the complementary group is encoded as 0." value={positiveClass} onChange={setPositiveClass} options={groupValues} allowEmpty emptyLabel="Auto-detect" />
              </div>
              <div className="contract-note"><CircleDot size={15} /><span>Class candidates are derived from all unique values in the column, not only the first 20 preview rows. Detected <strong>{groupValues.length}</strong> groups: {groupValues.length ? groupValues.join(', ') : 'select an outcome column first'}.{positiveClass && complementaryClass ? <> The positive class is <strong>{positiveClass}</strong>; the complementary group <strong>{complementaryClass}</strong> is encoded as 0.</> : null}</span></div>
            </div>

            <div className="card">
              <div className="card-heading compact"><span className="section-number">02B</span><div><div className="card-kicker">ANALYSIS PROTOCOL</div><h3><Activity size={19} />Analysis protocol and evidence outputs</h3><p>Define the disease context, clinical setting, execution phases, and figure-output protocol.</p></div></div>
              <div className="form-grid">
                <TextField label="Disease name / clinical question" badge={selectedPhases.includes('phase0') ? 'Required for Phase 0' : 'Optional'} error={requiredFieldErrors.includes('Disease name / clinical question for Phase 0')} description={diseaseName ? 'Detected from the filename or confirmed by the user.' : 'Auto-filled only when the filename provides a reliable signal; otherwise left blank.'} placeholder="e.g. chronic fatigue syndrome" value={diseaseName} onChange={setDiseaseName} />
                <SelectField label="Clinical application setting" badge="Recommended" description="Influences the evaluation emphasis and decision-threshold interpretation." value={clinicalScenario} onChange={setClinicalScenario} options={scenarios.map((item) => String(item.key || item.name || ''))} optionLabels={scenarioLabels} />
              </div>

              <div className="option-block">
                <h4>Execution phases</h4>
                <p className="option-description">Phase 1 and Phase 2 form the minimum analysis loop and are always enabled. Prior evidence retrieval and report generation are optional.</p>
                <div className="chips">
                  {phaseOptions.map((phase) => (
                    <button key={phase.key} disabled={!phase.optional} className={selectedPhases.includes(phase.key) ? 'chip selected' : 'chip'} onClick={() => togglePhase(phase.key)}>
                      {phase.label}<small>{phase.optional ? 'Optional' : 'Fixed'}</small>
                    </button>
                  ))}
                </div>
              </div>

              {selectedPhases.includes('phase3') && <div className="output-protocol">
                <div className="output-protocol-heading"><div><span>REPORTING PROTOCOL</span><h4>Evidence output settings</h4></div><small>Phase 3 only</small></div>
                <div className="form-grid">
                  <SelectField label="Figure style" description="Unify figure layout, color, and journal-style conventions." value={figureStyle} onChange={setFigureStyle} options={(figureStyles.length ? figureStyles : [{ key: 'nature', name: 'Nature', description: '' }]).map((item) => item.key)} optionLabels={figureStyleLabels} />
                  <SelectField label="Report language" description="Controls report text, figure titles, and explanatory notes." value={language} onChange={setLanguage} options={['en_US']} optionLabels={languageLabels} />
                </div>
                <details className="figure-selector">
                  <summary><span>Select output figures</span><strong>{Object.values(selectedFigures).filter(Boolean).length} / {figureOptions.length} enabled</strong></summary>
                  <div className="chips">
                    {figureOptions.map((figure) => (
                      <button
                        key={figure.key}
                        className={selectedFigures[figure.key] ? 'chip selected' : 'chip'}
                        onClick={() => setSelectedFigures((prev) => ({ ...prev, [figure.key]: !prev[figure.key] }))}
                      >
                        {figure.label}
                      </button>
                    ))}
                  </div>
                </details>
              </div>}

              {requiredFieldErrors.length > 0 && <div className="required-warning" role="alert"><strong>Required fields need attention</strong><ul>{requiredFieldErrors.map((item) => <li key={item}>{item}</li>)}</ul></div>}
              <button className="primary-button wide" onClick={createAndRunJob} disabled={running}>
                <Play size={18} />Lock protocol and start analysis
              </button>
            </div>
          </section>
        )}

        {activeStep === 'run' && (
          <section className="panel-grid trace-workspace">
            <div className="card trace-overview-card">
              <div className="trace-topline">
                <div>
                  <div className="card-kicker">LIVE PROVENANCE</div>
                  <h3><Activity size={20} />Execution trace</h3>
                  <p>Follow the scientific workflow at a glance. Detailed runtime events remain available below.</p>
                </div>
                <span className={`stream-badge ${streamStatus}`}>{streamStatusLabel(streamStatus)}</span>
              </div>

              <div className="trace-hero">
                <div className="trace-progress-dial" style={{ background: `conic-gradient(#78b857 ${progress * 3.6}deg, #dfe9e4 0deg)` }}>
                  <div><strong>{progress}%</strong><span>overall</span></div>
                </div>
                <div className="trace-focus">
                  <span>{formatPhaseName(currentActivity?.phase || job?.current_phase || 'queued')} · {formatActivityStatus(currentActivity?.status || job?.status || 'queued')}</span>
                  <h4>{currentActivity?.title || sanitizeTrackingText(String(job?.current_step_name || 'Preparing the analysis workflow'))}</h4>
                  <p>{currentActivity?.detail || 'Waiting for the next structured update from the analysis service.'}</p>
                  <div className="trace-focus-meta">
                    {currentActivity?.stepId && <b>Protocol step {currentActivity.stepId}</b>}
                    {currentActivity?.stepIndex && currentActivity.stepTotal && <b>{currentActivity.stepIndex} of {currentActivity.stepTotal} operations</b>}
                    {currentActivity?.elapsedSeconds !== undefined && <b>Elapsed {formatElapsed(currentActivity.elapsedSeconds)}</b>}
                  </div>
                </div>
                <div className="trace-run-facts">
                  <div><span>Run state</span><strong>{formatRunStatus(job?.status || 'queued')}</strong></div>
                  <div><span>Completed operations</span><strong>{completedActivityCount}</strong></div>
                  <div><span>Run elapsed</span><strong>{runtimeEstimate ? formatElapsed(runtimeEstimate.elapsedSeconds) : formatRunStatus(job?.status || 'queued')}</strong></div>
                </div>
              </div>

              {runtimeEstimate && (
                <div className={`eta-strip ${runtimeEstimate.signalAgeSeconds > 120 ? 'quiet' : 'live'}`}>
                  <div className="eta-primary">
                    <span className="eta-icon"><Clock size={17} /></span>
                    <div><small>Estimated time remaining</small><strong>{formatDurationRange(runtimeEstimate.lowSeconds, runtimeEstimate.highSeconds)}</strong></div>
                  </div>
                  <div className="eta-fact"><small>Likely finish window</small><strong>{runtimeEstimate.finishWindow}</strong></div>
                  <div className="eta-health">
                    <span><i />{runtimeEstimate.signalLabel}</span>
                    <small>{runtimeEstimate.confidenceLabel}. Estimates are guidance, not a deadline.</small>
                  </div>
                </div>
              )}

              {job?.error && <div className="error-detail"><strong>Failure reason: </strong>{sanitizeTrackingText(job.error)}</div>}
              <div className="phase-track">
                {selectedTrackingPhases.map((phase) => {
                  const value = Math.max(0, Math.min(100, job?.progress?.[phase.key] ?? phaseProgress[phase.key] ?? 0));
                  const isCurrent = job?.current_phase?.includes(phase.key) || currentActivity?.phase === phase.key;
                  const status = job?.status === 'failed' && isCurrent ? 'failed' : value >= 100 ? 'completed' : isCurrent ? 'running' : 'waiting';
                  return (
                    <div className={`phase-card ${status}`} key={phase.key}>
                      <div className="phase-card-top"><span>{phase.index}</span><strong>{formatActivityStatus(status)}</strong></div>
                      <h4>{phase.title}</h4>
                      <p>{phase.detail}</p>
                      <div className="phase-mini-bar"><i style={{ width: `${value}%` }} /></div>
                      <small>
                        {value}% complete
                        {status === 'running' && runtimeEstimatePhases[phase.key] && ` · ${formatCompactDuration(runtimeEstimatePhases[phase.key].median_seconds * Math.max(.03, 1 - value / 100))} left`}
                        {status === 'waiting' && runtimeEstimatePhases[phase.key] && ` · ~${formatCompactDuration(runtimeEstimatePhases[phase.key].median_seconds)}`}
                      </small>
                    </div>
                  );
                })}
              </div>
            </div>

            <div className="trace-detail-grid">
              <div className="card activity-card">
                <div className="run-header">
                  <div><div className="card-kicker">SCIENTIFIC WORKFLOW</div><h3><Workflow size={19} />Workflow timeline</h3></div>
                  <span className="activity-count">{completedActivityCount} completed</span>
                </div>

                {currentActivity && (
                  <div className={`current-activity-panel ${currentActivity.status}`}>
                    <div className="current-activity-label"><span className="live-pulse" />NOW · {formatPhaseName(currentActivity.phase)}{currentActivity.stepId ? ` · STEP ${currentActivity.stepId}` : ''}</div>
                    <div className="current-activity-title"><strong>{currentActivity.title}</strong><span>{formatActivityStatus(currentActivity.status)}</span></div>
                    <p>{currentActivity.detail}</p>
                    <div className="current-activity-footer">
                      <span>Started {formatTimestamp(currentActivity.startedAt || currentActivity.timestamp)}</span>
                      {currentActivity.stepIndex && currentActivity.stepTotal && <span>Operation {currentActivity.stepIndex} of {currentActivity.stepTotal}</span>}
                      <span>{currentActivity.status === 'running' ? `Active for ${formatEventDuration(currentActivity, clockNow)}` : `Finished at ${formatTimestamp(currentActivity.timestamp)}`}</span>
                    </div>
                  </div>
                )}

                {activityHistory.length > 0 && <div className="timeline-section-label">Earlier milestones</div>}
                <div className="activity-list">
                  {!currentActivity && <div className="activity-empty"><Loader2 className="spin" size={18} /><span>Preparing the first workflow update.</span></div>}
                  {activityHistory.map((event) => (
                    <div className={`activity-item ${event.status}`} key={event.id}>
                      <span className="activity-marker">{event.status === 'completed' ? <CheckCircle2 size={15} /> : event.status === 'failed' ? '!' : <CircleDot size={14} />}</span>
                      <div className="activity-copy">
                        <div className="activity-meta"><span>{formatPhaseName(event.phase)}{event.stepId ? ` · Step ${event.stepId}` : ''}</span><time>{formatTimestamp(event.timestamp)}{event.startedAt && event.startedAt !== event.timestamp ? ` · ${formatEventDuration(event, clockNow)}` : ''}</time></div>
                        <strong>{event.title}</strong>
                        <p>{event.detail}</p>
                      </div>
                      <span className="activity-status">{formatActivityStatus(event.status)}</span>
                    </div>
                  ))}
                </div>
              </div>

              <details className="card technical-log-card" open={job?.status === 'failed'}>
                <summary>
                  <div><div className="card-kicker">RUNTIME DIAGNOSTICS</div><h3><FileText size={19} />Technical event stream</h3><p>Raw messages for reproducibility and troubleshooting.</p></div>
                  <span>{visibleLogs.length} events</span>
                </summary>
                <div className="log-filters">
                  <select aria-label="Filter by phase" value={logPhaseFilter} onChange={(event) => setLogPhaseFilter(event.target.value)}>
                    <option value="all">All phases</option>
                    {trackingPhases.map((phase) => <option key={phase.key} value={phase.key}>{formatPhaseName(phase.key)}</option>)}
                  </select>
                  <select aria-label="Filter by level" value={logLevelFilter} onChange={(event) => setLogLevelFilter(event.target.value)}>
                    <option value="all">All levels</option><option value="system">System</option><option value="stdout">Output</option><option value="stderr">Error</option>
                  </select>
                </div>
                <div className="log-view">
                  {visibleLogs.length === 0 && <p className="muted">No events match the selected filters.</p>}
                  {visibleLogs.map((log) => (
                    <div className={`log-line ${log.category || 'technical'}`} key={log.id}>
                      <span>{formatTimestamp(log.timestamp)}</span>
                      <code>{formatLogLevel(log.type || log.level || 'info')}</code>
                      <p>{sanitizeTrackingText(log.content)}</p>
                    </div>
                  ))}
                </div>
              </details>
            </div>
          </section>
        )}

        {activeStep === 'results' && (
          <section className="panel-grid">
            <ResultsWorkspace results={results} artifacts={artifacts} />
          </section>
        )}
      </main>
    </div>
  );
}

function streamStatusLabel(status: 'idle' | 'connecting' | 'connected' | 'reconnecting' | 'closed' | 'error'): string {
  const labels = {
    idle: 'Offline',
    connecting: 'Connecting',
    connected: 'Live',
    reconnecting: 'Reconnecting',
    closed: 'Closed',
    error: 'Connection error',
  };
  return labels[status];
}

function formatPhaseName(value: string): string {
  const labels: Record<string, string> = {
    phase0: 'Phase 0 · Prior evidence',
    phase1: 'Phase 1 · Data and baseline models',
    phase2: 'Phase 2 · Panel optimization',
    phase3: 'Phase 3 · Evidence outputs',
    completed: 'Analysis complete',
    failed: 'Analysis failed',
    initializing: 'Initializing',
    queued: 'Queued',
  };
  return labels[value] || value.replace(/_/g, ' ');
}

function formatRunStatus(value: string): string {
  const labels: Record<string, string> = { queued: 'Queued', running: 'Running', completed: 'Completed', failed: 'Failed' };
  return labels[value] || value.replace(/_/g, ' ');
}

function formatActivityStatus(value: string): string {
  const labels: Record<string, string> = { queued: 'Queued', waiting: 'Waiting', running: 'In progress', completed: 'Completed', retrying: 'Retrying', failed: 'Failed' };
  return labels[value] || value.replace(/_/g, ' ');
}

function formatLogLevel(value: string): string {
  const labels: Record<string, string> = { system: 'System', stdout: 'Output', stderr: 'Error', error: 'Error', warning: 'Warning', info: 'Info' };
  return labels[value.toLowerCase()] || value;
}

function formatElapsed(seconds: number): string {
  const safeSeconds = Math.max(0, Math.round(seconds));
  if (safeSeconds < 60) return `${safeSeconds}s`;
  if (safeSeconds < 3600) return `${Math.floor(safeSeconds / 60)}m ${safeSeconds % 60}s`;
  return `${Math.floor(safeSeconds / 3600)}h ${Math.floor((safeSeconds % 3600) / 60)}m`;
}

function formatCompactDuration(seconds: number): string {
  const safeSeconds = Math.max(0, Math.round(seconds));
  if (safeSeconds < 60) return '<1 min';
  if (safeSeconds < 3600) return `${Math.max(1, Math.round(safeSeconds / 60))} min`;
  const hours = Math.floor(safeSeconds / 3600);
  const minutes = Math.round((safeSeconds % 3600) / 60);
  return minutes ? `${hours}h ${minutes}m` : `${hours}h`;
}

function formatDurationRange(lowSeconds: number, highSeconds: number): string {
  if (highSeconds < 60) return 'Less than a minute';
  const low = formatCompactDuration(lowSeconds);
  const high = formatCompactDuration(Math.max(lowSeconds, highSeconds));
  return low === high ? `About ${high}` : `${low}–${high}`;
}

function formatClockTime(date: Date): string {
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', hour12: false });
}

function formatRelativeSeconds(seconds: number): string {
  if (seconds < 5) return 'just now';
  if (seconds < 60) return `${seconds}s`;
  return `${Math.round(seconds / 60)}m`;
}

function formatEventDuration(event: ActivityEvent, now: number): string {
  const start = Date.parse(event.startedAt || event.timestamp);
  const end = event.status === 'running' ? now : Date.parse(event.timestamp);
  if (Number.isNaN(start) || Number.isNaN(end)) return 'duration unavailable';
  return formatElapsed(Math.max(0, (end - start) / 1000));
}

function formatTimestamp(timestamp?: string): string {
  if (!timestamp) return '--:--:--';
  const date = new Date(timestamp);
  return Number.isNaN(date.getTime()) ? '--:--:--' : date.toLocaleTimeString([], { hour12: false });
}

function sanitizeTrackingText(content: string): string {
  const replacements: Array<[RegExp, string]> = [
    [/\u5df2\u751f\u6210\u8fd0\u884c\u914d\u7f6e\uff1a?/g, 'Run configuration created: '],
    [/\u5df2\u8fde\u63a5\u4efb\u52a1\u5b9e\u65f6\u6d41/g, 'Live task stream connected.'],
    [/\u5f00\u59cb\u6267\u884c/g, 'Starting '],
    [/\u521d\u59cb\u5316/g, ' initialized'],
    [/\u52a0\u8f7d\u6570\u636e\u6587\u4ef6/g, 'Loading the dataset'],
    [/\u6570\u636e\u5f62\u72b6/g, 'Dataset shape'],
    [/\u8f93\u51fa\u540c\u6b65/g, 'output synchronized'],
    [/\u5b8c\u6574\u6027\u6821\u9a8c/g, 'integrity validation'],
    [/\u9636\u6bb5\u7ed3\u679c/g, 'phase result'],
    [/\u5b8c\u6210/g, 'completed'],
    [/\u9519\u8bef/g, 'error'],
    [/\u8b66\u544a/g, 'warning'],
    [/\u7cfb\u7edf/g, 'system'],
    [/\u6807\u51c6\u5316/g, 'standardization'],
    [/\u7279\u5f81/g, 'feature'],
    [/\u6a21\u578b/g, 'model'],
    [/\u7ed3\u679c/g, 'result'],
    [/\u5019\u9009\u4ee3\u8c22\u7269/g, 'candidate metabolites'],
    [/\u786e\u8ba4\u7684\u751f\u7269\u6807\u5fd7\u7269/g, 'confirmed biomarkers'],
    [/\u76ee\u6807\u901a\u8def/g, 'target pathways'],
    [/\u6570\u636e\u6587\u4ef6/g, 'data file'],
    [/\u5f53\u524d\u6570\u636e/g, 'current data'],
    [/\u672a\u5b8c\u6574\u5b8c\u6210/g, 'did not complete'],
    [/\u5df2\u963b\u6b62\u8fdb\u5165/g, 'blocked from entering'],
    [/\u8fd0\u884c\u914d\u7f6e/g, 'run configuration'],
    [/\u7528\u6237\u5269\u4f59\u989d\u5ea6/g, 'remaining user quota'],
    [/\u9700\u8981\u9884\u6263\u8d39\u989d\u5ea6/g, 'required pre-deduction'],
    [/\u9884\u6263\u8d39\u989d\u5ea6\u5931\u8d25/g, 'pre-deduction failed'],
    [/\u989d\u5ea6\u4e0d\u8db3/g, 'insufficient quota'],
  ];
  let result = replacements.reduce((text, [pattern, replacement]) => text.replace(pattern, replacement), String(content || ''));
  result = result.replace(/[\u3400-\u9fff]+/g, ' ').replace(/\s{2,}/g, ' ').trim();
  return result.length >= 3 ? result : 'A runtime event was received. See server diagnostics for details.';
}

function StepButton({ step, activeStep, icon, index, title, meta, onClick, disabled, done }: {
  step: StepKey;
  activeStep: StepKey;
  icon: React.ReactNode;
  index: string;
  title: string;
  meta: string;
  onClick: (step: StepKey) => void;
  disabled?: boolean;
  done?: boolean;
}) {
  return (
    <button disabled={disabled} className={activeStep === step ? 'step-button active' : 'step-button'} onClick={() => onClick(step)}>
      <span className="step-index">{index}</span>
      <span className="step-icon">{done ? <CheckCircle2 size={17} /> : icon}</span>
      <span className="step-copy"><strong>{title}</strong><small>{meta}</small></span>
      <ChevronRight className="step-chevron" size={15} />
    </button>
  );
}

function SelectField({ label, value, onChange, options, allowEmpty, emptyLabel = 'Auto / not specified', optionLabels, badge, description, error }: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  options: string[];
  allowEmpty?: boolean;
  emptyLabel?: string;
  optionLabels?: Record<string, string>;
  badge?: string;
  description?: string;
  error?: boolean;
}) {
  return (
    <label className={error ? 'field field-error' : 'field'}>
      <span>{label}{badge && <small className="field-badge">{badge}</small>}</span>
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        {allowEmpty && <option value="">{emptyLabel}</option>}
        {options.map((option) => <option key={option} value={option}>{optionLabels?.[option] || option}</option>)}
      </select>
      {description && <small className="field-help">{description}</small>}
      {error && <small className="field-error-message">Required before starting the analysis.</small>}
    </label>
  );
}

function TextField({ label, value, onChange, placeholder, badge, description, error }: { label: string; value: string; onChange: (value: string) => void; placeholder?: string; badge?: string; description?: string; error?: boolean }) {
  return (
    <label className={error ? 'field field-error' : 'field'}>
      <span>{label}{badge && <small className="field-badge">{badge}</small>}</span>
      <input value={value} placeholder={placeholder} onChange={(event) => onChange(event.target.value)} />
      {description && <small className="field-help">{description}</small>}
      {error && <small className="field-error-message">Required before starting the analysis.</small>}
    </label>
  );
}

function PreviewCard({ fileRecord }: { fileRecord: FileRecord | null }) {
  if (!fileRecord) {
    return <div className="card empty-state dataset-empty">
      <div className="metabolite-map" aria-hidden="true">
        <span className="map-node node-a" /><span className="map-node node-b" /><span className="map-node node-c" /><span className="map-node node-d" />
        <i className="map-link link-a" /><i className="map-link link-b" /><i className="map-link link-c" />
        <FlaskConical size={26} />
      </div>
      <div className="card-kicker">DATASET AUDIT</div><h3>Waiting for a dataset preview</h3><p>After upload, this panel will show sample size, variable roles, missingness, and the metabolite-field overview.</p>
    </div>;
  }
  const visibleColumns = fileRecord.preview.columns.slice(0, 8);
  return (
    <div className="card preview-card">
      <div className="card-heading compact"><span className="section-number">A</span><div><div className="card-kicker">DATASET AUDIT</div><h3><FileText size={19} />Dataset structure preview</h3><p>{fileRecord.filename}</p></div></div>
      <div className="metric-grid">
        <Metric label="Samples" value={fileRecord.preview.n_rows} />
        <Metric label="Columns" value={fileRecord.preview.n_columns} />
        <Metric label="File type" value={fileRecord.file_type} />
      </div>
      <div className="table-wrap">
        <table>
          <thead><tr><th>Column</th><th>Type</th><th>Missing rate</th><th>Suggested role</th></tr></thead>
          <tbody>
            {visibleColumns.map((col) => (
              <tr key={col.name}><td>{col.name}</td><td>{col.dtype}</td><td>{(col.missing_rate * 100).toFixed(1)}%</td><td>{col.suggested_role}</td></tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: React.ReactNode }) {
  return <div className="metric"><span>{label}</span><strong>{value}</strong></div>;
}

function EvidenceFlow() {
  const stages = [
    { id: 'P0', title: 'Prior evidence', meta: 'Literature & pathway priors', icon: <Network size={16} /> },
    { id: 'P1', title: 'Feature engineering', meta: 'QC · selection · modeling', icon: <Layers3 size={16} /> },
    { id: 'P2', title: 'Panel validation', meta: 'Performance · stability · cost', icon: <ShieldCheck size={16} /> },
    { id: 'P3', title: 'Scientific outputs', meta: 'Figures · report · artifacts', icon: <FileText size={16} /> },
  ];
  return <div className="evidence-flow" aria-label="Analysis evidence chain">
    {stages.map((stage, index) => <div className="evidence-stage" key={stage.id}>
      <div className="evidence-stage-mark">{stage.icon}</div>
      <div><span>{stage.id}</span><strong>{stage.title}</strong><small>{stage.meta}</small></div>
      {index < stages.length - 1 && <ChevronRight className="evidence-arrow" size={16} />}
    </div>)}
  </div>;
}

function ResultsWorkspace({ results, artifacts }: { results: any | null; artifacts: Artifact[] }) {
  const [tab, setTab] = useState('overview');
  if (!results) return <div className="card empty-state"><BarChart3 size={36} /><h3>Waiting for results</h3><p>Phase-specific results and downloadable artifacts will appear when the task is complete.</p></div>;
  const phase0 = results.phase0 || {};
  const phase1 = results.phase1 || {};
  const phase2 = results.phase2 || {};
  const phase3 = results.phase3 || {};
  const featureDisplayNames = results.feature_display_names || {};
  const dataSummary = phase1.data_summary || {};
  const phase1Summary = phase1.phase1_summary || {};
  const metrics = phase2.comprehensive_metrics || {};
  const tabs = [['overview', 'Overview'], ['phase0', 'Phase 0 evidence'], ['phase1', 'Phase 1 modeling'], ['phase2', 'Final panel'], ['phase3', 'Reports & figures'], ['artifacts', 'Artifacts']];
  const unresolvedPhase1Steps = Array.isArray(phase1Summary.unresolved_failed_step_ids) ? phase1Summary.unresolved_failed_step_ids : [];
  const hasPhase1Error = Boolean(
    phase1.error ||
    unresolvedPhase1Steps.length > 0 ||
    (!phase1Summary.completed_effectively && (phase1Summary.had_error || phase1Summary.last_error || phase1.last_error))
  );
  return <div className="results-workspace">
    <div className="card results-hero"><div><div className="eyebrow">EVIDENCE SYNTHESIS · COMPLETE</div><h3><Network size={20} />Metabolomics evidence workspace</h3><p>{results.summary?.disease_name || 'Disease not specified'} · {results.summary?.clinical_scenario || 'Clinical setting not specified'}</p></div><span className="result-status success"><CheckCircle2 size={14} />{results.status || 'completed'}</span></div>
    <div className="result-tabs">{tabs.map(([key, label]) => <button key={key} className={tab === key ? 'result-tab active' : 'result-tab'} onClick={() => setTab(key)}>{label}</button>)}</div>
    {tab === 'overview' && <EvidenceFlow />}
    {tab === 'overview' && <div className="result-section-grid"><div className="card result-card"><h4>Task overview</h4><div className="result-metric-grid"><Metric label="Task status" value={results.status || 'completed'} /><Metric label="Input file" value={results.summary?.input_file || '-'} /><Metric label="Samples" value={dataSummary.n_rows ?? '-'} /><Metric label="Columns" value={dataSummary.n_cols ?? '-'} /><Metric label="Best model" value={phase2.selected_model || phase1Summary.best_model || '-'} /><Metric label="ROC-AUC" value={formatMetric(phase2.roc_auc ?? metrics.roc_auc)} /><Metric label="Final panel" value={Array.isArray(phase2.features) ? phase2.features.length : '-'} /><Metric label="Phase 3 figures" value={phase3.figure_count ?? 0} /></div></div><div className="card result-card"><h4>Run summary</h4><ResultList items={[['Disease / question', results.summary?.disease_name], ['Clinical setting', results.summary?.clinical_scenario], ['Report path', phase3.report_path], ['Stop reason', phase2.stop_reason]]} /></div>{hasPhase1Error && <div className="card result-warning"><strong>Phase 1 requires attention</strong><p>{sanitizeTrackingText(phase1Summary.last_error || phase1.last_error || phase1.error || 'Phase 1 did not complete.')}</p><span>Phase 2 results can still be viewed independently; interpret the final panel together with this error.</span></div>}</div>}
    {tab === 'overview' && <DataQualityPanel data={dataSummary} />}
    {tab === 'phase0' && <Phase0ResultPanel phase0={phase0} />}
    {tab === 'phase1' && <Phase1ResultPanel phase1={phase1} summary={phase1Summary} featureDisplayNames={featureDisplayNames} />}
    {tab === 'phase2' && <Phase2ResultPanel phase2={phase2} metrics={metrics} featureDisplayNames={featureDisplayNames} />}
    {tab === 'phase3' && <Phase3ResultPanel phase3={phase3} />}
    {tab === 'artifacts' && <ArtifactsPanel artifacts={artifacts} />}
  </div>;
}

function DataQualityPanel({ data }: { data: any }) {
  const columns = Array.isArray(data.columns) ? data.columns : [];
  return <div className="card result-card"><h4>Data profile</h4><div className="result-metric-grid"><Metric label="Samples" value={data.n_rows ?? '-'} /><Metric label="Columns" value={data.n_cols ?? '-'} /><Metric label="Target column" value={data.target_column || 'Defined by protocol'} /></div>{columns.length > 0 && <details className="compact-details"><summary>View input fields ({columns.length})</summary><div className="tag-list">{columns.slice(0, 40).map((column: string) => <span key={column} className="data-tag">{column}</span>)}</div>{columns.length > 40 && <p className="muted">Showing the first 40 of {columns.length} fields.</p>}</details>}</div>;
}

function Phase0ResultPanel({ phase0 }: { phase0: any }) {
  const biomarkers = Array.isArray(phase0.confirmed_biomarkers) ? phase0.confirmed_biomarkers : [];
  const directPathways = Array.isArray(phase0.top_pathways) ? phase0.top_pathways : [];
  const pathways = directPathways.length > 0 ? directPathways : (phase0.feature_definitions?.target_pathways || []);
  const pathwaySummary = phase0.pathway_summary || {};
  return <div className="result-section-grid"><div className="card result-card"><h4>Phase 0 prior evidence</h4><div className="result-metric-grid"><Metric label="Candidates" value={phase0.candidates?.length ?? 0} /><Metric label="Confirmed biomarkers" value={biomarkers.length} /><Metric label="Cache hit" value={phase0.cache_hit ? 'Yes' : 'No'} /><Metric label="Disease" value={phase0.disease_name || '-'} /></div><h4 className="subheading">Confirmed biomarkers</h4>{biomarkers.length === 0 ? <p className="muted">No confirmed biomarkers are available for this result.</p> : <div className="table-wrap"><table><thead><tr><th>Name</th><th>HMDB ID</th><th>Confidence</th></tr></thead><tbody>{biomarkers.slice(0, 20).map((item: any, index: number) => <tr key={`${item.name || item.metabolite || 'biomarker'}-${index}`}><td>{item.name || item.metabolite || '-'}</td><td>{item.hmdb_id || item.hmdb || item.id || '-'}</td><td>{formatMetric(item.confidence_score ?? item.score)}</td></tr>)}</tbody></table></div>}</div><div className="card result-card"><h4>Priority pathways</h4>{pathways.length === 0 ? <div className="pathway-empty"><strong>No mapped pathways</strong><p>{pathwaySummary.message || 'The confirmed biomarkers do not have a pathway mapping in this result.'}</p></div> : <><p className="result-card-intro">{pathwaySummary.message || 'Pathways retained by the Phase 0 evidence workflow.'}</p><div className="rank-list pathway-list">{pathways.slice(0, 20).map((pathway: any, index: number) => <div className="rank-item" key={`${String(pathway.name || pathway.pathway_name || pathway)}-${index}`}><b>{index + 1}</b><span>{pathway.name || pathway.pathway_name || String(pathway)}{pathway.supporting_biomarker_count ? <small>{pathway.supporting_biomarker_count} supporting biomarker{pathway.supporting_biomarker_count === 1 ? '' : 's'}</small> : null}</span></div>)}</div></>}</div></div>;
}

function Phase1ResultPanel({ phase1, summary, featureDisplayNames }: { phase1: any; summary: any; featureDisplayNames: Record<string, string> }) {
  const features = summary.final_selected_features || phase1.phase1_output_sync?.selected_features || [];
  const unresolvedPhase1Steps = Array.isArray(summary.unresolved_failed_step_ids) ? summary.unresolved_failed_step_ids : [];
  const activePhase1Error = (!summary.completed_effectively || summary.had_error || unresolvedPhase1Steps.length > 0) ? summary.last_error : '';
  return <div className="result-section-grid"><div className="card result-card"><h4>Phase 1 execution summary</h4><div className="result-metric-grid"><Metric label="Completion status" value={summary.completed_effectively ? 'Effectively completed' : summary.had_error ? 'Error' : summary.completed ? 'Completed' : 'Incomplete'} /><Metric label="Best model" value={summary.best_model || '-'} /><Metric label="Best score" value={formatMetric(summary.best_model_score)} /><Metric label="Retries" value={summary.retry_count_final ?? '-'} /></div>{activePhase1Error && <div className="result-warning"><strong>Error details</strong><p>{sanitizeTrackingText(activePhase1Error)}</p></div>}</div><div className="card result-card"><h4>Final selected features</h4><p className="result-card-intro">Names match the metabolite columns in the uploaded dataset. Internal HMDB identifiers remain available in downloadable artifacts.</p>{features.length === 0 ? <p className="muted">No Phase 1 features were selected.</p> : <div className="tag-list">{features.slice(0, 50).map((feature: string) => <span key={feature} title={feature} className="data-tag accent">{displayFeatureName(feature, featureDisplayNames)}</span>)}</div>}</div></div>;
}

function Phase2ResultPanel({ phase2, metrics, featureDisplayNames }: { phase2: any; metrics: any; featureDisplayNames: Record<string, string> }) {
  const metricItems: Array<[string, unknown]> = [['ROC-AUC', metrics.roc_auc ?? phase2.roc_auc], ['AUPRC', metrics.auprc], ['Accuracy', metrics.accuracy], ['Sensitivity', metrics.sensitivity], ['Specificity', metrics.specificity], ['Precision', metrics.precision], ['Brier score', metrics.brier_score]];
  return <div className="result-section-grid"><div className="card result-card"><h4>Final panel evaluation</h4><div className="result-metric-grid">{metricItems.map(([label, value]) => <Metric key={label} label={label} value={formatMetric(value)} />)}</div><MetricProfile items={metricItems} /><ResultList items={[['Selected model', phase2.selected_model], ['Stop reason', phase2.stop_reason], ['Clinical setting', phase2.clinical_scenario], ['Optimal depth', phase2.global_best_depth]]} /></div><div className="card result-card"><h4>Final biomarker panel</h4><p className="result-card-intro">Panel members are shown using the original uploaded metabolite names.</p>{Array.isArray(phase2.features) && phase2.features.length ? <div className="rank-list biomarker-list">{phase2.features.map((feature: string, index: number) => <div className="rank-item" key={feature} title={feature}><b>{index + 1}</b><span>{displayFeatureName(feature, featureDisplayNames)}</span><i aria-hidden="true" /></div>)}</div> : <p className="muted">No final panel is available.</p>}</div></div>;
}

function MetricProfile({ items }: { items: Array<[string, unknown]> }) {
  const visible = items.filter(([, value]) => typeof value === 'number' && Number.isFinite(value as number)).slice(0, 6);
  if (!visible.length) return null;
  return <div className="metric-profile">
    <div className="metric-profile-title"><span>PERFORMANCE PROFILE</span><small>0—1 normalized scale</small></div>
    {visible.map(([label, value]) => {
      const numeric = Math.max(0, Math.min(1, Number(value)));
      const displayWidth = String(label) === 'Brier score' ? (1 - numeric) * 100 : numeric * 100;
      return <div className="metric-profile-row" key={String(label)}><span>{String(label)}</span><div><i style={{ width: `${displayWidth}%` }} /></div><strong>{formatMetric(value)}</strong></div>;
    })}
  </div>;
}

function Phase3ResultPanel({ phase3 }: { phase3: any }) {
  const excludedFigureIds = new Set(['fig4h', 'fig4h_alt', 'objective_shift', 'objective_shift_radar']);
  const figures = (Array.isArray(phase3.figures) ? phase3.figures : []).filter((figure: any) => !excludedFigureIds.has(String(figure.figure_id || '')) && !String(figure.title || '').includes('Clinical Validation Composite'));
  const dataFiles = Array.isArray(phase3.data_files) ? phase3.data_files : [];
  const finalReports = (Array.isArray(phase3.final_reports) ? phase3.final_reports : []).filter((file: any) => ['html', 'pdf'].includes(String(file.format || '').toLowerCase()));
  const summary = phase3.manifest_summary || {};
  const requiredTasks = Number(summary.required_task_count || 0);
  const completedTasks = Number(summary.completed_task_count || 0);
  const incomplete = requiredTasks > 0 && completedTasks < requiredTasks;
  return <div className="phase3-results-layout">
    <div className="card result-card final-report-card">
      <div className="result-card-heading"><div><span className="result-kicker">PRIMARY DELIVERABLE</span><h4>Final evidence report</h4><p className="result-card-intro">The reader-facing report that consolidates the analysis, figures, methods, and interpretation.</p></div><FileText size={22} /></div>
      {finalReports.length === 0 ? <p className="muted">The final evidence report was not generated for this run.</p> : <div className="final-report-actions">{finalReports.map((file: any) => { const format = String(file.format || 'file').toLowerCase(); const label = format === 'html' ? 'Open HTML report' : 'Download PDF'; return <a key={file.path || file.name} className="primary-button" href={apiService.url(file.download_url || '')} target="_blank" rel="noreferrer"><Download size={15} />{label}</a>; })}</div>}
      <div className="result-metric-grid report-metrics"><Metric label="Figures" value={phase3.figure_count ?? figures.length} /><Metric label="Figure style" value={phase3.figure_style || '-'} /><Metric label="Formats" value={Array.isArray(phase3.output_formats) ? phase3.output_formats.join(', ') : '-'} /></div>
      {incomplete && <div className="result-warning"><strong>Figure pipeline requires attention</strong><p>{completedTasks} of {requiredTasks} figure tasks completed.</p><span>Available outputs remain downloadable; failed tasks are documented in the report.</span></div>}
    </div>

    <div className="card result-card figure-gallery-card">
      <div className="result-card-heading"><div><span className="result-kicker">SCIENTIFIC OUTPUTS</span><h4>Figures</h4><p className="result-card-intro">Preview and download the publication-oriented figures generated by Phase 3.</p></div><div className="result-heading-actions">{figures.length > 0 && phase3.figure_bundle_download_url && <a className="secondary-button compact-button" href={apiService.url(phase3.figure_bundle_download_url)} download><Archive size={15} />Download all figures</a>}<span className="result-count">{figures.length}</span></div></div>
      {figures.length === 0 ? <p className="muted">No rendered figures are available.</p> : <div className="figure-list figure-list-scroll">{figures.map((figure: any) => {
        const variants = Array.isArray(figure.download_variants) && figure.download_variants.length ? figure.download_variants : [{ format: 'download', download_url: figure.download_url }];
        return <div className="figure-item" key={figure.figure_id || figure.title}>
          <div className="figure-content">
            {figure.preview_url ? <a className="figure-preview-link" href={apiService.url(figure.preview_url)} target="_blank" rel="noreferrer"><img className="figure-preview" src={apiService.url(figure.preview_url)} alt={figure.title || figure.figure_id || 'Rendered figure'} loading="lazy" /></a> : <div className="figure-preview-placeholder">Preview available through the downloadable files below.</div>}
            <div className="figure-meta"><strong>{figure.title || figure.figure_id}</strong><span>{figure.figure_id || '-'} · {figure.source_task || 'Phase 3'}</span></div>
          </div>
          <div className="figure-downloads">{variants.map((variant: any) => <a key={`${figure.figure_id}-${variant.format}`} className="secondary-button compact-button" href={apiService.url(variant.download_url || '')} target="_blank" rel="noreferrer">{String(variant.format || 'file').toUpperCase()}</a>)}</div>
        </div>;
      })}</div>}
    </div>

    <div className="card result-card source-data-card">
      <div className="result-card-heading"><div><span className="result-kicker">REPRODUCIBILITY</span><h4>Figure source data</h4><p className="result-card-intro">Machine-readable coordinates and summaries used to reproduce the rendered figures.</p></div><Database size={20} /></div>
      {dataFiles.length === 0 ? <p className="muted">No figure source data are available.</p> : <div className="artifact-list source-data-scroll">{dataFiles.map((file: any) => <a key={file.path || file.name} className="artifact-item" href={apiService.url(file.download_url || '')} target="_blank" rel="noreferrer"><div><strong>{file.name || file.path}</strong><span>{String(file.format || 'file').toUpperCase()} · {formatBytes(file.size_bytes)}</span></div><Download size={16} /></a>)}</div>}
    </div>

    <details className="card result-card technical-outputs-card">
      <summary><span><span className="result-kicker">AUDIT MATERIALS</span><strong>Technical output inventory</strong><small>Optional files for reproducibility and review</small></span><ChevronRight size={17} /></summary>
      <div className="technical-output-body"><p className="result-card-intro"><strong>Pipeline summary</strong> is a machine-readable execution summary. <strong>Figure manifest</strong> is the machine-readable inventory of figure tasks, files, and statuses. They are useful for auditing or reproducing a run, but are not substitutes for the final evidence report.</p><div className="link-row">{phase3.report_download_url && <a className="secondary-button" href={apiService.url(phase3.report_download_url)} target="_blank" rel="noreferrer"><Download size={15} />Download pipeline summary</a>}{phase3.figure_manifest_download_url && <a className="secondary-button" href={apiService.url(phase3.figure_manifest_download_url)} target="_blank" rel="noreferrer"><Download size={15} />Download figure manifest</a>}</div></div>
    </details>
  </div>;
}

function ArtifactsPanel({ artifacts }: { artifacts: Artifact[] }) {
  const [filter, setFilter] = useState('all');
  const phases = Array.from(new Set(artifacts.map((artifact) => artifact.phase)));
  const visible = filter === 'all' ? artifacts : artifacts.filter((artifact) => artifact.phase === filter);
  return <div className="card result-card artifacts-panel"><div className="run-header"><div><h4><Download size={18} />Artifacts</h4><p className="result-card-intro">All files generated during the run, grouped by analysis phase.</p></div><select value={filter} onChange={(event) => setFilter(event.target.value)}><option value="all">All phases ({artifacts.length})</option>{phases.map((phase) => <option key={phase} value={phase}>{phase} ({artifacts.filter((artifact) => artifact.phase === phase).length})</option>)}</select></div>{visible.length === 0 ? <p className="muted">No artifacts are available.</p> : <div className="artifact-list artifact-list-scroll">{visible.map((artifact) => <a key={artifact.name} className="artifact-item" href={apiService.url(artifact.download_url)} target="_blank" rel="noreferrer"><div><strong>{artifact.name}</strong><span>{artifact.phase} · {artifact.type} · {artifact.format} · {formatBytes(artifact.size_bytes)}</span></div><Download size={16} /></a>)}</div>}</div>;
}

function ResultList({ items }: { items: Array<[string, unknown]> }) { return <div className="result-list">{items.map(([label, value]) => <div key={label}><span>{label}</span><strong>{value === undefined || value === null || value === '' ? '-' : String(value)}</strong></div>)}</div>; }
function humanizeKey(value: string): string { return value.split('_').filter(Boolean).map((word) => word.charAt(0).toUpperCase() + word.slice(1)).join(' '); }
function displayFeatureName(value: unknown, displayNames: Record<string, string>): string {
  const raw = String(value ?? '');
  if (!raw) return '-';
  const exact = displayNames[raw] || displayNames[raw.toUpperCase()];
  if (exact) return exact;
  return raw.replace(/HMDB\d+/gi, (token) => displayNames[token.toUpperCase()] || token);
}
function inferDiseaseName(filename: string): string {
  const normalized = filename.replace(/\.[^.]+$/, '').toLowerCase().replace(/[_-]+/g, ' ').replace(/\s+/g, ' ').trim();
  const diseaseAliases: Array<[RegExp, string]> = [
    [/\b(me\s*)?cfs\b|chronic fatigue syndrome/, 'chronic fatigue syndrome'],
    [/\blung cancer\b|\blung carcinoma\b|\bnsclc\b/, 'lung cancer'],
    [/\bliver cancer\b|\bhepatocellular carcinoma\b|\bhcc\b/, 'liver cancer'],
    [/\bbreast cancer\b|\bbrca\b/, 'breast cancer'],
    [/\bcolorectal cancer\b|\bcolon cancer\b|\bcrc\b/, 'colorectal cancer'],
    [/\btype 2 diabetes\b|\bt2d\b|\bdiabetes mellitus\b/, 'type 2 diabetes mellitus'],
    [/\balzheimer(?:'s)?(?: disease)?\b|\bad\b/, "Alzheimer's disease"],
    [/\bparkinson(?:'s)?(?: disease)?\b|\bpd\b/, "Parkinson's disease"],
  ];
  return diseaseAliases.find(([pattern]) => pattern.test(normalized))?.[1] || '';
}
function formatMetric(value: unknown): string { if (typeof value !== 'number' || !Number.isFinite(value)) return value === undefined || value === null ? '-' : String(value); return value.toFixed(3).replace(/0+$/, '').replace(/\.$/, ''); }
function formatBytes(value?: number): string { if (!value) return '-'; if (value < 1024) return `${value} B`; if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`; return `${(value / (1024 * 1024)).toFixed(1)} MB`; }

export default App;
