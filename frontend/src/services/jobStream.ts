export type JobStreamMessage = {
  type?: string;
  job_id?: string;
  status?: string;
  state?: string;
  current_phase?: string;
  current_step?: string;
  current_step_name?: string;
  current_step_status?: string;
  stage_id?: string | number;
  step_id?: string | number;
  step_name?: string;
  title?: string;
  detail?: string;
  phase_progress?: number;
  step_index?: number;
  step_total?: number;
  elapsed_seconds?: number;
  progress?: Record<string, number>;
  phase?: string;
  message?: string;
  timestamp?: string;
  log?: { id?: string; timestamp?: string; type?: string; level?: string; content?: string; phase?: string };
  runtime_config_path?: string;
  [key: string]: unknown;
};

export type JobStreamHandlers = {
  onOpen?: () => void;
  onMessage?: (message: JobStreamMessage) => void;
  onError?: (message: string) => void;
  onClose?: () => void;
};

const MAX_RECONNECT_ATTEMPTS = 3;
const RECONNECT_DELAY_MS = 2000;

function getWebSocketBase(): string {
  const configured = import.meta.env.VITE_WS_BASE_URL || '';
  if (configured) return configured;
  const appBaseUrl = (import.meta.env.BASE_URL || '/').replace(/\/$/, '');
  return `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.host}${appBaseUrl}`;
}

class JobStreamClient {
  private socket: WebSocket | null = null;
  private jobId: string | null = null;
  private handlers: JobStreamHandlers = {};
  private reconnectAttempts = 0;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private manualClose = false;

  connect(jobId: string, handlers: JobStreamHandlers): void {
    this.disconnect();
    this.jobId = jobId;
    this.handlers = handlers;
    this.reconnectAttempts = 0;
    this.manualClose = false;
    this.open();
  }

  private open(): void {
    if (!this.jobId) return;
    const socket = new WebSocket(`${getWebSocketBase()}/api/v1/jobs/${this.jobId}/stream`);
    this.socket = socket;

    socket.onopen = () => {
      this.reconnectAttempts = 0;
      this.handlers.onOpen?.();
    };

    socket.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data) as JobStreamMessage;
        this.handlers.onMessage?.(message);
      } catch {
        this.handlers.onError?.('Unable to parse a task-stream message.');
      }
    };

    socket.onerror = () => {
      this.handlers.onError?.('Live task connection failed. Verify that the backend service is available.');
    };

    socket.onclose = () => {
      this.handlers.onClose?.();
      if (!this.manualClose && this.reconnectAttempts < MAX_RECONNECT_ATTEMPTS) {
        this.reconnectAttempts += 1;
        this.reconnectTimer = setTimeout(() => this.open(), RECONNECT_DELAY_MS);
      }
    };
  }

  disconnect(): void {
    this.manualClose = true;
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
    this.reconnectTimer = null;
    this.socket?.close();
    this.socket = null;
    this.jobId = null;
  }
}

export const jobStream = new JobStreamClient();
