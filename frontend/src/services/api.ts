import axios, { AxiosError, AxiosInstance, AxiosRequestConfig } from 'axios';

const APP_BASE_URL = (import.meta.env.BASE_URL || '/').replace(/\/$/, '');
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || APP_BASE_URL;

export class ApiError extends Error {
  status?: number;
  details?: unknown;

  constructor(message: string, status?: number, details?: unknown) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.details = details;
  }
}

function getErrorMessage(error: AxiosError<unknown>): string {
  const data = error.response?.data;
  if (typeof data === 'string' && data.trim()) return data;
  if (data && typeof data === 'object') {
    const record = data as Record<string, unknown>;
    if (typeof record.detail === 'string') return record.detail;
    if (typeof record.message === 'string') return record.message;
    if (typeof record.error === 'string') return record.error;
  }
  return error.message || '请求失败';
}

class ApiService {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      timeout: 30000,
      headers: { 'Content-Type': 'application/json' },
    });

    this.client.interceptors.request.use((config) => {
      const token = localStorage.getItem('token');
      if (token) config.headers.Authorization = `Bearer ${token}`;
      return config;
    });

    this.client.interceptors.response.use(
      (response) => response,
      (error: AxiosError<unknown>) => {
        const apiError = new ApiError(getErrorMessage(error), error.response?.status, error.response?.data);
        console.error('[API]', apiError.message);
        return Promise.reject(apiError);
      },
    );
  }

  url(path: string): string {
    if (/^https?:\/\//.test(path)) return path;
    return `${API_BASE_URL}${path}`;
  }

  async get<T>(url: string, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.client.get<T>(url, config);
    return response.data;
  }

  async post<T, D = unknown>(url: string, data?: D, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.client.post<T>(url, data, config);
    return response.data;
  }

  async uploadFile<T>(url: string, file: File, onProgress?: (progress: number) => void): Promise<T> {
    const formData = new FormData();
    formData.append('file', file);
    const response = await this.client.post<T>(url, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      onUploadProgress: (event) => {
        if (onProgress && event.total) onProgress(Math.round((event.loaded * 100) / event.total));
      },
    });
    return response.data;
  }
}

export const apiService = new ApiService();
