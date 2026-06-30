import axios from 'axios';

const BACKEND_BASE_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: BACKEND_BASE_URL,
});

export const fetchModels = () => api.get('/models');

export const fetchSettings = () => api.get('/settings');

export const saveSettings = (settings) => api.post('/settings', settings);

export const fetchHistory = (filter) => {
  const url = filter === 'all' ? '/history' : `/history?status=${filter}`;
  return api.get(url);
};

export const fetchStats = () => api.get('/stats');

export const inspectBatch = (formData, modelName) =>
  api.post(`/inspect-batch?model_name=${modelName}`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });

export const simulateTrigger = (modelName) =>
  api.post(`/simulate-trigger?model_name=${modelName}`);

export default api;
