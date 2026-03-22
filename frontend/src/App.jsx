import { useState, useEffect } from 'react';
import * as api from './services/api';
import { ANGLES } from './constants';
import Navbar from './components/Navbar';
import InspectionConsole from './components/InspectionConsole';
import HistoryDashboard from './components/HistoryDashboard';
import AlertsManager from './components/AlertsManager';
import './Inspection.css';
import './History.css';

function App() {
  // Navigation
  const [activePage, setActivePage] = useState('console');
  const [isArchiveView, setIsArchiveView] = useState(false);
  const [selectedSession, setSelectedSession] = useState(null);

  // Settings
  const [systemSettings, setSystemSettings] = useState({
    smtp: { server: 'smtp.gmail.com', port: 587, user: '', password: '' },
    alerts: []
  });

  // Console
  const [angleData, setAngleData] = useState({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeAngle, setActiveAngle] = useState('G01');
  const [viewMode, setViewMode] = useState('heatmap');
  const [globalResult, setGlobalResult] = useState(null);

  // History
  const [history, setHistory] = useState([]);
  const [stats, setStats] = useState({ total: 0, passes: 0, fails: 0, pass_rate: 0 });
  const [filter, setFilter] = useState('all');

  // Models
  const [availableModels, setAvailableModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('');

  // Revoke blob URLs when angleData changes to prevent memory leaks
  useEffect(() => {
    return () => {
      Object.values(angleData)
        .map(d => d.previewUrl)
        .filter(u => u?.startsWith('blob:'))
        .forEach(url => URL.revokeObjectURL(url));
    };
  }, [angleData]);

  useEffect(() => {
    const fetchInitialData = async () => {
      try {
        const [modelsRes, settingsRes] = await Promise.all([
          api.fetchModels(),
          api.fetchSettings(),
        ]);
        setAvailableModels(modelsRes.data);
        if (modelsRes.data.length > 0 && !selectedModel) {
          setSelectedModel(modelsRes.data[0]);
        }
        setSystemSettings(settingsRes.data);
      } catch (err) {
        console.error("Failed to fetch initial data:", err);
      }
    };
    fetchInitialData();
  }, []);

  useEffect(() => {
    if (activePage === 'history') {
      api.fetchHistory(filter).then(r => setHistory(r.data))
        .catch(err => console.error("Failed to fetch history:", err));
      api.fetchStats().then(r => setStats(r.data))
        .catch(err => console.error("Failed to fetch stats:", err));
    }
  }, [activePage, filter]);

  const clearState = () => {
    setAngleData({});
    setGlobalResult(null);
    setError(null);
  };

  const runBatchInspection = async () => {
    const anglesToInspect = ANGLES.filter(a => angleData[a.id]?.selectedFile);
    if (anglesToInspect.length === 0) {
      setError("No images uploaded to inspect.");
      return;
    }

    setLoading(true);
    setError(null);
    setGlobalResult(null);

    const formData = new FormData();
    anglesToInspect.forEach(angle => {
      formData.append(angle.id, angleData[angle.id].selectedFile);
    });

    try {
      const response = await api.inspectBatch(formData, selectedModel);
      const { overall_status, angles: results } = response.data;

      const newAngleData = { ...angleData };
      Object.keys(results).forEach(id => {
        newAngleData[id] = { ...newAngleData[id], result: results[id] };
      });

      setAngleData(newAngleData);
      setGlobalResult(overall_status);
    } catch (err) {
      console.error(err);
      setError('Inspection Failed: ' + (err.response?.data?.detail || 'System error'));
    } finally {
      setLoading(false);
    }
  };

  const handleSimulateTrigger = async () => {
    setLoading(true);
    try {
      const response = await api.simulateTrigger(selectedModel);
      if (activePage === 'history') {
        api.fetchHistory(filter).then(r => setHistory(r.data));
        api.fetchStats().then(r => setStats(r.data));
      } else {
        const simData = {};
        ANGLES.forEach(a => {
          if (response.data.angles[a.id]) {
            simData[a.id] = {
              result: response.data.angles[a.id],
              previewUrl: response.data.angles[a.id].original_image
            };
          }
        });
        setAngleData(simData);
        setGlobalResult(response.data.overall_status);
      }
    } catch (err) {
      setError("Simulation failed: " + (err.response?.data?.detail || err.message));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-root">
      <Navbar
        activePage={activePage}
        setActivePage={setActivePage}
        isArchiveView={isArchiveView}
        setIsArchiveView={setIsArchiveView}
        setSelectedSession={setSelectedSession}
        clearState={clearState}
        simulateTrigger={handleSimulateTrigger}
        loading={loading}
      />

      <div className="inspection-container">
        {activePage === 'console' && (
          <InspectionConsole
            globalResult={globalResult}
            isArchiveView={isArchiveView}
            selectedSession={selectedSession}
            angleData={angleData}
            setAngleData={setAngleData}
            activeAngle={activeAngle}
            setActiveAngle={setActiveAngle}
            viewMode={viewMode}
            setViewMode={setViewMode}
            selectedModel={selectedModel}
            setSelectedModel={setSelectedModel}
            availableModels={availableModels}
            loading={loading}
            error={error}
            setError={setError}
            setGlobalResult={setGlobalResult}
            setActivePage={setActivePage}
            setIsArchiveView={setIsArchiveView}
            setSelectedSession={setSelectedSession}
            runBatchInspection={runBatchInspection}
            clearState={clearState}
          />
        )}
        {activePage === 'history' && (
          <HistoryDashboard
            history={history}
            stats={stats}
            filter={filter}
            setFilter={setFilter}
            setAngleData={setAngleData}
            setGlobalResult={setGlobalResult}
            setSelectedSession={setSelectedSession}
            setActiveAngle={setActiveAngle}
            setIsArchiveView={setIsArchiveView}
            setActivePage={setActivePage}
          />
        )}
        {activePage === 'alerts' && (
          <AlertsManager
            systemSettings={systemSettings}
            setSystemSettings={setSystemSettings}
          />
        )}
      </div>
    </div>
  );
}

export default App;
