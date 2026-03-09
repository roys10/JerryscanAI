import { useState, useEffect } from 'react';
import axios from 'axios';
import { Upload, Brain, CheckCircle, XCircle, AlertCircle, Loader2, Camera, RefreshCw, History, LayoutDashboard, Search, Filter, Settings, Bell, Plus, Trash2, Edit2, Mail, Globe } from 'lucide-react';
import './Inspection.css';
import './History.css';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

function App() {
  // Navigation
  const [activePage, setActivePage] = useState('console'); // 'console' or 'history'
  const [isArchiveView, setIsArchiveView] = useState(false);
  const [selectedSession, setSelectedSession] = useState(null);

  // Settings State
  const [systemSettings, setSystemSettings] = useState({
    smtp: {
      server: 'smtp.gmail.com',
      port: 587,
      user: '',
      password: ''
    },
    alerts: []
  });
  const [editingRule, setEditingRule] = useState(null);
  const [isRuleModalOpen, setIsRuleModalOpen] = useState(false);
  const [showAdvancedSettings, setShowAdvancedSettings] = useState(false);
  const [settingsLoading, setSettingsLoading] = useState(false);

  // Console State
  const [angleData, setAngleData] = useState({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // History State
  const [history, setHistory] = useState([]);
  const [stats, setStats] = useState({ total: 0, passes: 0, fails: 0, pass_rate: 0 });
  const [filter, setFilter] = useState('all'); // 'all', 'PASS', 'FAIL'

  // Multi-Model State
  const [availableModels, setAvailableModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('');

  // Angle Selection State
  const [activeAngle, setActiveAngle] = useState('front');
  const angles = [
    { id: 'front', label: 'Front View' },
    { id: 'back', label: 'Back View' },
    { id: 'side_l', label: 'Left Side' },
    { id: 'side_r', label: 'Right Side' },
  ];

  // View Mode State
  const [viewMode, setViewMode] = useState('heatmap'); // 'heatmap' or 'segmentation'

  // Global Result State
  const [globalResult, setGlobalResult] = useState(null);

  // Get current angle's data or empty object
  const currentData = angleData[activeAngle] || {};
  const { selectedFile, previewUrl, result } = currentData;

  useEffect(() => {
    fetchModels();
    fetchSettings();
    if (activePage === 'history') {
      fetchHistory();
      fetchStats();
    }
  }, [activePage, filter]);

  const fetchSettings = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/settings`);
      setSystemSettings(response.data);
    } catch (err) {
      console.error("Failed to fetch settings:", err);
    }
  };

  const saveSettings = async (e) => {
    if (e) e.preventDefault();
    setSettingsLoading(true);
    try {
      const response = await axios.post(`${API_BASE_URL}/settings`, systemSettings);
      setSystemSettings(response.data.settings);
    } catch (err) {
      console.error("Failed to save settings:", err);
      alert("Failed to save settings.");
    } finally {
      setSettingsLoading(false);
    }
  };

  const fetchModels = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/models`);
      setAvailableModels(response.data);
      if (response.data.length > 0 && !selectedModel) {
        setSelectedModel(response.data[0]);
      }
    } catch (err) {
      console.error("Failed to fetch models:", err);
    }
  };

  const fetchHistory = async () => {
    try {
      const url = filter === 'all'
        ? `${API_BASE_URL}/history`
        : `${API_BASE_URL}/history?status=${filter}`;
      const response = await axios.get(url);
      setHistory(response.data);
    } catch (err) {
      console.error("Failed to fetch history:", err);
    }
  };

  const fetchStats = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/stats`);
      setStats(response.data);
    } catch (err) {
      console.error("Failed to fetch stats:", err);
    }
  };

  const simulateTrigger = async () => {
    setLoading(true);
    try {
      const response = await axios.post(`${API_BASE_URL}/simulate-trigger?model_name=${selectedModel}`);
      if (activePage === 'history') {
        fetchHistory();
        fetchStats();
      } else {
        // Load the simulated results into the console view
        const simData = {};
        angles.forEach(a => {
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


  const handleFileChange = (event) => {
    const file = event.target.files[0];
    processFile(file);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processFile(e.dataTransfer.files[0]);
    }
  };

  const processFile = (file) => {
    if (!file) return;

    if (!file.type.startsWith('image/')) {
      setError('Please upload an image file (JPEG/PNG/BMP)');
      return;
    }

    // Update state for THIS angle only
    setAngleData(prev => ({
      ...prev,
      [activeAngle]: {
        selectedFile: file,
        previewUrl: URL.createObjectURL(file),
        result: null
      }
    }));
    // Clear global result when new data comes in
    setGlobalResult(null);
    setError(null);
    setViewMode('heatmap');
  };

  const runBatchInspection = async () => {
    const anglesToInspect = angles.filter(a => angleData[a.id]?.selectedFile);
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
      const response = await axios.post(`${API_BASE_URL}/inspect-batch?model_name=${selectedModel}`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });

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


  const clearState = () => {
    setAngleData({});
    setGlobalResult(null);
    setError(null);
  };

  // Calculate overall stats
  const inspectedCount = Object.values(angleData).filter(d => d.selectedFile || d.result).length;

  const renderNavbar = () => (
    <nav className="navbar">
      <div className="navbar-content">
        <div className="navbar-left">
          <div className="title" style={{ color: 'white', display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <Brain size={24} color="var(--primary-color)" />
            <span style={{ fontWeight: '700', letterSpacing: '0.05em' }}>JERRYSCAN AI</span>
          </div>

          <div style={{ width: '1px', height: '24px', background: '#334155', margin: '0 2rem' }} />

          <div
            className={`nav-item ${activePage === 'console' && !isArchiveView ? 'active' : ''}`}
            onClick={() => {
              setActivePage('console');
              setIsArchiveView(false);
              setSelectedSession(null);
              clearState();
            }}
          >
            <LayoutDashboard size={18} /> Manual Inspection
          </div>
          <div
            className={`nav-item ${activePage === 'history' && !isArchiveView ? 'active' : ''}`}
            onClick={() => {
              setActivePage('history');
              setIsArchiveView(false);
              setSelectedSession(null);
            }}
          >
            <History size={18} /> History & Analytics
          </div>
          <div
            className={`nav-item ${activePage === 'alerts' && !isArchiveView ? 'active' : ''}`}
            onClick={() => {
              setActivePage('alerts');
              setIsArchiveView(false);
              setSelectedSession(null);
            }}
          >
            <Bell size={18} /> System Alerts
          </div>
        </div>
        <div className="navbar-right" style={{ display: 'flex', alignItems: 'center' }}>
          <div style={{ width: '1px', height: '24px', background: '#334155', marginRight: '1.5rem' }} />
          {isArchiveView ? (
            <button
              className="btn-simulation"
              onClick={() => { setActivePage('history'); setIsArchiveView(false); setSelectedSession(null); }}
              style={{ background: '#1e293b', borderColor: '#334155' }}
            >
              <XCircle size={16} /> Close Report
            </button>
          ) : (
            <button
              className="btn-simulation"
              onClick={simulateTrigger}
              disabled={loading}
              style={{ marginRight: '-0.5rem' }}
            >
              {loading ? <Loader2 className="spin" size={16} /> : <RefreshCw size={16} />}
              Simulation Trigger
            </button>
          )}
        </div>
      </div>
    </nav>
  );


  const renderConsole = () => (
    <>
      {/* GLOBAL STATUS BANNER (Old design restored) */}
      {globalResult && (
        <div className={`global-banner ${globalResult === 'PASS' ? 'banner-pass' : globalResult === 'FAIL' ? 'banner-fail' : 'banner-neutral'}`}
          style={isArchiveView ? { borderStyle: 'dashed', opacity: 0.95, marginBottom: '0.5rem' } : {}}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '1rem' }}>
            {isArchiveView ? <History size={32} /> : (globalResult === 'PASS' ? <CheckCircle size={32} /> : globalResult === 'FAIL' ? <XCircle size={32} /> : <AlertCircle size={32} />)}
            <span>{isArchiveView ? 'ARCHIVED REPORT:' : 'JERRYCAN STATUS:'} {globalResult}</span>
          </div>
          {globalResult === 'FAIL' && <div style={{ fontSize: '0.9rem', marginTop: '0.25rem', opacity: 0.9 }}>
            {isArchiveView ? 'Defects were detected during this session.' : 'Defects detected in one or more angles. Check details below.'}
          </div>}
        </div>
      )}

      {/* ARCHIVE METADATA BAR */}
      {isArchiveView && (
        <div style={{
          display: 'flex',
          justifyContent: 'center',
          gap: '2rem',
          padding: '0.5rem',
          background: '#f8fafc',
          borderRadius: '0.375rem',
          marginBottom: '1.5rem',
          fontSize: '0.8rem',
          color: '#64748b',
          border: '1px solid #e2e8f0',
          animation: 'fadeIn 0.5s ease-out'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <History size={14} />
            <span><strong>Log Time:</strong> {selectedSession ? new Date(selectedSession.timestamp).toLocaleString() : 'N/A'}</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Brain size={14} />
            <span><strong>Model Set:</strong> {selectedSession?.model_name || 'Standard'}</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <AlertCircle size={14} />
            <span><strong>Mode:</strong> Read-Only Archive Report</span>
          </div>
        </div>
      )}

      <div className="main-content">
        {/* Left Panel: Controls & Angles */}
        <div className="control-panel">
          <div className="card">
            <h3>Model Configuration</h3>
            <div style={{ marginTop: '0.5rem' }}>
              <select
                className="model-selector"
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                disabled={isArchiveView || loading}
              >
                {availableModels.length === 0 && <option value="">Loading models...</option>}
                {availableModels.map(name => (
                  <option key={name} value={name}>{name.replace(/_/g, ' ')}</option>
                ))}
              </select>
              <div style={{ marginTop: '0.5rem', fontSize: '0.7rem', color: 'var(--text-muted)', fontStyle: 'italic' }}>
                {isArchiveView ? 'Running inspection is disabled in archive view' : 'Select optimized model set for current batch.'}
              </div>
            </div>
          </div>

          <div className="card">
            <h3>{isArchiveView ? 'Historical Data' : 'Camera Selection'}</h3>
            <div className="angle-grid">
              {angles.map((angle) => {
                const hasData = angleData[angle.id]?.selectedFile || angleData[angle.id]?.result;
                const result = angleData[angle.id]?.result;
                const status = result?.status;

                let statusColor = '#9ca3af';
                if (status === 'PASS') statusColor = '#10b981';
                if (status === 'FAIL') statusColor = '#ef4444';
                if (status === 'UNAVAILABLE') statusColor = '#f59e0b';

                return (
                  <div
                    key={angle.id}
                    className={`angle-btn ${activeAngle === angle.id ? 'active' : ''}`}
                    onClick={() => setActiveAngle(angle.id)}
                    style={{ position: 'relative' }}
                  >
                    <Camera size={20} style={{ marginBottom: '0.25rem' }} />
                    <div>{angle.label}</div>
                    {hasData && (
                      <div style={{
                        position: 'absolute', top: 6, right: 6, width: 10, height: 10, borderRadius: '50%',
                        backgroundColor: statusColor, border: '1px solid white'
                      }}></div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>

          <div className="card">
            <h3>Actions</h3>
            {!isArchiveView ? (
              <>
                <button
                  className="btn-primary"
                  onClick={runBatchInspection}
                  disabled={loading || inspectedCount === 0}
                >
                  {loading ? <Loader2 className="spin" size={20} /> : <Brain size={20} />}
                  {loading ? 'Inspecting Batch...' : `Run Inspection (${inspectedCount})`}
                </button>
                <div style={{ marginTop: '1rem' }}>
                  <button className="btn-secondary" onClick={clearState}>
                    <RefreshCw size={16} /> Reset Session
                  </button>
                </div>
              </>
            ) : (
              <div style={{ textAlign: 'center' }}>
                <div style={{ marginBottom: '1rem', padding: '0.75rem', background: '#f8fafc', borderRadius: '0.375rem', fontSize: '0.85rem', color: 'var(--text-muted)', border: '1px solid #e2e8f0' }}>
                  <AlertCircle size={16} style={{ verticalAlign: 'middle', marginRight: '0.5rem' }} />
                  Viewing archived report. Manual controls are disabled.
                </div>
                <button className="btn-primary" onClick={() => { setActivePage('history'); setIsArchiveView(false); setSelectedSession(null); }}>
                  <History size={18} /> Back to History
                </button>
              </div>
            )}
          </div>

          {error && (
            <div style={{ color: '#ef4444', background: '#fee2e2', padding: '1rem', borderRadius: '0.5rem', display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
              <AlertCircle size={20} /> <span style={{ fontSize: '0.9rem' }}>{error}</span>
            </div>
          )}
        </div>

        {/* Right Panel: Viewport */}
        <div className="card" style={{ minHeight: '600px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
            <h3>{angles.find(a => a.id === activeAngle)?.label}</h3>
            {result && result.status !== 'UNAVAILABLE' && (
              <div style={{ display: 'flex', gap: '0.5rem', background: '#f3f4f6', padding: '0.25rem', borderRadius: '0.375rem' }}>
                <button onClick={() => setViewMode('heatmap')} style={{ border: 'none', padding: '0.25rem 0.75rem', cursor: 'pointer', background: viewMode === 'heatmap' ? 'white' : 'transparent', borderRadius: '0.25rem' }}>Anomaly Map</button>
                <button onClick={() => setViewMode('segmentation')} style={{ border: 'none', padding: '0.25rem 0.75rem', cursor: 'pointer', background: viewMode === 'segmentation' ? 'white' : 'transparent', borderRadius: '0.25rem' }}>Pred Mask</button>
              </div>
            )}
          </div>

          {!previewUrl && !result ? (
            <div className={`upload-zone ${isArchiveView ? 'disabled' : ''}`} onClick={() => !isArchiveView && document.getElementById('fileInput').click()}>
              <Upload size={32} color={isArchiveView ? '#94a3b8' : "var(--primary-color)"} />
              <h4>{isArchiveView ? 'No Image Data' : 'Upload Image'}</h4>
              {!isArchiveView && <input id="fileInput" type="file" hidden accept="image/*" onChange={handleFileChange} />}
            </div>
          ) : (
            <div className="preview-container">
              {result ? (
                result.status === 'UNAVAILABLE' ? (
                  <div style={{ textAlign: 'center', color: 'white' }}>
                    <AlertCircle size={48} color="#f59e0b" />
                    <h3 style={{ color: '#f59e0b', margin: '0.5rem 0' }}>Model Unavailable</h3>
                    <p style={{ margin: '0.5rem 0' }}>No AI model loaded for this angle.</p>
                    {result.original_image && <img src={result.original_image} style={{ maxWidth: '200px', opacity: 0.5 }} />}
                  </div>
                ) : (
                  <>
                    <div className={`status-badge ${result.status === 'PASS' ? 'status-pass' : 'status-fail'}`}>
                      {result.status} ({result.score_percentage?.toFixed(1)}%)
                    </div>
                    <img src={viewMode === 'heatmap' ? result.heatmap_image : result.segmentation_image} className="preview-image" />
                  </>
                )
              ) : (
                <img src={previewUrl} className="preview-image" />
              )}
            </div>
          )}
        </div>
      </div>
    </>
  );

  const renderHistory = () => (
    <div className="history-container">
      <div className="stats-grid">
        <div className="stat-card">
          <h4>Total Scans</h4>
          <div className="stat-value">{stats.total}</div>
        </div>
        <div className="stat-card">
          <h4>Pass Rate</h4>
          <div className="stat-value">{stats.pass_rate?.toFixed(1)}%</div>
        </div>
        <div className="stat-card">
          <h4>Passes</h4>
          <div className="stat-value" style={{ color: '#10b981' }}>{stats.passes}</div>
        </div>
        <div className="stat-card fail">
          <h4>Defects Found</h4>
          <div className="stat-value" style={{ color: '#ef4444' }}>{stats.fails}</div>
        </div>
      </div>

      <div className="history-controls">
        <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
          <Filter size={18} />
          <select
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            style={{ padding: '0.5rem', borderRadius: '0.375rem', border: '1px solid var(--border-color)' }}
          >
            <option value="all">All Results</option>
            <option value="PASS">Pass Only</option>
            <option value="FAIL">Fail Only</option>
          </select>
        </div>
        <div style={{ color: 'var(--text-muted)' }}>Showing last {history.length} records</div>
      </div>

      <div className="history-table-container">
        <table className="history-table">
          <thead>
            <tr>
              <th>Timestamp</th>
              <th>Jerrycan ID</th>
              <th>Model Version</th>
              <th>Overall Status</th>
              <th>Angles Checked</th>
              <th>Action</th>
            </tr>
          </thead>
          <tbody>
            {history.map(session => (
              <tr key={session.id} onClick={() => {
                const mappedData = {};
                Object.keys(session.angles).forEach(id => {
                  mappedData[id] = {
                    result: session.angles[id],
                    previewUrl: session.angles[id].original_image
                  };
                });
                setAngleData(mappedData);
                setGlobalResult(session.overall_status);
                setSelectedSession(session);

                // Find first angle with data to focus on
                const firstAngle = Object.keys(session.angles)[0];
                if (firstAngle) setActiveAngle(firstAngle);

                setIsArchiveView(true);
                setActivePage('console');
              }}>
                <td>{new Date(session.timestamp).toLocaleString()}</td>
                <td><code style={{ fontSize: '0.75rem' }}>{session.id.split('-')[0]}...</code></td>
                <td style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>{session.model_name || 'Standard'}</td>
                <td>
                  <span className={`status-row-badge ${session.overall_status === 'PASS' ? 'badge-pass' : 'badge-fail'}`}>
                    {session.overall_status}
                  </span>
                </td>
                <td>
                  <div style={{ display: 'flex', gap: '4px' }}>
                    {[['front', 'Front'], ['back', 'Back'], ['side_l', 'L'], ['side_r', 'R']].map(([id, label]) => (
                      <div key={id} style={{
                        width: 14, height: 14, borderRadius: '2px', fontSize: '8px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'white',
                        background: !session.angles[id] ? '#e5e7eb' : (session.angles[id].status === 'PASS' ? '#10b981' : (session.angles[id].status === 'FAIL' ? '#ef4444' : '#f59e0b'))
                      }} title={label}>{label}</div>
                    ))}
                  </div>
                </td>
                <td style={{ color: 'var(--primary-color)', fontWeight: 600 }}>View Details</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );


  const addOrUpdateRule = async () => {
    const updatedAlerts = editingRule.id
      ? systemSettings.alerts.map(r => r.id === editingRule.id ? editingRule : r)
      : [...systemSettings.alerts, { ...editingRule, id: Date.now().toString() }];

    const newSettings = { ...systemSettings, alerts: updatedAlerts };
    setSystemSettings(newSettings);
    setIsRuleModalOpen(false);
    setEditingRule(null);

    // Immediate persistence
    try {
      await axios.post(`${API_BASE_URL}/settings`, newSettings);
    } catch (err) {
      console.error("Failed to persist rule change:", err);
    }
  };

  const deleteRule = async (id) => {
    const newSettings = { ...systemSettings, alerts: systemSettings.alerts.filter(r => r.id !== id) };
    setSystemSettings(newSettings);
    try {
      await axios.post(`${API_BASE_URL}/settings`, newSettings);
    } catch (err) {
      console.error("Failed to persist rule deletion:", err);
    }
  };

  const toggleRule = async (id) => {
    const newSettings = {
      ...systemSettings,
      alerts: systemSettings.alerts.map(r => r.id === id ? { ...r, enabled: !r.enabled } : r)
    };
    setSystemSettings(newSettings);
    try {
      await axios.post(`${API_BASE_URL}/settings`, newSettings);
    } catch (err) {
      console.error("Failed to persist rule toggle:", err);
    }
  };

  const openRuleEditor = (rule = null) => {
    setEditingRule(rule || {
      name: '',
      type: 'consecutive_fails',
      threshold: 3,
      window: 50,
      emails: [],
      webhook_url: '',
      enabled: true
    });
    setIsRuleModalOpen(true);
  };

  const renderAlerts = () => (
    <div className="history-container" style={{ animation: 'fadeIn 0.4s ease-out' }}>
      {/* 1. Custom Rules Management Section */}
      <div className="card" style={{ marginBottom: '1.5rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
          <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Bell size={20} color="var(--primary-color)" /> Custom Alert Rules
          </h3>
          <button className="btn-primary" style={{ width: 'auto', padding: '0.5rem 1.25rem' }} onClick={() => openRuleEditor()}>
            <Plus size={18} /> New Rule
          </button>
        </div>

        <div className="history-table-container">
          <table className="history-table">
            <thead>
              <tr>
                <th>Rule Name</th>
                <th>Condition</th>
                <th>Threshold</th>
                <th>Recipients</th>
                <th>Status</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {systemSettings.alerts.length === 0 ? (
                <tr>
                  <td colSpan="6" style={{ textAlign: 'center', padding: '3rem', color: 'var(--text-muted)' }}>
                    No custom rules defined yet. Create one to start monitoring.
                  </td>
                </tr>
              ) : (
                systemSettings.alerts.map(rule => (
                  <tr key={rule.id}>
                    <td style={{ fontWeight: 600 }}>{rule.name}</td>
                    <td>
                      <span style={{ fontSize: '0.8rem', padding: '2px 8px', borderRadius: '12px', background: '#f1f5f9', color: '#475569' }}>
                        {rule.type === 'consecutive_fails' ? 'Failure Streak' : 'Pass Rate Drop'}
                      </span>
                    </td>
                    <td>
                      {rule.type === 'consecutive_fails' ? `${rule.threshold} items` : `${rule.threshold}% (Sample: ${rule.window})`}
                    </td>
                    <td>
                      <div style={{ display: 'flex', gap: '4px', flexWrap: 'wrap' }}>
                        {rule.emails.length > 0 && <Mail size={14} title={rule.emails.join(', ')} color="var(--primary-color)" />}
                        {rule.webhook_url && <Globe size={14} title={rule.webhook_url} color="var(--primary-color)" />}
                        {rule.emails.length === 0 && !rule.webhook_url && <span style={{ fontStyle: 'italic', color: '#94a3b8' }}>None</span>}
                      </div>
                    </td>
                    <td>
                      <button
                        onClick={() => toggleRule(rule.id)}
                        style={{ border: 'none', background: 'transparent', cursor: 'pointer', color: rule.enabled ? '#10b981' : '#94a3b8', fontWeight: 600, fontSize: '0.85rem' }}
                      >
                        {rule.enabled ? 'Enabled' : 'Disabled'}
                      </button>
                    </td>
                    <td>
                      <div style={{ display: 'flex', gap: '0.75rem' }}>
                        <Edit2 size={16} className="btn-icon" style={{ cursor: 'pointer', color: '#64748b' }} onClick={() => openRuleEditor(rule)} />
                        <Trash2 size={16} className="btn-icon" style={{ cursor: 'pointer', color: '#ef4444' }} onClick={() => deleteRule(rule.id)} />
                      </div>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* 2. Global SMTP Section */}
      <div style={{ marginBottom: '2rem' }}>
        <div className="card" style={{ maxWidth: '600px', margin: '0 auto' }}>
          <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1.25rem' }}>
            <Mail size={18} color="var(--primary-color)" /> SMTP Server (Email Provider)
          </h3>
          <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '1rem', marginBottom: '1rem' }}>
            <div className="form-group">
              <label>Server</label>
              <input type="text" value={systemSettings.smtp.server} onChange={(e) => setSystemSettings({ ...systemSettings, smtp: { ...systemSettings.smtp, server: e.target.value } })} className="modal-input" />
            </div>
            <div className="form-group">
              <label>Port</label>
              <input type="number" value={systemSettings.smtp.port} onChange={(e) => setSystemSettings({ ...systemSettings, smtp: { ...systemSettings.smtp, port: parseInt(e.target.value) || 587 } })} className="modal-input" />
            </div>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
            <div className="form-group">
              <label>User / Sender</label>
              <input type="text" value={systemSettings.smtp.user} onChange={(e) => setSystemSettings({ ...systemSettings, smtp: { ...systemSettings.smtp, user: e.target.value } })} className="modal-input" />
            </div>
            <div className="form-group">
              <label>App Password</label>
              <input type="password" value={systemSettings.smtp.password} onChange={(e) => setSystemSettings({ ...systemSettings, smtp: { ...systemSettings.smtp, password: e.target.value } })} className="modal-input" />
            </div>
          </div>
          <div style={{ marginTop: '1.5rem', display: 'flex', justifyContent: 'flex-end' }}>
            <button onClick={saveSettings} disabled={settingsLoading} className="btn-primary" style={{ width: 'auto', padding: '0.5rem 1.5rem', fontSize: '0.8rem' }}>
              {settingsLoading ? <Loader2 className="spin" size={14} /> : 'Save SMTP Settings'}
            </button>
          </div>
        </div>
      </div>

      {/* RULE EDITOR MODAL */}
      {isRuleModalOpen && (
        <div className="modal-backdrop">
          <div className="modal-content" style={{ maxWidth: '500px', animation: 'scaleIn 0.2s ease-out' }}>
            <h3 style={{ marginBottom: '1.5rem' }}>{editingRule.id ? 'Edit Alert Rule' : 'Create New Alert Rule'}</h3>

            <div className="form-group" style={{ marginBottom: '1rem' }}>
              <label>Rule Name</label>
              <input
                type="text"
                value={editingRule.name}
                onChange={e => setEditingRule({ ...editingRule, name: e.target.value })}
                className="modal-input"
              />
            </div>

            <div className="form-group" style={{ marginBottom: '1rem' }}>
              <label>Trigger Condition</label>
              <select
                value={editingRule.type}
                onChange={e => setEditingRule({ ...editingRule, type: e.target.value })}
                className="modal-input"
              >
                <option value="consecutive_fails">Failure Streak</option>
                <option value="pass_rate">Pass Rate Drop</option>
              </select>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1rem' }}>
              <div className="form-group">
                <label>{editingRule.type === 'consecutive_fails' ? 'Items Count' : 'Min Pass %'}</label>
                <input
                  type="number"
                  value={editingRule.threshold}
                  onChange={e => setEditingRule({ ...editingRule, threshold: parseFloat(e.target.value) || 0 })}
                  className="modal-input"
                />
              </div>
              {editingRule.type === 'pass_rate' && (
                <div className="form-group">
                  <label>Sample Size (last X)</label>
                  <input
                    type="number"
                    value={editingRule.window}
                    onChange={e => setEditingRule({ ...editingRule, window: parseInt(e.target.value) || 50 })}
                    className="modal-input"
                  />
                </div>
              )}
            </div>

            <div className="form-group" style={{ marginBottom: '1rem' }}>
              <label>Email Recipients</label>
              <input
                type="text"
                value={editingRule.emails.join(', ')}
                onChange={e => setEditingRule({ ...editingRule, emails: e.target.value.split(',').map(s => s.trim()).filter(s => s !== '') })}
                className="modal-input"
              />
            </div>

            <div className="form-group" style={{ marginBottom: '1.5rem' }}>
              <label>Webhook URL (Optional)</label>
              <input
                type="url"
                placeholder="https://..."
                value={editingRule.webhook_url}
                onChange={e => setEditingRule({ ...editingRule, webhook_url: e.target.value })}
                className="modal-input"
              />
            </div>

            <div className="form-group" style={{ marginBottom: '1.5rem' }}>
              <label>Status</label>
              <select
                value={editingRule.enabled ? 'enabled' : 'disabled'}
                onChange={e => setEditingRule({ ...editingRule, enabled: e.target.value === 'enabled' })}
                className="modal-input"
              >
                <option value="enabled">Enabled</option>
                <option value="disabled">Disabled</option>
              </select>
            </div>

            <div style={{ display: 'flex', gap: '1rem', justifyContent: 'flex-end', alignItems: 'center', marginTop: '1.5rem' }}>
              <button
                className="btn-secondary"
                style={{ padding: '0.75rem 2rem', fontSize: '0.85rem', whiteSpace: 'nowrap' }}
                onClick={() => setIsRuleModalOpen(false)}
              >
                Cancel
              </button>
              <button
                className="btn-primary"
                style={{ width: 'auto', padding: '0.75rem 4rem', fontWeight: 600, whiteSpace: 'nowrap' }}
                onClick={addOrUpdateRule}
              >
                Save Rule
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );


  return (
    <div className="app-root">
      {renderNavbar()}

      <div className="inspection-container">
        {activePage === 'console' && renderConsole()}
        {activePage === 'history' && renderHistory()}
        {activePage === 'alerts' && renderAlerts()}
      </div>
    </div>
  );
}

export default App;
