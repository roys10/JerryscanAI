import { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { Upload, Brain, CheckCircle, XCircle, AlertCircle, Loader2, Camera, RefreshCw, History, LayoutDashboard, Search, Filter, Settings, Bell, Plus, Trash2, Edit2, Mail, Globe, Lock } from 'lucide-react';
import './Inspection.css';
import './History.css';
import { ANGLES, ANGLE_IDS } from './constants';

const BACKEND_BASE_URL = import.meta.env.VITE_BACKEND_URL
  || (import.meta.env.PROD ? '' : 'http://localhost:8000');
function apiErrorDetail(err, fallback) {
  const detail = err.response?.data?.detail;
  if (typeof detail === 'string') return detail;
  if (detail && typeof detail === 'object') return detail.error || JSON.stringify(detail);
  return err.message || fallback;
}

function ResultImage({ src, ...props }) {
  return src ? <img src={src} {...props} /> : null;
}

function displayModelName(displayName, modelId) {
  if (displayName) return displayName;
  return modelId ? modelId.replace(/_/g, ' ') : 'Model is not loaded';
}

function unavailableThresholdMessage(angle, modelConfiguration) {
  const unavailable = modelConfiguration?.unavailable_angles?.[angle];
  const detail = unavailable?.detail?.toLowerCase() || '';
  if (unavailable?.stage === 'contract_validation' && detail.includes('decision_threshold')) {
    return 'Threshold is not set';
  }
  return 'Model is not loaded';
}

function qualityFailureBoundary(result) {
  if (Number.isFinite(result?.image_threshold)) {
    return Math.min(100, Math.max(0, 100 - result.image_threshold));
  }
  if (Number.isFinite(result?.quality_failure_boundary_percentage)) {
    return result.quality_failure_boundary_percentage;
  }
  return null;
}

function thresholdQualityBoundary(contract) {
  if (Number.isFinite(contract?.quality_failure_boundary_percentage)) {
    return contract.quality_failure_boundary_percentage;
  }
  if (Number.isFinite(contract?.value)) {
    return Math.min(100, Math.max(0, 100 - contract.value));
  }
  return null;
}

function qualityScore(result) {
  if (Number.isFinite(result?.raw_image_score)) {
    const quality = 100 - result.raw_image_score;
    return Math.min(100, Math.max(0, quality));
  }
  return Number.isFinite(result?.quality_score_percentage)
    ? result.quality_score_percentage
    : null;
}

function resultViews(result) {
  if (!result) return [];

  return [
    result.defect_overlay_image && {
      id: 'defect',
      label: 'Defect Location',
      src: result.defect_overlay_image,
      alt: 'Defect localization overlay',
    },
    result.heatmap_image && {
      id: 'heatmap',
      label: 'Anomaly Map',
      src: result.heatmap_image,
      alt: 'PatchCore anomaly map',
    },
    result.segmentation_image && {
      id: 'segmentation',
      label: 'Preprocessing Mask',
      src: result.segmentation_image,
      alt: 'Preprocessed model input',
    },
  ].filter(Boolean);
}

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
      password_configured: false
    },
    alerts: []
  });
  const [editingRule, setEditingRule] = useState(null);
  const [isRuleModalOpen, setIsRuleModalOpen] = useState(false);
  const [settingsLoading, setSettingsLoading] = useState(false);

  // Console State
  const [angleData, setAngleData] = useState({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const inspectionInFlight = useRef(false);

  // History State
  const [history, setHistory] = useState([]);
  const [stats, setStats] = useState({ total: 0, decision_count: 0, passes: 0, faults: 0, wrong_inputs: 0, pass_rate: null });
  const [filter, setFilter] = useState('all'); // 'all', 'PASS', 'FAIL'

  // The production service owns exactly one configured model.
  const [modelConfiguration, setModelConfiguration] = useState(null);

  // Angle Selection State
  const [activeAngle, setActiveAngle] = useState('G01');
  const [angles, setAngles] = useState(ANGLES);
  const backendAvailableAngles = Array.isArray(modelConfiguration?.available_angles)
    ? modelConfiguration.available_angles
    : Array.isArray(modelConfiguration?.required_angles)
      ? modelConfiguration.required_angles
      : ANGLE_IDS;
  const availableAngleIds = isArchiveView ? angles.map(angle => angle.id) : backendAvailableAngles;
  const unavailableAngles = modelConfiguration?.unavailable_angles || {};

  // View Mode State
  const [viewMode, setViewMode] = useState('defect');

  // Global Result State
  const [globalResult, setGlobalResult] = useState(null);

  // Get current angle's data or empty object
  const currentData = angleData[activeAngle] || {};
  const { previewUrl, result } = currentData;
  const displayedQualityBoundary = qualityFailureBoundary(result);
  const configuredThresholdAngles = Array.isArray(modelConfiguration?.configured_angles)
    ? modelConfiguration.configured_angles
    : Object.keys(modelConfiguration?.decision_thresholds || {});
  const displayedThresholds = isArchiveView && Number.isFinite(result?.image_threshold)
    ? [[result.angle || activeAngle, { value: result.image_threshold }]]
    : configuredThresholdAngles.map(angle => [
      angle,
      modelConfiguration?.decision_thresholds?.[angle] || null,
    ]);
  const showThresholdConfiguration = isArchiveView
    ? Number.isFinite(result?.image_threshold)
    : Boolean(modelConfiguration?.ready_for_inference && modelConfiguration?.model_id);
  const availableResultViews = resultViews(result);
  const selectedResultView = availableResultViews.find(view => view.id === viewMode)
    || availableResultViews[0];

  useEffect(() => {
    fetchModelConfiguration();
    fetchSettings();
    if (activePage === 'history') {
      fetchHistory();
      fetchStats();
    }
  }, [activePage, filter]);

  const fetchSettings = async () => {
    try {
      const response = await axios.get(`${BACKEND_BASE_URL}/settings`);
      setSystemSettings(response.data);
    } catch (err) {
      console.error("Failed to fetch settings:", err);
    }
  };

  const saveSettings = async (e) => {
    if (e) e.preventDefault();
    setSettingsLoading(true);
    try {
      const response = await axios.post(`${BACKEND_BASE_URL}/settings`, systemSettings);
      setSystemSettings(response.data.settings);
    } catch (err) {
      console.error("Failed to save settings:", err);
      alert("Failed to save settings.");
    } finally {
      setSettingsLoading(false);
    }
  };

  const fetchModelConfiguration = async () => {
    try {
      const response = await axios.get(`${BACKEND_BASE_URL}/health`);
      setModelConfiguration(response.data);
      const configuredFromBackend = response.data.configured_angles;
      const configured = Array.isArray(configuredFromBackend)
        ? configuredFromBackend
        : ANGLE_IDS;
      // The production camera layout remains visible even while one model is
      // unavailable. Extra declared angles are appended for forward compatibility.
      const visible = [...new Set([...ANGLE_IDS, ...configured])];
      setAngles(visible.map(id => ({ id, label: id })));
      setActiveAngle(current => visible.includes(current) ? current : visible[0]);
    } catch (err) {
      console.error("Failed to fetch model configuration:", err);
      setModelConfiguration(null);
      setAngles(ANGLES);
      setActiveAngle(current => ANGLE_IDS.includes(current) ? current : ANGLE_IDS[0]);
    }
  };

  const fetchHistory = async () => {
    try {
      const url = filter === 'all'
        ? `${BACKEND_BASE_URL}/history`
        : `${BACKEND_BASE_URL}/history?status=${filter}`;
      const response = await axios.get(url);
      setHistory(response.data);
    } catch (err) {
      console.error("Failed to fetch history:", err);
    }
  };

  const fetchStats = async () => {
    try {
      const response = await axios.get(`${BACKEND_BASE_URL}/stats`);
      setStats(response.data);
    } catch (err) {
      console.error("Failed to fetch stats:", err);
    }
  };

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    processFile(file);
  };

  const processFile = (file) => {
    if (!file) return;

    if (!availableAngleIds.includes(activeAngle)) {
      const reason = unavailableAngles[activeAngle]?.detail;
      setError(`${activeAngle} is unavailable${reason ? `: ${reason}` : '.'}`);
      return;
    }

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
    setViewMode('defect');
  };

  const runBatchInspection = async () => {
    if (inspectionInFlight.current) return;

    const selectedAngles = angles.filter(a => angleData[a.id]?.selectedFile);
    if (selectedAngles.length === 0) {
      setError('Upload at least one camera image to inspect.');
      return;
    }
    const unavailableSelected = selectedAngles.filter(a => !availableAngleIds.includes(a.id));
    if (unavailableSelected.length > 0) {
      setError(`Cannot inspect unavailable camera models: ${unavailableSelected.map(a => a.label).join(', ')}.`);
      return;
    }
    const anglesToInspect = selectedAngles;

    inspectionInFlight.current = true;
    setLoading(true);
    setError(null);
    setGlobalResult(null);

    try {
      const formData = new FormData();
      anglesToInspect.forEach(angle => {
        formData.append(angle.id, angleData[angle.id].selectedFile);
      });
      // Let the browser add the multipart boundary to Content-Type.
      const response = await axios.post(`${BACKEND_BASE_URL}/inspect-batch`, formData);

      const { overall_status, angles: results } = response.data;

      const newAngleData = { ...angleData };
      Object.keys(results).forEach(id => {
        newAngleData[id] = { ...newAngleData[id], result: results[id] };
      });

      setAngleData(newAngleData);
      setGlobalResult(overall_status);
      setViewMode('defect');

    } catch (err) {
      console.error(err);
      setError('Inspection failed: ' + apiErrorDetail(err, 'Request failed'));
    } finally {
      inspectionInFlight.current = false;
      setLoading(false);
    }
  };


  const clearState = () => {
    setAngleData({});
    setGlobalResult(null);
    setError(null);
  };

  // Calculate overall stats
  const uploadedCount = angles.filter(angle => angleData[angle.id]?.selectedFile).length;

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
        {isArchiveView && (
          <div className="navbar-right" style={{ display: 'flex', alignItems: 'center' }}>
            <div style={{ width: '1px', height: '24px', background: '#334155', marginRight: '1.5rem' }} />
            <button
              className="btn-report-close"
              onClick={() => { setActivePage('history'); setIsArchiveView(false); setSelectedSession(null); }}
              style={{ background: '#1e293b', borderColor: '#334155' }}
            >
              <XCircle size={16} /> Close Report
            </button>
          </div>
        )}
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
            <span><strong>Model Set:</strong> {displayModelName(
              result?.model_display_name,
              selectedSession?.model_name
            )}</span>
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
            <div className="model-configuration">
              <div className="model-configuration-row">
                <span>Model</span>
                <strong>{isArchiveView
                  ? displayModelName(
                    result?.model_display_name,
                    result?.model_id || selectedSession?.model_name
                  )
                  : displayModelName(
                    modelConfiguration?.display_name,
                    modelConfiguration?.model_id
                  )}</strong>
              </div>
              {showThresholdConfiguration && (
                <div className="model-configuration-row">
                  <span>Thresholds by camera</span>
                  <div className="model-threshold-list">
                    {displayedThresholds.map(([angle, contract]) => (
                      <div className="model-threshold-item" key={angle}>
                        {Number.isFinite(contract?.value) && contract.available !== false ? (
                          <>
                            <strong>{angle}: {thresholdQualityBoundary(contract).toFixed(1)}% quality</strong>
                            <small>Raw PatchCore threshold: {Number(contract.value).toFixed(4)}</small>
                          </>
                        ) : (
                          <strong>{angle}: {unavailableThresholdMessage(angle, modelConfiguration)}</strong>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}
              {!isArchiveView && modelConfiguration?.coverage === 'partial' && (
                <div className="model-availability-warning" role="status">
                  Partial coverage: {availableAngleIds.length} of {modelConfiguration.configured_angles?.length || angles.length} camera models available.
                </div>
              )}
            </div>
          </div>

          <div className="card">
            <h3>{isArchiveView ? 'Historical Data' : 'Camera Selection'}</h3>
            <div className="angle-grid">
              {angles.map((angle) => {
                const hasData = angleData[angle.id]?.selectedFile || angleData[angle.id]?.result;
                const result = angleData[angle.id]?.result;
                const status = result?.status;
                const isAvailable = availableAngleIds.includes(angle.id);

                let statusColor = '#9ca3af';
                if (status === 'PASS') statusColor = '#10b981';
                if (status === 'FAIL') statusColor = '#ef4444';
                if (status === 'WRONG_INPUT') statusColor = '#f59e0b';

                return (
                  <div
                    key={angle.id}
                    className={`angle-btn ${activeAngle === angle.id ? 'active' : ''} ${!isAvailable ? 'unavailable' : ''}`}
                    onClick={() => setActiveAngle(angle.id)}
                    style={{ position: 'relative' }}
                    aria-disabled={!isAvailable && !isArchiveView}
                    title={!isAvailable ? (unavailableAngles[angle.id]?.detail || 'Model unavailable') : undefined}
                  >
                    <Camera size={20} style={{ marginBottom: '0.25rem' }} />
                    <div>{angle.label}</div>
                    {!isAvailable && !isArchiveView && (
                      <small className="angle-availability">Unavailable</small>
                    )}
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
                  disabled={loading || uploadedCount === 0}
                  aria-busy={loading}
                >
                  {loading ? (
                    <Loader2 className="batch-loading-spinner" size={20} aria-hidden="true" />
                  ) : (
                    <Brain size={20} aria-hidden="true" />
                  )}
                  {loading
                    ? 'Inspecting Batch...'
                    : `Run Inspection (${uploadedCount} image${uploadedCount === 1 ? '' : 's'})`}
                </button>
                <div style={{ marginTop: '1rem' }}>
                  <button className="btn-secondary" onClick={clearState} disabled={loading}>
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
            {result && result.status !== 'WRONG_INPUT' && (
              <div className="result-view-toggle" role="group" aria-label="Inspection visualization">
                {availableResultViews.map(view => (
                  <button
                    key={view.id}
                    type="button"
                    className={selectedResultView?.id === view.id ? 'active' : ''}
                    aria-pressed={selectedResultView?.id === view.id}
                    onClick={() => setViewMode(view.id)}
                  >
                    {view.label}
                  </button>
                ))}
              </div>
            )}
          </div>

          {!previewUrl && !result ? (
            <div
              className={`upload-zone ${isArchiveView || !availableAngleIds.includes(activeAngle) ? 'disabled' : ''}`}
              onClick={() => !isArchiveView && availableAngleIds.includes(activeAngle) && document.getElementById('fileInput').click()}
            >
              <Upload size={32} color={isArchiveView || !availableAngleIds.includes(activeAngle) ? '#94a3b8' : "var(--primary-color)"} />
              <h4>
                {isArchiveView
                  ? 'No Image Data'
                  : availableAngleIds.includes(activeAngle)
                    ? 'Upload Image'
                    : `${activeAngle} model unavailable`}
              </h4>
              {!isArchiveView && !availableAngleIds.includes(activeAngle) && (
                <p>{unavailableAngles[activeAngle]?.detail || 'This camera is not currently inspectable.'}</p>
              )}
              {!isArchiveView && availableAngleIds.includes(activeAngle) && <input id="fileInput" type="file" hidden accept="image/*" onChange={handleFileChange} />}
            </div>
          ) : (
            <div className="preview-container">
              {result ? (
                result.status === 'WRONG_INPUT' ? (
                  <div style={{ textAlign: 'center', color: 'white' }}>
                    <AlertCircle size={48} color="#f59e0b" />
                    <h3 style={{ color: '#f59e0b', margin: '0.5rem 0' }}>Wrong input</h3>
                    <p style={{ margin: '0.5rem 0' }}>{result.error?.detail || 'Could not inspect this input image.'}</p>
                    {result.original_image && <ResultImage src={result.original_image} alt="Inspection evidence" style={{ maxWidth: '200px', opacity: 0.5 }} />}
                  </div>
                ) : (
                  <>
                    <div className={`status-badge ${result.status === 'PASS' ? 'status-pass' : result.status === 'FAIL' ? 'status-fail' : 'status-wrong-input'}`}>
                      {result.status}
                    </div>
                    <ResultImage
                      src={selectedResultView?.src}
                      alt={selectedResultView?.alt || 'Inspection visualization'}
                      className="preview-image"
                    />
                  </>
                )
              ) : (
                <ResultImage src={previewUrl} alt="Uploaded jerrycan" className="preview-image" />
              )}
            </div>
          )}
          {result && result.status !== 'WRONG_INPUT' && qualityScore(result) !== null && (
            <div className="result-summary" aria-label="Quality result details">
              <div>
                <span className="result-summary-label">Quality score</span>
                <strong className={result.status === 'FAIL' ? 'quality-fail' : 'quality-pass'}>
                  {qualityScore(result).toFixed(1)}%
                </strong>
              </div>
              <div className="result-summary-explanation">
                100% means no measured anomaly. Failure threshold: {displayedQualityBoundary}%.
                This relative quality index is not confidence, probability, or accuracy.
                <span>Raw model score: {result.raw_image_score.toFixed(4)}</span>
              </div>
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
          <div className="stat-value">{stats.pass_rate == null ? 'N/A' : `${stats.pass_rate.toFixed(1)}%`}</div>
        </div>
        <div className="stat-card">
          <h4>Passes</h4>
          <div className="stat-value" style={{ color: '#10b981' }}>{stats.passes}</div>
        </div>
        <div className="stat-card fail">
          <h4>Defects Found</h4>
          <div className="stat-value" style={{ color: '#ef4444' }}>{stats.faults}</div>
        </div>
        <div className="stat-card">
          <h4>Wrong Inputs</h4>
          <div className="stat-value" style={{ color: '#f59e0b' }}>{stats.wrong_inputs}</div>
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
            <option value="WRONG_INPUT">Wrong Input Only</option>
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
                setViewMode('defect');

                // Find first angle with data to focus on
                const firstAngle = Object.keys(session.angles)[0];
                if (firstAngle) setActiveAngle(firstAngle);

                setIsArchiveView(true);
                setActivePage('console');
              }}>
                <td>{new Date(session.timestamp).toLocaleString()}</td>
                <td><code style={{ fontSize: '0.75rem' }}>{session.id.split('-')[0]}...</code></td>
                <td style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                  {displayModelName(null, session.model_name)}
                </td>
                <td>
                  <span className={`status-row-badge ${session.overall_status === 'PASS' ? 'badge-pass' : session.overall_status === 'FAIL' ? 'badge-fail' : 'badge-wrong-input'}`}>
                    {session.overall_status}
                  </span>
                </td>
                <td>
                  <div style={{ display: 'flex', gap: '4px' }}>
                    {ANGLES.map(({ id, label }) => (
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
      await axios.post(`${BACKEND_BASE_URL}/settings`, newSettings);
    } catch (err) {
      console.error("Failed to persist rule change:", err);
    }
  };

  const deleteRule = async (id) => {
    const newSettings = { ...systemSettings, alerts: systemSettings.alerts.filter(r => r.id !== id) };
    setSystemSettings(newSettings);
    try {
      await axios.post(`${BACKEND_BASE_URL}/settings`, newSettings);
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
      await axios.post(`${BACKEND_BASE_URL}/settings`, newSettings);
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
        <div className="card" style={{ maxWidth: '700px', margin: '0 auto' }}>
          <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1.5rem' }}>
            <Mail size={18} color="var(--primary-color)" /> SMTP Server (Email Provider)
          </h3>
          <div className="smtp-form-grid">
            <div className="form-group">
              <label htmlFor="smtp-server">Server</label>
              <input id="smtp-server" type="text" placeholder="smtp.gmail.com" value={systemSettings.smtp.server} onChange={(e) => setSystemSettings({ ...systemSettings, smtp: { ...systemSettings.smtp, server: e.target.value } })} className="modal-input" />
            </div>
            <div className="form-group">
              <label htmlFor="smtp-port">Port</label>
              <input id="smtp-port" type="number" min="1" max="65535" value={systemSettings.smtp.port} onChange={(e) => setSystemSettings({ ...systemSettings, smtp: { ...systemSettings.smtp, port: parseInt(e.target.value) || 587 } })} className="modal-input" />
            </div>
            <div className="form-group form-group-full">
              <label htmlFor="smtp-user">User / Sender</label>
              <input id="smtp-user" type="email" placeholder="alerts@example.com" value={systemSettings.smtp.user} onChange={(e) => setSystemSettings({ ...systemSettings, smtp: { ...systemSettings.smtp, user: e.target.value } })} className="modal-input" />
            </div>
            <div className="form-group form-group-full">
              <label>SMTP Credential</label>
              <div className="modal-input smtp-readonly-field" title={systemSettings.smtp.password_configured ? 'Configured securely by environment' : 'Not configured — set SMTP_PASSWORD on the server'}>
                <Lock size={14} />
                <span>
                  {systemSettings.smtp.password_configured ? 'Configured securely by environment' : 'Not configured — set SMTP_PASSWORD on the server'}
                </span>
              </div>
            </div>
          </div>
          <div style={{ marginTop: '1.75rem', display: 'flex', justifyContent: 'flex-end' }}>
            <button onClick={saveSettings} disabled={settingsLoading} className="btn-primary" style={{ width: 'auto', padding: '0.75rem 1.5rem', fontSize: '0.85rem' }}>
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
