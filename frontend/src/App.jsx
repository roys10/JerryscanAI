import { useState, useEffect } from 'react';
import axios from 'axios';
import { Upload, Brain, CheckCircle, XCircle, AlertCircle, Loader2, Camera, RefreshCw, History, LayoutDashboard, Search, Filter } from 'lucide-react';
import './Inspection.css';
import './History.css';

function App() {
  // Navigation
  const [activePage, setActivePage] = useState('console'); // 'console' or 'history'

  // Console State
  const [angleData, setAngleData] = useState({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // History State
  const [history, setHistory] = useState([]);
  const [stats, setStats] = useState({ total: 0, passes: 0, fails: 0, pass_rate: 0 });
  const [filter, setFilter] = useState('all'); // 'all', 'PASS', 'FAIL'
  const [selectedSession, setSelectedSession] = useState(null);

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
    if (activePage === 'history') {
      fetchHistory();
      fetchStats();
    }
  }, [activePage, filter]);

  const fetchHistory = async () => {
    try {
      const url = filter === 'all'
        ? 'http://localhost:8000/history'
        : `http://localhost:8000/history?status=${filter}`;
      const response = await axios.get(url);
      setHistory(response.data);
    } catch (err) {
      console.error("Failed to fetch history:", err);
    }
  };

  const fetchStats = async () => {
    try {
      const response = await axios.get('http://localhost:8000/stats');
      setStats(response.data);
    } catch (err) {
      console.error("Failed to fetch stats:", err);
    }
  };

  const simulateTrigger = async () => {
    setLoading(true);
    try {
      const response = await axios.post('http://localhost:8000/simulate-trigger');
      // After simulation, maybe show the result in the console? 
      // For now, let's just refresh history if we are there.
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
              previewUrl: response.data.angles[a.id].original_image // If backend returns it
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
      const response = await axios.post('http://localhost:8000/inspect-batch', formData, {
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


  const renderConsole = () => (
    <>
      {/* GLOBAL STATUS BANNER (Old design restored) */}
      {globalResult && (
        <div className={`global-banner ${globalResult === 'PASS' ? 'banner-pass' : globalResult === 'FAIL' ? 'banner-fail' : 'banner-neutral'}`}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '1rem' }}>
            {globalResult === 'PASS' ? <CheckCircle size={32} /> : globalResult === 'FAIL' ? <XCircle size={32} /> : <AlertCircle size={32} />}
            <span>JERRYCAN STATUS: {globalResult}</span>
          </div>
          {globalResult === 'FAIL' && <div style={{ fontSize: '0.9rem', marginTop: '0.25rem', opacity: 0.9 }}>Defects detected in one or more angles. Check details below.</div>}
        </div>
      )}

      <div className="main-content">
        {/* Left Panel: Controls & Angles */}
        <div className="control-panel">
          <div className="card">
            <h3>Camera Selection</h3>
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
            <div className="upload-zone" onClick={() => document.getElementById('fileInput').click()}>
              <Upload size={32} color="var(--primary-color)" />
              <h4>Upload Image</h4>
              <input id="fileInput" type="file" hidden accept="image/*" onChange={handleFileChange} />
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
              <th>Overall Status</th>
              <th>Angles Checked</th>
              <th>Action</th>
            </tr>
          </thead>
          <tbody>
            {history.map(session => (
              <tr key={session.id} onClick={() => {
                setAngleData(session.angles);
                setGlobalResult(session.overall_status);
                setActivePage('console');
              }}>
                <td>{new Date(session.timestamp).toLocaleString()}</td>
                <td><code style={{ fontSize: '0.75rem' }}>{session.id.split('-')[0]}...</code></td>
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


  return (
    <div className="inspection-container">
      <header className="header">
        <div style={{ display: 'flex', alignItems: 'center', gap: '2rem' }}>
          <div className="title">
            <Brain size={28} color="var(--primary-color)" />
            <span>JerryScan AI</span>
          </div>
          <nav style={{ display: 'flex', gap: '1.5rem', borderLeft: '1px solid var(--border-color)', paddingLeft: '1.5rem' }}>
            <div
              onClick={() => setActivePage('console')}
              style={{
                cursor: 'pointer',
                fontWeight: activePage === 'console' ? '700' : '500',
                color: activePage === 'console' ? 'var(--primary-color)' : 'var(--text-muted)',
                display: 'flex', alignItems: 'center', gap: '0.5rem'
              }}
            >
              <LayoutDashboard size={18} /> Manual Inspection
            </div>
            <div
              onClick={() => setActivePage('history')}
              style={{
                cursor: 'pointer',
                fontWeight: activePage === 'history' ? '700' : '500',
                color: activePage === 'history' ? 'var(--primary-color)' : 'var(--text-muted)',
                display: 'flex', alignItems: 'center', gap: '0.5rem'
              }}
            >
              <History size={18} /> History & Analytics
            </div>
          </nav>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <button
            className="btn-secondary"
            onClick={simulateTrigger}
            disabled={loading}
            style={{ width: 'auto', padding: '0.5rem 1.25rem', fontSize: '0.85rem' }}
          >
            {loading ? <Loader2 className="spin" size={16} /> : <RefreshCw size={16} />}
            &nbsp; Simulate Remote Trigger
          </button>
          <div style={{ color: 'var(--text-muted)', fontSize: '0.8rem', borderLeft: '1px solid var(--border-color)', paddingLeft: '1rem' }}>
            <strong>Operator:</strong> Admin
          </div>
        </div>
      </header>

      {activePage === 'console' ? renderConsole() : renderHistory()}
    </div>
  );
}

export default App;
