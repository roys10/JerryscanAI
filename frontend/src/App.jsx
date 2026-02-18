
import { useState } from 'react';
import axios from 'axios';
import { Upload, Brain, CheckCircle, XCircle, AlertCircle, Loader2, Camera, RefreshCw } from 'lucide-react';
import './Inspection.css';

function App() {
  // State for each angle
  const [angleData, setAngleData] = useState({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Angle Selection State
  const [activeAngle, setActiveAngle] = useState('front');
  const angles = [
    { id: 'front', label: 'Front View' },
    { id: 'back', label: 'Back View' },
    { id: 'side_l', label: 'Side Left' },
    { id: 'side_r', label: 'Side Right' },
    { id: 'top', label: 'Top View' },
    { id: 'bottom', label: 'Bottom View' },
  ];

  // View Mode State
  const [viewMode, setViewMode] = useState('heatmap'); // 'heatmap' or 'segmentation'

  // Global Result State
  const [globalResult, setGlobalResult] = useState(null);

  // Get current angle's data or empty object
  const currentData = angleData[activeAngle] || {};
  const { selectedFile, previewUrl, result } = currentData;

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
    // 1. Identify valid uploads
    const anglesToInspect = angles.filter(a => angleData[a.id]?.selectedFile);

    if (anglesToInspect.length === 0) {
      setError("No images uploaded to inspect.");
      return;
    }

    setLoading(true);
    setError(null);
    setGlobalResult(null);

    try {
      // 2. Create Promises for all angles
      const promises = anglesToInspect.map(async (angle) => {
        const formData = new FormData();
        formData.append('file', angleData[angle.id].selectedFile);
        formData.append('angle_id', angle.id);

        try {
          const response = await axios.post('http://localhost:8000/inspect', formData, {
            headers: { 'Content-Type': 'multipart/form-data' },
          });
          return { id: angle.id, data: response.data, error: null };
        } catch (err) {
          console.error(`Error inspecting ${angle.id}:`, err);
          return { id: angle.id, data: null, error: err };
        }
      });

      // 3. Wait for all
      const results = await Promise.all(promises);

      // 4. Update State
      const newAngleData = { ...angleData };
      let anyFail = false;
      let anyPass = false;
      let errorMsg = null;

      results.forEach(res => {
        if (res.data) {
          newAngleData[res.id] = { ...newAngleData[res.id], result: res.data };
          if (res.data.status === 'FAIL') anyFail = true;
          if (res.data.status === 'PASS') anyPass = true;
        } else if (res.error) {
          // Network/Server error for this specific angle
          errorMsg = res.error.response?.data?.detail || "Inspection Failed";
        }
      });

      setAngleData(newAngleData);

      // 5. Determine Global Status
      if (errorMsg) {
        setError(`Partial Error: ${errorMsg}`);
      }

      if (anyFail) {
        setGlobalResult("FAIL");
      } else if (anyPass) {
        setGlobalResult("PASS");
      } else {
        setGlobalResult("UNAVAILABLE"); // All were unavailable or empty
      }

    } catch (err) {
      console.error(err);
      setError('System Error: Batch inspection failed.');
    } finally {
      setLoading(false);
    }
  };

  const clearState = () => {
    setAngleData({});
    setGlobalResult(null);
    setError(null);
  }

  // Calculate overall stats
  const inspectedCount = Object.values(angleData).filter(d => d.selectedFile).length;

  return (
    <div className="inspection-container">
      <header className="header">
        <div className="title">
          <Brain size={28} color="var(--primary-color)" />
          <span>JerryScan AI <span style={{ fontSize: '0.8em', color: 'var(--text-muted)', fontWeight: '400' }}>| Inspection Console</span></span>
        </div>
        <div style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>
          <strong>Operator:</strong> Admin &nbsp;|&nbsp; <strong>Ready to Inspect:</strong> {inspectedCount} Angles
        </div>
      </header>

      {/* GLOBAL STATUS BANNER */}
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
                const hasData = angleData[angle.id]?.selectedFile;
                const result = angleData[angle.id]?.result;
                const status = result?.status;

                let statusColor = '#9ca3af'; // Grey (Default)
                if (status === 'PASS') statusColor = '#10b981';
                if (status === 'FAIL') statusColor = '#ef4444';
                if (status === 'UNAVAILABLE') statusColor = '#f59e0b'; // Amber

                return (
                  <div
                    key={angle.id}
                    className={`angle-btn ${activeAngle === angle.id ? 'active' : ''}`}
                    onClick={() => setActiveAngle(angle.id)}
                    style={{ position: 'relative' }}
                  >
                    <Camera size={20} style={{ marginBottom: '0.25rem' }} />
                    <div>{angle.label}</div>

                    {/* Status Indicator */}
                    {hasData && (
                      <div style={{
                        position: 'absolute', top: 6, right: 6, width: 10, height: 10, borderRadius: '50%',
                        backgroundColor: statusColor,
                        border: '1px solid white'
                      }} title={status || "Uploaded"}></div>
                    )}
                  </div>
                );
              })}
            </div>
            <div style={{ marginTop: '1rem', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
              * Select an angle to upload. Green dot = Pass, Red = Fail, Orange = Model Unavailable.
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
              {loading ? 'Inspecting All on 6 Cameras...' : `Run Inspection (${inspectedCount})`}
            </button>

            <div style={{ marginTop: '1rem' }}>
              <button
                className="btn-secondary"
                onClick={clearState}
              >
                <RefreshCw size={16} style={{ marginRight: '0.5rem', verticalAlign: 'middle' }} /> Reset Session
              </button>
            </div>
          </div>

          {error && (
            <div style={{ color: '#ef4444', background: '#fee2e2', padding: '1rem', borderRadius: '0.5rem', display: 'flex', gap: '0.5rem' }}>
              <AlertCircle size={20} /> <span style={{ fontSize: '0.9rem' }}>{error}</span>
            </div>
          )}
        </div>

        {/* Right Panel: Viewport */}
        <div className="card" style={{ minHeight: '600px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
            <h3>Inspection View - {angles.find(a => a.id === activeAngle)?.label}</h3>

            {result && result.status !== 'UNAVAILABLE' && (
              <div style={{ display: 'flex', gap: '0.5rem', background: '#f3f4f6', padding: '0.25rem', borderRadius: '0.375rem' }}>
                <button
                  onClick={() => setViewMode('heatmap')}
                  style={{
                    border: 'none', background: viewMode === 'heatmap' ? 'white' : 'transparent',
                    padding: '0.25rem 0.75rem', borderRadius: '0.25rem', fontSize: '0.875rem', fontWeight: 500,
                    boxShadow: viewMode === 'heatmap' ? '0 1px 2px rgba(0,0,0,0.1)' : 'none', cursor: 'pointer'
                  }}
                >Anomaly Map</button>
                <button
                  onClick={() => setViewMode('segmentation')}
                  style={{
                    border: 'none', background: viewMode === 'segmentation' ? 'white' : 'transparent',
                    padding: '0.25rem 0.75rem', borderRadius: '0.25rem', fontSize: '0.875rem', fontWeight: 500,
                    boxShadow: viewMode === 'segmentation' ? '0 1px 2px rgba(0,0,0,0.1)' : 'none', cursor: 'pointer'
                  }}
                >Pred Mask</button>
              </div>
            )}
          </div>

          {!previewUrl ? (
            <div
              className="upload-zone"
              onDragOver={handleDragOver}
              onDrop={handleDrop}
              onClick={() => document.getElementById('fileInput').click()}
            >
              <div style={{ background: 'white', padding: '1rem', borderRadius: '50%', boxShadow: '0 4px 6px rgba(0,0,0,0.05)', marginBottom: '1.5rem' }}>
                <Upload size={32} className="upload-icon" style={{ marginBottom: 0 }} />
              </div>
              <h4 style={{ margin: 0, fontSize: '1.1rem', color: 'var(--text-main)' }}>Upload Image</h4>
              <p style={{ color: 'var(--text-muted)', marginTop: '0.5rem' }}>
                Drag & drop or click to browse
              </p>
              <input
                id="fileInput"
                type="file"
                hidden
                accept="image/*"
                onChange={handleFileChange}
              />
            </div>
          ) : (
            <div className="preview-container">
              {result ? (
                <>
                  {result.status === 'UNAVAILABLE' ? (
                    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', color: 'white' }}>
                      <AlertCircle size={48} color="#f59e0b" style={{ marginBottom: '1rem' }} />
                      <h3 style={{ color: '#f59e0b' }}>Model Unavailable</h3>
                      <p>No AI model loaded for this angle.</p>
                      <img src={previewUrl} style={{ maxWidth: '200px', marginTop: '1rem', opacity: 0.5 }} />
                    </div>
                  ) : (
                    <>
                      <div className={`status-badge ${result.status === 'PASS' ? 'status-pass' : 'status-fail'}`}>
                        {result.status} (Score: {result.score_percentage?.toFixed(2)}%)
                      </div>
                      <img
                        src={viewMode === 'heatmap' ? result.heatmap_image : result.segmentation_image}
                        alt="Result"
                        className="preview-image"
                      />
                      <div className="image-label" style={{ position: 'absolute', bottom: 10, left: 10, background: 'rgba(0,0,0,0.7)', color: 'white', padding: '4px 8px', borderRadius: '4px', fontSize: '12px' }}>
                        {viewMode === 'heatmap' ? 'Image + Anomaly Map' : 'Image + Pred Mask'}
                      </div>
                    </>
                  )}
                </>
              ) : (
                <img src={previewUrl} alt="Preview" className="preview-image" />
              )}
            </div>
          )}

          <div style={{ marginTop: '1rem', display: 'flex', gap: '2rem', justifyContent: 'center', color: 'var(--text-muted)', fontSize: '0.9rem' }}>
            {(result && result.status !== 'UNAVAILABLE') ? (
              viewMode === 'heatmap' ? (
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <span style={{ fontSize: '0.8rem' }}>Normal</span>
                  <div style={{
                    width: 100, height: 12, borderRadius: 6,
                    background: 'linear-gradient(90deg, blue, cyan, yellow, red)'
                  }}></div>
                  <span style={{ fontSize: '0.8rem' }}>Anomaly</span>
                </div>
              ) : (
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <div style={{ width: 12, height: 12, borderRadius: '50%', border: '2px solid red' }}></div> Pred Mask
                </div>
              )
            ) : null}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
