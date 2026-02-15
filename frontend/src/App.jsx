
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
    setError(null);
    setViewMode('heatmap'); // Reset view mode
  };

  const handleInspect = async () => {
    if (!selectedFile) return;

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('angle_id', activeAngle);

    try {
      const response = await axios.post('http://localhost:8000/inspect', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      // Update result for THIS angle
      setAngleData(prev => ({
        ...prev,
        [activeAngle]: {
          ...prev[activeAngle],
          result: response.data
        }
      }));

    } catch (err) {
      console.error(err);
      setError('Inspection failed. Ensure backend is running.');
    } finally {
      setLoading(false);
    }
  };

  const clearState = () => {
    setAngleData(prev => {
      const newState = { ...prev };
      delete newState[activeAngle]; // Clear data for current angle
      return newState;
    });
    setError(null);
  }

  // Calculate overall stats (optional, for fun)
  const inspectedCount = Object.values(angleData).filter(d => d.result).length;

  return (
    <div className="inspection-container">
      <header className="header">
        <div className="title">
          <Brain size={28} color="var(--primary-color)" />
          <span>JerryScan AI <span style={{ fontSize: '0.8em', color: 'var(--text-muted)', fontWeight: '400' }}>| Inspection Console</span></span>
        </div>
        <div style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>
          <strong>Operator:</strong> Admin &nbsp;|&nbsp; <strong>Inspected Angles:</strong> {inspectedCount}/{angles.length}
        </div>
      </header>

      <div className="main-content">
        {/* Left Panel: Controls & Angles */}
        <div className="control-panel">
          <div className="card">
            <h3>Camera Selection</h3>
            <div className="angle-grid">
              {angles.map((angle) => {
                const hasData = angleData[angle.id]?.selectedFile;
                const status = angleData[angle.id]?.result?.status;

                return (
                  <div
                    key={angle.id}
                    className={`angle-btn ${activeAngle === angle.id ? 'active' : ''}`}
                    onClick={() => setActiveAngle(angle.id)}
                    style={{ position: 'relative' }}
                  >
                    <Camera size={20} style={{ marginBottom: '0.25rem' }} />
                    <div>{angle.label}</div>

                    {/* Mini status indicator dot */}
                    {hasData && (
                      <div style={{
                        position: 'absolute', top: 6, right: 6, width: 8, height: 8, borderRadius: '50%',
                        backgroundColor: status === 'PASS' ? '#10b981' : status === 'FAIL' ? '#ef4444' : '#9ca3af'
                      }}></div>
                    )}
                  </div>
                );
              })}
            </div>
            <div style={{ marginTop: '1rem', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
              * Select an angle to view or upload its inspection image.
            </div>
          </div>

          <div className="card">
            <h3>Actions</h3>
            <button
              className="btn-primary"
              onClick={handleInspect}
              disabled={loading || !selectedFile}
            >
              {loading ? <Loader2 className="spin" size={20} /> : <Brain size={20} />}
              {loading ? 'Analyzing...' : 'Run Inspection'}
            </button>

            <div style={{ marginTop: '1rem' }}>
              <button
                className="btn-secondary"
                onClick={clearState}
              >
                <RefreshCw size={16} style={{ marginRight: '0.5rem', verticalAlign: 'middle' }} /> Reset Current View
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

            {result && (
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
                  <div className={`status-badge ${result.status === 'PASS' ? 'status-pass' : 'status-fail'}`}>
                    {result.status} (Score: {result.score.toFixed(2)})
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
              ) : (
                <img src={previewUrl} alt="Preview" className="preview-image" />
              )}
            </div>
          )}

          <div style={{ marginTop: '1rem', display: 'flex', gap: '2rem', justifyContent: 'center', color: 'var(--text-muted)', fontSize: '0.9rem' }}>
            {viewMode === 'heatmap' ? (
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
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
