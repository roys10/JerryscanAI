import { CheckCircle, XCircle, AlertCircle, History, Brain, Upload, Camera, Loader2, RefreshCw } from 'lucide-react';
import { ANGLES } from '../constants';
import AngleSelector from './AngleSelector';
import ImageViewport from './ImageViewport';

export default function InspectionConsole({
  globalResult,
  isArchiveView,
  selectedSession,
  angleData,
  setAngleData,
  activeAngle,
  setActiveAngle,
  viewMode,
  setViewMode,
  selectedModel,
  setSelectedModel,
  availableModels,
  loading,
  error,
  setError,
  setGlobalResult,
  setActivePage,
  setIsArchiveView,
  setSelectedSession,
  runBatchInspection,
  clearState,
}) {
  const currentData = angleData[activeAngle] || {};
  const { selectedFile, previewUrl, result } = currentData;
  const inspectedCount = Object.values(angleData).filter(d => d.selectedFile || d.result).length;

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
    setAngleData(prev => ({
      ...prev,
      [activeAngle]: {
        selectedFile: file,
        previewUrl: URL.createObjectURL(file),
        result: null
      }
    }));
    setGlobalResult(null);
    setError(null);
    setViewMode('heatmap');
  };

  return (
    <>
      {/* GLOBAL STATUS BANNER */}
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

          <AngleSelector
            angles={ANGLES}
            activeAngle={activeAngle}
            setActiveAngle={setActiveAngle}
            angleData={angleData}
            isArchiveView={isArchiveView}
          />

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
        <ImageViewport
          activeAngle={activeAngle}
          angles={ANGLES}
          result={result}
          previewUrl={previewUrl}
          viewMode={viewMode}
          setViewMode={setViewMode}
          isArchiveView={isArchiveView}
          handleFileChange={handleFileChange}
          handleDragOver={handleDragOver}
          handleDrop={handleDrop}
        />
      </div>
    </>
  );
}
