import { Upload, AlertCircle } from 'lucide-react';

export default function ImageViewport({
  activeAngle,
  angles,
  result,
  previewUrl,
  viewMode,
  setViewMode,
  isArchiveView,
  handleFileChange,
  handleDragOver,
  handleDrop,
}) {
  return (
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
  );
}
