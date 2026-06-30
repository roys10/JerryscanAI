import { Camera } from 'lucide-react';

export default function AngleSelector({ angles, activeAngle, setActiveAngle, angleData, isArchiveView }) {
  return (
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
  );
}
