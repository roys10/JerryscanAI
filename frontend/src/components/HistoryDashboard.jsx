import { Filter, History } from 'lucide-react';

export default function HistoryDashboard({
  history,
  stats,
  filter,
  setFilter,
  setAngleData,
  setGlobalResult,
  setSelectedSession,
  setActiveAngle,
  setIsArchiveView,
  setActivePage,
}) {
  const openSession = (session) => {
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

    const firstAngle = Object.keys(session.angles)[0];
    if (firstAngle) setActiveAngle(firstAngle);

    setIsArchiveView(true);
    setActivePage('console');
  };

  return (
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
              <tr key={session.id} onClick={() => openSession(session)}>
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
                    {[['G01', 'G01'], ['G02', 'G02'], ['G03', 'G03'], ['G04', 'G04']].map(([id, label]) => (
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
}
