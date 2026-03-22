import { Brain, LayoutDashboard, History, Bell, XCircle, RefreshCw, Loader2 } from 'lucide-react';

export default function Navbar({
  activePage,
  setActivePage,
  isArchiveView,
  setIsArchiveView,
  setSelectedSession,
  clearState,
  simulateTrigger,
  loading,
}) {
  return (
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
}
