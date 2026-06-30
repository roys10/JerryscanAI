import { useState } from 'react';
import { Bell, Plus, Edit2, Trash2, Mail, Globe, Loader2 } from 'lucide-react';
import * as api from '../services/api';

export default function AlertsManager({ systemSettings, setSystemSettings }) {
  const [editingRule, setEditingRule] = useState(null);
  const [isRuleModalOpen, setIsRuleModalOpen] = useState(false);
  const [settingsLoading, setSettingsLoading] = useState(false);

  const saveSmtpSettings = async (e) => {
    if (e) e.preventDefault();
    setSettingsLoading(true);
    try {
      const response = await api.saveSettings(systemSettings);
      setSystemSettings(response.data.settings);
    } catch (err) {
      console.error("Failed to save settings:", err);
      alert("Failed to save settings.");
    } finally {
      setSettingsLoading(false);
    }
  };

  const addOrUpdateRule = async () => {
    const originalSettings = systemSettings;
    const updatedAlerts = editingRule.id
      ? systemSettings.alerts.map(r => r.id === editingRule.id ? editingRule : r)
      : [...systemSettings.alerts, { ...editingRule, id: Date.now().toString() }];

    const newSettings = { ...systemSettings, alerts: updatedAlerts };
    setSystemSettings(newSettings);
    setIsRuleModalOpen(false);
    setEditingRule(null);

    try {
      await api.saveSettings(newSettings);
    } catch (err) {
      console.error("Failed to persist rule change:", err);
      alert("Failed to save rule. Reverting changes.");
      setSystemSettings(originalSettings);
    }
  };

  const deleteRule = async (id) => {
    const originalSettings = systemSettings;
    const newSettings = { ...systemSettings, alerts: systemSettings.alerts.filter(r => r.id !== id) };
    setSystemSettings(newSettings);
    try {
      await api.saveSettings(newSettings);
    } catch (err) {
      console.error("Failed to persist rule deletion:", err);
      alert("Failed to delete rule. Reverting changes.");
      setSystemSettings(originalSettings);
    }
  };

  const toggleRule = async (id) => {
    const originalSettings = systemSettings;
    const newSettings = {
      ...systemSettings,
      alerts: systemSettings.alerts.map(r => r.id === id ? { ...r, enabled: !r.enabled } : r)
    };
    setSystemSettings(newSettings);
    try {
      await api.saveSettings(newSettings);
    } catch (err) {
      console.error("Failed to persist rule toggle:", err);
      alert("Failed to toggle rule. Reverting changes.");
      setSystemSettings(originalSettings);
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

  return (
    <div className="history-container" style={{ animation: 'fadeIn 0.4s ease-out' }}>
      {/* Custom Rules Management */}
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

      {/* Global SMTP Section */}
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
            <button onClick={saveSmtpSettings} disabled={settingsLoading} className="btn-primary" style={{ width: 'auto', padding: '0.5rem 1.5rem', fontSize: '0.8rem' }}>
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
              <input type="text" value={editingRule.name} onChange={e => setEditingRule({ ...editingRule, name: e.target.value })} className="modal-input" />
            </div>

            <div className="form-group" style={{ marginBottom: '1rem' }}>
              <label>Trigger Condition</label>
              <select value={editingRule.type} onChange={e => setEditingRule({ ...editingRule, type: e.target.value })} className="modal-input">
                <option value="consecutive_fails">Failure Streak</option>
                <option value="pass_rate">Pass Rate Drop</option>
              </select>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1rem' }}>
              <div className="form-group">
                <label>{editingRule.type === 'consecutive_fails' ? 'Items Count' : 'Min Pass %'}</label>
                <input type="number" value={editingRule.threshold} onChange={e => setEditingRule({ ...editingRule, threshold: parseFloat(e.target.value) || 0 })} className="modal-input" />
              </div>
              {editingRule.type === 'pass_rate' && (
                <div className="form-group">
                  <label>Sample Size (last X)</label>
                  <input type="number" value={editingRule.window} onChange={e => setEditingRule({ ...editingRule, window: parseInt(e.target.value) || 50 })} className="modal-input" />
                </div>
              )}
            </div>

            <div className="form-group" style={{ marginBottom: '1rem' }}>
              <label>Email Recipients</label>
              <input type="text" value={editingRule.emails.join(', ')} onChange={e => setEditingRule({ ...editingRule, emails: e.target.value.split(',').map(s => s.trim()).filter(s => s !== '') })} className="modal-input" />
            </div>

            <div className="form-group" style={{ marginBottom: '1.5rem' }}>
              <label>Webhook URL (Optional)</label>
              <input type="url" placeholder="https://..." value={editingRule.webhook_url} onChange={e => setEditingRule({ ...editingRule, webhook_url: e.target.value })} className="modal-input" />
            </div>

            <div className="form-group" style={{ marginBottom: '1.5rem' }}>
              <label>Status</label>
              <select value={editingRule.enabled ? 'enabled' : 'disabled'} onChange={e => setEditingRule({ ...editingRule, enabled: e.target.value === 'enabled' })} className="modal-input">
                <option value="enabled">Enabled</option>
                <option value="disabled">Disabled</option>
              </select>
            </div>

            <div style={{ display: 'flex', gap: '1rem', justifyContent: 'flex-end', alignItems: 'center', marginTop: '1.5rem' }}>
              <button className="btn-secondary" style={{ padding: '0.75rem 2rem', fontSize: '0.85rem', whiteSpace: 'nowrap' }} onClick={() => setIsRuleModalOpen(false)}>
                Cancel
              </button>
              <button className="btn-primary" style={{ width: 'auto', padding: '0.75rem 4rem', fontWeight: 600, whiteSpace: 'nowrap' }} onClick={addOrUpdateRule}>
                Save Rule
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
