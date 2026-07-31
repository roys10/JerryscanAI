import { useEffect, useMemo, useState } from 'react'
import { Activity, Archive, Check, ChevronRight, FlaskConical, FolderSearch, Upload, X } from 'lucide-react'

const api = async (path, options = {}) => {
  const response = await fetch(path, {
    headers: options.body instanceof FormData ? undefined : { 'Content-Type': 'application/json' },
    ...options,
  })
  const value = await response.json().catch(() => ({}))
  if (!response.ok) throw new Error(value.detail || 'Request failed')
  return value
}

const metricText = (metric, percent = false) => {
  if (!metric?.available) return 'Not available'
  return percent ? `${(metric.value * 100).toFixed(2)}%` : String(metric.value)
}

function Status({ model }) {
  const ready = model.status !== 'incomplete'
  return <span className={`status ${ready ? 'ready' : 'incomplete'}`}>{ready ? <Check size={14} /> : <X size={14} />}{model.status.replaceAll('_', ' ')}</span>
}

function Models({ models, refresh }) {
  const [mode, setMode] = useState('path')
  const [form, setForm] = useState({ checkpoint_path: '', display_name: '', root: '' })
  const [message, setMessage] = useState('')
  const [editing, setEditing] = useState(null)
  const [contract, setContract] = useState({})
  const edit = model => {
    setEditing(model.id)
    setContract({ angle: model.angle || '', manifest_sha256: model.manifest_sha256 || '', preprocessing_config_path: model.preprocessing_config_path || '', derivative_root: model.derivative_root || '', image_threshold: model.image_threshold ?? '' })
  }
  const saveContract = async (event, modelId) => {
    event.preventDefault(); setMessage('')
    const body = { ...contract }
    if (body.image_threshold === '') delete body.image_threshold
    else body.image_threshold = Number(body.image_threshold)
    try { await api(`/api/models/${modelId}`, { method: 'PATCH', body: JSON.stringify(body) }); setEditing(null); setMessage('Contract updated. Manually entered thresholds are marked unverified.'); refresh() }
    catch (error) { setMessage(error.message) }
  }
  const submitPath = async (event) => {
    event.preventDefault(); setMessage('')
    try {
      await api('/api/models/import', { method: 'POST', body: JSON.stringify(form) })
      setMessage('Model registered. Missing contract fields are shown below.'); refresh()
    } catch (error) { setMessage(error.message) }
  }
  const discover = async (event) => {
    event.preventDefault(); setMessage('')
    try { await api('/api/models/discover', { method: 'POST', body: JSON.stringify({ root: form.root }) }); setMessage('Discovery complete.'); refresh() }
    catch (error) { setMessage(error.message) }
  }
  const upload = async (event) => {
    event.preventDefault(); setMessage('Uploading…')
    try { await api('/api/models/upload', { method: 'POST', body: new FormData(event.currentTarget) }); setMessage('Upload registered.'); refresh(); event.currentTarget.reset() }
    catch (error) { setMessage(error.message) }
  }
  return <section>
    <div className="heading"><div><p className="eyebrow">Model registry</p><h2>PatchCore models</h2><p>Register files already on this machine or upload a bundle. The lab never guesses missing preprocessing or calibration.</p></div></div>
    <div className="segmented">
      <button className={mode === 'path' ? 'active' : ''} onClick={() => setMode('path')}><Archive size={16}/>Import path</button>
      <button className={mode === 'discover' ? 'active' : ''} onClick={() => setMode('discover')}><FolderSearch size={16}/>Discover folder</button>
      <button className={mode === 'upload' ? 'active' : ''} onClick={() => setMode('upload')}><Upload size={16}/>Upload</button>
    </div>
    {mode === 'path' && <form className="panel form-grid" onSubmit={submitPath}>
      <label>Checkpoint path<input required value={form.checkpoint_path} onChange={e => setForm({...form, checkpoint_path: e.target.value})} placeholder="/models/run/G01.ckpt" /></label>
      <label>Display name<input value={form.display_name} onChange={e => setForm({...form, display_name: e.target.value})} placeholder="Optional" /></label>
      <button className="primary">Register model</button>
    </form>}
    {mode === 'discover' && <form className="panel inline" onSubmit={discover}><label>Models folder<input required value={form.root} onChange={e => setForm({...form, root: e.target.value})} placeholder="/models" /></label><button className="primary">Discover</button></form>}
    {mode === 'upload' && <form className="panel form-grid" onSubmit={upload}>
      <label>Display name<input name="display_name" required /></label><label>Checkpoint (.ckpt)<input name="checkpoint" type="file" accept=".ckpt" required /></label>
      <label>Training metadata<input name="metadata" type="file" accept=".json" /></label><label>Preprocessing config<input name="preprocessing_config" type="file" accept=".json" /></label>
      <button className="primary">Upload bundle</button>
    </form>}
    {message && <p className="notice">{message}</p>}
    <div className="model-list">{models.map(model => <article className="model-card" key={model.id}>
      <div><h3>{model.display_name}</h3><p>{model.angle || 'Angle missing'} · {model.preprocessing_id || 'Preprocessing missing'}</p></div><Status model={model}/>
      <dl><div><dt>Artifact</dt><dd>{(model.checkpoint_size_bytes / 1024 ** 2).toFixed(1)} MiB</dd></div><div><dt>Image threshold</dt><dd>{model.image_threshold ?? 'Not declared'}</dd></div></dl>
      {model.issues.length > 0 && <ul className="issues">{model.issues.map(issue => <li key={issue}>{issue.replaceAll('_', ' ')}</li>)}</ul>}
      <button className="secondary" onClick={() => edit(model)}>Edit contract</button>
      {editing === model.id && <form className="contract-form" onSubmit={event => saveContract(event, model.id)}>
        <label>Camera angle<input value={contract.angle} onChange={e => setContract({...contract, angle: e.target.value})}/></label>
        <label>Manifest SHA-256<input value={contract.manifest_sha256} onChange={e => setContract({...contract, manifest_sha256: e.target.value})}/></label>
        <label>Preprocessing config path<input value={contract.preprocessing_config_path} onChange={e => setContract({...contract, preprocessing_config_path: e.target.value})}/></label>
        <label>Verified derivative root<input value={contract.derivative_root} onChange={e => setContract({...contract, derivative_root: e.target.value})}/></label>
        <label>Image threshold<input type="number" step="any" value={contract.image_threshold} onChange={e => setContract({...contract, image_threshold: e.target.value})} placeholder="Optional; manual/unverified"/></label>
        <div><button className="primary">Save contract</button><button type="button" className="secondary" onClick={() => setEditing(null)}>Cancel</button></div>
      </form>}
    </article>)}</div>
  </section>
}

function NewComparison({ models, onCreated }) {
  const ready = models.filter(model => model.status !== 'incomplete')
  const [selected, setSelected] = useState([])
  const [form, setForm] = useState({ name: '', source_root: '', manifest: '', split: 'val', image_count: 50, seed: 42, force_live_preprocessing: false, locked_test_confirmation: '' })
  const [message, setMessage] = useState('')
  const toggle = id => setSelected(items => items.includes(id) ? items.filter(item => item !== id) : items.length < 4 ? [...items, id] : items)
  const submit = async event => {
    event.preventDefault(); setMessage('Validating comparison…')
    try {
      const result = await api('/api/comparisons', { method: 'POST', body: JSON.stringify({ ...form, model_ids: selected, image_count: Number(form.image_count), seed: Number(form.seed) }) })
      onCreated(result.comparison_id)
    } catch (error) { setMessage(error.message) }
  }
  return <section><div className="heading"><div><p className="eyebrow">New comparison</p><h2>Compare the same original cans</h2><p>Select one to four PatchCore models. Each receives its declared preprocessed representation of the same frozen sample IDs.</p></div><span className="counter">{selected.length}/4 models</span></div>
    <form onSubmit={submit}>
      <div className="select-models">{ready.map(model => <button type="button" key={model.id} className={selected.includes(model.id) ? 'selected' : ''} onClick={() => toggle(model.id)}><span><strong>{model.display_name}</strong><small>{model.preprocessing_id}</small></span>{selected.includes(model.id) && <Check/>}</button>)}</div>
      {ready.length === 0 && <div className="empty">Register a complete PatchCore bundle first.</div>}
      <div className="panel form-grid">
        <label>Comparison name<input value={form.name} onChange={e => setForm({...form, name: e.target.value})} placeholder="Optional" /></label>
        <label>Original image root<input required value={form.source_root} onChange={e => setForm({...form, source_root: e.target.value})} placeholder="/data/raw/G01" /></label>
        <label>Frozen manifest<input required value={form.manifest} onChange={e => setForm({...form, manifest: e.target.value})} placeholder="/repo/data_manifests/G01/split_v2.csv" /></label>
        <label>Dataset split<select value={form.split} onChange={e => setForm({...form, split: e.target.value})}><option value="val">Validation (recommended)</option><option value="train">Train — diagnostics only</option><option value="test">Locked test</option></select></label>
        <label>Image count<input type="number" min="1" value={form.image_count} onChange={e => setForm({...form, image_count: e.target.value})} /></label>
        <label>Selection seed<input type="number" value={form.seed} onChange={e => setForm({...form, seed: e.target.value})} /></label>
        {form.split === 'test' && <label className="danger-field">Type RUN LOCKED TEST<input value={form.locked_test_confirmation} onChange={e => setForm({...form, locked_test_confirmation: e.target.value})} /></label>}
        <label className="check"><input type="checkbox" checked={form.force_live_preprocessing} onChange={e => setForm({...form, force_live_preprocessing: e.target.checked})}/><span><strong>Force live preprocessing</strong><small>Bypass verified caches and measure end-to-end time.</small></span></label>
      </div>
      <div className="actions"><button className="primary large" disabled={selected.length === 0}>Run comparison <ChevronRight size={18}/></button><p>{message}</p></div>
    </form>
  </section>
}

function Metric({ label, value, note }) { return <div className="metric"><span>{label}</span><strong>{value}</strong>{note && <small>{note}</small>}</div> }

function Results({ comparisons, selectedId, selectComparison }) {
  const [detail, setDetail] = useState(null)
  const [sampleId, setSampleId] = useState(null)
  useEffect(() => {
    if (!selectedId) return
    let active = true
    const load = async () => { const value = await api(`/api/comparisons/${selectedId}`); if (active) { setDetail(value); if (!sampleId && value.results[0]) setSampleId(value.results[0].sample_id) } }
    load(); const timer = setInterval(load, 2500)
    return () => { active = false; clearInterval(timer) }
  }, [selectedId, sampleId])
  const sampleRows = useMemo(() => detail?.results.filter(row => row.sample_id === sampleId) || [], [detail, sampleId])
  return <section><div className="heading"><div><p className="eyebrow">Saved results</p><h2>Comparison history</h2><p>Runs persist independently of this browser and can resume after interruption.</p></div></div>
    <div className="results-layout"><aside className="run-list">{comparisons.map(run => <button key={run.comparison_id} className={run.comparison_id === selectedId ? 'active' : ''} onClick={() => selectComparison(run.comparison_id)}><span><strong>{run.name}</strong><small>{run.image_count} images · seed {run.seed}</small></span><em>{run.state}</em></button>)}</aside>
      <div>{!detail ? <div className="empty">Select a saved comparison.</div> : <>
        <div className="run-header"><div><h3>{detail.config.name}</h3><p>{detail.config.split} · {detail.status.completed}/{detail.status.total || '?'} evaluations</p></div><span className={`status ${detail.status.state === 'completed' ? 'ready' : 'incomplete'}`}><Activity size={14}/>{detail.status.state}</span></div>
        {detail.summary && Object.entries(detail.summary).map(([modelId, summary]) => <article className="summary" key={modelId}><h3>{modelId}</h3>{summary.excluded_unpaired_count > 0 && <p className="error">Incomplete paired coverage: {summary.excluded_unpaired_count} samples excluded. Resume to retry errors.</p>}<div className="metric-grid"><Metric label="Paired samples" value={summary.paired_sample_count}/><Metric label="False-positive rate" value={metricText(summary.false_positive_rate, true)} note={summary.false_positive_rate.reason}/><Metric label="Median raw score" value={summary.raw_score?.median.toFixed(5) ?? 'N/A'}/><Metric label="p99 raw score" value={summary.raw_score?.p99.toFixed(5) ?? 'N/A'}/><Metric label="p95 total latency" value={summary.total_ms ? `${summary.total_ms.p95.toFixed(1)} ms` : 'N/A'}/><Metric label="AUROC" value={metricText(summary.auroc)} note={summary.auroc.reason}/><Metric label="Recall" value={metricText(summary.recall)} note={summary.recall.reason}/></div></article>)}
        {detail.results.length > 0 && <><div className="sample-toolbar"><label>Sample explorer<select value={sampleId || ''} onChange={e => setSampleId(e.target.value)}>{[...new Set(detail.results.map(row => row.sample_id))].map(id => <option key={id}>{id}</option>)}</select></label></div>
          <div className="original"><h4>Original camera image</h4><img src={`/api/comparisons/${selectedId}/originals/${sampleId}`} /></div>
          <div className="sample-grid">{sampleRows.map(row => <article key={row.model_id}><h4>{row.model_id}</h4>{row.status === 'completed' ? <><div className="images"><figure><img src={`/api/comparisons/${selectedId}/assets/${row.input_asset}`} /><figcaption>Actual model input</figcaption></figure><figure><img src={`/api/comparisons/${selectedId}/assets/${row.heatmap_asset}`} /><figcaption>Display heatmap</figcaption></figure></div><dl><div><dt>Raw score</dt><dd>{row.raw_image_score.toFixed(6)}</dd></div><div><dt>Decision</dt><dd>{row.prediction || 'Not calibrated'}</dd></div><div><dt>Preprocessing</dt><dd>{row.preprocessing_source}</dd></div><div><dt>Total</dt><dd>{row.total_ms.toFixed(1)} ms</dd></div></dl></> : <p className="error">{row.error}</p>}</article>)}</div></>}
      </>}</div></div>
  </section>
}

function SingleImage({ models }) {
  const ready = models.filter(model => model.status !== 'incomplete')
  const [modelId, setModelId] = useState('')
  const [result, setResult] = useState(null)
  const [message, setMessage] = useState('')
  const submit = async event => {
    event.preventDefault(); setMessage('Running live preprocessing and PatchCore inference…'); setResult(null)
    try { const value = await api(`/api/single-image/${modelId}`, { method: 'POST', body: new FormData(event.currentTarget) }); setResult(value); setMessage('') }
    catch (error) { setMessage(error.message) }
  }
  const asset = filename => filename ? `/api/single-image/${result.id}/assets/${filename}` : null
  return <section><div className="heading"><div><p className="eyebrow">Single image</p><h2>Run one original camera image</h2><p>The selected model’s registered preprocessing always runs live before PatchCore inference.</p></div></div>
    <form className="panel form-grid" onSubmit={submit}>
      <label>PatchCore model<select required value={modelId} onChange={e => setModelId(e.target.value)}><option value="">Choose a model</option>{ready.map(model => <option value={model.id} key={model.id}>{model.display_name} · {model.preprocessing_id}</option>)}</select></label>
      <label>Original image<input required name="image" type="file" accept="image/*"/></label>
      <button className="primary">Run image</button>
    </form>{message && <p className="notice">{message}</p>}
    {result && <><div className="metric-grid single-metrics"><Metric label="Raw image score" value={result.raw_image_score.toFixed(6)}/><Metric label="Decision" value={result.prediction || 'Not calibrated'} note={result.image_threshold == null ? 'No verified image threshold' : `Threshold ${result.image_threshold}`}/><Metric label="Preprocessing" value={`${result.preprocessing_ms.toFixed(1)} ms`}/><Metric label="Inference" value={`${result.inference_ms.toFixed(1)} ms`}/><Metric label="Total" value={`${result.total_ms.toFixed(1)} ms`}/><Metric label="Quality flags" value={result.quality_flags.length ? result.quality_flags.join(', ') : 'None'}/></div>
      <div className="single-images">{[['Original', result.assets.original], ['Actual model input', result.assets.input], ['Mask', result.assets.mask], ['Heatmap', result.assets.heatmap]].filter(([, file]) => file).map(([label, file]) => <figure key={label}><img src={asset(file)}/><figcaption>{label}</figcaption></figure>)}</div></>}
  </section>
}

export default function App() {
  const [tab, setTab] = useState('compare')
  const [models, setModels] = useState([])
  const [comparisons, setComparisons] = useState([])
  const [selectedRun, setSelectedRun] = useState(null)
  const refresh = async () => { setModels(await api('/api/models')); setComparisons(await api('/api/comparisons')) }
  useEffect(() => {
    let active = true
    const load = async () => {
      const [nextModels, nextComparisons] = await Promise.all([api('/api/models'), api('/api/comparisons')])
      if (active) { setModels(nextModels); setComparisons(nextComparisons) }
    }
    const initial = setTimeout(load, 0)
    const timer = setInterval(() => api('/api/comparisons').then(value => active && setComparisons(value)), 3000)
    return () => { active = false; clearTimeout(initial); clearInterval(timer) }
  }, [])
  const created = id => { setSelectedRun(id); setTab('results'); refresh() }
  return <><header><div className="brand"><div className="mark"><FlaskConical/></div><div><strong>Jerryscan</strong><span>Model Lab · PatchCore</span></div></div><nav><button className={tab === 'compare' ? 'active' : ''} onClick={() => setTab('compare')}>New comparison</button><button className={tab === 'single' ? 'active' : ''} onClick={() => setTab('single')}>Single image</button><button className={tab === 'results' ? 'active' : ''} onClick={() => setTab('results')}>Results</button><button className={tab === 'models' ? 'active' : ''} onClick={() => setTab('models')}>Models <span>{models.length}</span></button></nav></header>
    <main>{tab === 'models' && <Models models={models} refresh={refresh}/>} {tab === 'compare' && <NewComparison models={models} onCreated={created}/>} {tab === 'single' && <SingleImage models={models}/>} {tab === 'results' && <Results comparisons={comparisons} selectedId={selectedRun || comparisons[0]?.comparison_id} selectComparison={setSelectedRun}/>}</main>
    <footer>Research application · Validation is the default · Locked test requires explicit confirmation</footer></>
}
