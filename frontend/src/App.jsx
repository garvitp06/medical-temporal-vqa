import React, { useEffect, useMemo, useRef, useState } from 'react'
import { api } from './api.js'

const WINDOWS = { Lung: [-600, 1500], Mediastinum: [40, 400], Bone: [400, 1800] }
const SUGGESTED = {
  single: ['Is there a suspicious region in this scan?', 'Where is the suspicious region?',
    'Which lung contains the suspicious finding?', 'Which zone of the lung contains the largest suspicious region?',
    'What abnormality is visible in this region?'],
  temporal: ['Has the suspected lesion increased or decreased in size since the previous scan?',
    'Is there a new suspicious region compared with the previous examination?',
    'What is the volume change of the lesion?'],
}
const fmt = (v, d = 0, u = '') => (v === null || v === undefined || Number.isNaN(v)) ? '—' : `${Number(v).toLocaleString(undefined, { maximumFractionDigits: d, minimumFractionDigits: d })}${u}`
const fmtDate = d => d && d.length === 8 ? `${d.slice(0, 4)}-${d.slice(4, 6)}-${d.slice(6)}` : (d || '—')

function useStudies() {
  const [studies, setStudies] = useState([])
  const refresh = () => api.studies().then(setStudies).catch(() => {})
  useEffect(() => { refresh(); const t = setInterval(refresh, 3000); return () => clearInterval(t) }, [])
  return [studies, refresh]
}

function Badge({ kind, children }) { return <span className={`badge ${kind || ''}`}>{children}</span> }

function Sidebar({ studies, selected, earlier, onSelect, onEarlier, refresh, health }) {
  const fileRef = useRef()
  const [busy, setBusy] = useState(null)
  const later = studies.find(s => s.study_id === selected)
  const run = async (label, fn) => { setBusy(label); try { await fn() } catch (e) { alert(e.message) } finally { setBusy(null); refresh() } }
  return (
    <aside className="sidebar">
      <div className="brand">
        <div className="brand-title">Temporal CT VQA</div>
        <div className="brand-sub">Research prototype · not for clinical use</div>
      </div>
      <div className="actions">
        <button onClick={() => fileRef.current.click()} disabled={!!busy}>Upload DICOM series</button>
        <input ref={fileRef} type="file" multiple hidden onChange={e => run('upload', () => api.upload(e.target.files))} />
        <button className="ghost" onClick={() => run('demo', api.demo)} disabled={!!busy}>Load synthetic demo</button>
        {busy && <div className="muted small">{busy === 'upload' ? 'Uploading…' : 'Importing synthetic studies…'}</div>}
      </div>
      <div className="section-label">Studies</div>
      <div className="study-list">
        {studies.length === 0 && <div className="muted small pad">No studies yet.</div>}
        {studies.map(s => (
          <div key={s.study_id} className={`study ${s.study_id === selected ? 'active' : ''} ${s.study_id === earlier ? 'prior' : ''}`}
            onClick={() => s.status === 'ready' && onSelect(s.study_id)}>
            <div className="study-row">
              <span className="mono">{fmtDate(s.study_date)}</span>
              <Badge kind={s.status}>{s.status}</Badge>
            </div>
            <div className="study-row small muted">
              <span>pt {s.patient_key || '…'}</span>
              {s.synthetic && <Badge kind="synthetic">SYNTHETIC</Badge>}
            </div>
            {s.error && <div className="small error-text">{s.error}</div>}
            {later && s.study_id !== selected && s.status === 'ready' && s.patient_key === later.patient_key && (
              <button className="link small" onClick={e => { e.stopPropagation(); onEarlier(s.study_id === earlier ? null : s.study_id) }}>
                {s.study_id === earlier ? 'Remove as prior' : 'Use as prior exam'}
              </button>)}
          </div>
        ))}
      </div>
      {health && <div className="health small muted">
        <div>Detector: <b>{health.detector}</b> · thr {fmt(health.threshold, 1)}</div>
        <div>VQA: {health.vqa_loaded ? 'loaded' : 'not loaded'} · {health.device}</div>
        {health.warnings?.map((w, i) => <div key={i} className="warn-text">{w}</div>)}
      </div>}
    </aside>
  )
}

function Viewer({ title, nz, dims, z, setZ, ctUrl, heatUrl, showHeat, boxes, footer }) {
  const onWheel = e => { e.preventDefault(); setZ(Math.max(0, Math.min(nz - 1, z + (e.deltaY > 0 ? 1 : -1)))) }
  const ref = useRef()
  useEffect(() => {
    const el = ref.current; if (!el) return
    el.addEventListener('wheel', onWheel, { passive: false }); return () => el.removeEventListener('wheel', onWheel)
  })
  const [H, W] = dims
  return (
    <div className="viewer">
      <div className="viewer-head"><span>{title}</span><span className="mono small">slice {z + 1}/{nz}</span></div>
      <div className="canvas" ref={ref} style={{ aspectRatio: `${W} / ${H}` }}>
        <img src={ctUrl} alt="CT slice" draggable={false} />
        {showHeat && <img src={heatUrl} alt="" className="heat" draggable={false} />}
        <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none">
          {boxes.map(b => (
            <g key={b.id}>
              <rect x={b.x0} y={b.y0} width={b.x1 - b.x0} height={b.y1 - b.y0} className={`box ${b.cls || ''}`} />
              <text x={b.x0} y={Math.max(4, b.y0 - 1.5)} className="box-label">{b.id}</text>
            </g>))}
        </svg>
        <div className="orient l">R</div><div className="orient r">L</div>
      </div>
      <input type="range" min={0} max={nz - 1} value={z} onChange={e => setZ(+e.target.value)} />
      {footer}
    </div>
  )
}

const boxesFor = (cands, z, cls) => cands.filter(c => c.bbox_voxel[0] <= z && z < c.bbox_voxel[3]).map(c => ({
  id: c.candidate_id, x0: c.bbox_voxel[2], y0: c.bbox_voxel[1], x1: c.bbox_voxel[5], y1: c.bbox_voxel[4], cls }))

function CandidateTable({ cands, onJump, title = 'Candidate regions' }) {
  return (
    <div className="card">
      <div className="card-title">{title} <span className="muted small">(suspicious regions for review — not diagnoses)</span></div>
      {cands.length === 0 ? <div className="muted small">No region exceeded the validation-selected threshold.</div> :
        <table>
          <thead><tr><th>ID</th><th>Side / zone</th><th>Mask vol (mm³)</th><th>BBox est. (mm³)</th><th>Longest ⌀ (mm)</th><th>Max score</th><th>Slice</th></tr></thead>
          <tbody>{cands.map(c => (
            <tr key={c.candidate_id} onClick={() => onJump(c.key_slice)} className="clickable">
              <td className="mono">{c.candidate_id}</td><td>{c.laterality} / {c.zone}</td><td className="num">{fmt(c.volume_mm3)}</td>
              <td className="num muted">{fmt(c.bbox_volume_mm3)}</td><td className="num">{fmt(c.longest_diameter_mm, 1)}</td>
              <td className="num">{fmt(c.max_score, 1)}</td><td className="num">{c.key_slice + 1}</td></tr>))}
          </tbody></table>}
    </div>
  )
}

function Measurements({ m }) {
  if (!m) return null
  const rows = [['Candidate-mask volume', m.volume_mm3, 0, ' mm³'], ['Bounding-box estimate', m.bbox_volume_mm3, 0, ' mm³'],
    ['Longest diameter', m.longest_diameter_mm, 1, ' mm']]
  return (
    <table className="meas">
      <thead><tr><th></th><th>Earlier</th><th>Later</th><th>Δ</th><th>%Δ</th></tr></thead>
      <tbody>
        {rows.map(([n, r, d, u]) => <tr key={n}><td>{n}</td><td className="num">{fmt(r.earlier, d, u)}</td><td className="num">{fmt(r.later, d, u)}</td>
          <td className="num">{fmt(r.absolute, d, u)}</td><td className="num">{r.percent === null ? <span className="muted small">{r.case === 'new_candidate_percent_undefined' ? 'undefined (new)' : '—'}</span> : fmt(r.percent, 1, '%')}</td></tr>)}
        <tr><td>Max anomaly score</td><td className="num">{fmt(m.anomaly_score_max.earlier, 1)}</td><td className="num">{fmt(m.anomaly_score_max.later, 1)}</td><td className="num">{fmt(m.anomaly_score_max.absolute, 1)}</td><td className="muted small">not growth</td></tr>
        <tr><td>Centroid displacement</td><td colSpan={4} className="num">{fmt(m.centroid_displacement_mm, 1, ' mm')}</td></tr>
      </tbody>
    </table>
  )
}

function Answer({ a, onJump }) {
  if (!a) return null
  const ans = a.answer ?? a.status
  return (
    <div className="card answer">
      <div className="answer-head">
        <span className="answer-main">{String(ans)}</span>
        <Badge>{a.task}</Badge>
        {a.routing && <span className="muted small">routed {a.routing.method}{a.routing.confidence ? ` (${(a.routing.confidence * 100).toFixed(0)}%)` : ''}</span>}
        {a.model_confidence != null && <span className="muted small">· model conf {(a.model_confidence * 100).toFixed(0)}%</span>}
      </div>
      <p className="answer-text">{a.answer_text}</p>
      {a.registration_quality && <div className={`reg ${a.registration_quality.reliable ? 'ok' : 'bad'}`}>
        Registration: <b>{a.registration_quality.quality}</b> · lung Dice {fmt(a.registration_quality.lung_dice_before, 3)} → {fmt(a.registration_quality.lung_dice_after, 3)} · NCC {fmt(a.registration_quality.ncc_lung_after, 2)}
      </div>}
      {a.measurements && <Measurements m={a.measurements} />}
      {a.evidence?.length > 0 && <div className="small">Evidence: {a.evidence.map(e => (
        <button key={e.candidate_id} className="link" onClick={() => onJump(e.key_slice)}>{e.candidate_id} · slice {e.key_slice + 1}</button>))}</div>}
      {a.uncertainty?.length > 0 && <ul className="unc">{a.uncertainty.map((u, i) => <li key={i}>{u}</li>)}</ul>}
      {a.result_id && <a className="link small" href={api.reportUrl(a.result_id)} target="_blank" rel="noreferrer">Export structured research report (JSON)</a>}
    </div>
  )
}

export default function App() {
  const [studies, refresh] = useStudies()
  const [health, setHealth] = useState(null)
  const [selected, setSelected] = useState(null)
  const [earlier, setEarlier] = useState(null)
  const [study, setStudy] = useState(null)
  const [z, setZ] = useState(0)
  const [win, setWin] = useState(WINDOWS.Lung)
  const [heat, setHeat] = useState(true)
  const [question, setQuestion] = useState(SUGGESTED.single[0])
  const [answer, setAnswer] = useState(null)
  const [temporal, setTemporal] = useState(null)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState(null)

  useEffect(() => { api.health().then(setHealth).catch(e => setErr(e.message)) }, [])
  useEffect(() => {
    if (!selected) return
    setAnswer(null); setTemporal(null)
    api.study(selected).then(s => {
      setStudy(s)
      const c = s.summary?.candidates?.[0]
      setZ(c ? c.key_slice : Math.floor((s.summary?.shape_zyx?.[0] || 1) / 2))
    })
  }, [selected])
  useEffect(() => { setTemporal(null); setAnswer(null) }, [earlier])

  const sm = study?.summary
  const nz = sm?.shape_zyx?.[0] || 1
  const dims = sm ? [sm.shape_zyx[1], sm.shape_zyx[2]] : [1, 1]
  const cands = sm?.candidates || []
  const eCands = temporal?.entities ? temporal.entities.filter(e => e.earlier).map(e => e.earlier) : []

  const ask = async () => {
    setBusy(true); setErr(null)
    try {
      if (earlier) {
        const r = await api.compare(earlier, selected, question)
        setTemporal(r); setAnswer({ ...r.answer, result_id: r.result_id })
      } else setAnswer(await api.ask(selected, question))
    } catch (e) { setErr(e.message) } finally { setBusy(false) }
  }

  return (
    <div className="app">
      <Sidebar studies={studies} selected={selected} earlier={earlier} onSelect={id => { setSelected(id); setEarlier(null) }}
        onEarlier={setEarlier} refresh={refresh} health={health} />
      <main>
        <div className="banner">Research prototype. Model outputs are suspicious regions that require professional review; they are not diagnoses. {study?.synthetic && <b>This study is SYNTHETIC demo data.</b>}</div>
        {!study ? <div className="empty">Select a processed study, upload a DICOM series, or load the synthetic demo.</div> : <>
          <div className="toolbar">
            <div className="seg">{Object.entries(WINDOWS).map(([k, v]) => <button key={k} className={win === v ? 'on' : ''} onClick={() => setWin(v)}>{k}</button>)}</div>
            <label className="small">WC <input type="number" value={win[0]} onChange={e => setWin([+e.target.value, win[1]])} /></label>
            <label className="small">WW <input type="number" value={win[1]} onChange={e => setWin([win[0], +e.target.value])} /></label>
            <label className="small check"><input type="checkbox" checked={heat} onChange={e => setHeat(e.target.checked)} /> anomaly heatmap</label>
            <span className="muted small">Study {fmtDate(study.study_date)} · {sm.spacing_xyz.map(v => v.toFixed(1)).join('×')} mm · {sm.detector} · vol score {fmt(sm.volume_score, 1)}</span>
          </div>
          {sm.warnings?.length > 0 && <div className="warnbox">{sm.warnings.map((w, i) => <div key={i}>⚠ {w}</div>)}</div>}
          <div className={`viewers ${temporal ? 'two' : ''}`}>
            {temporal && <Viewer title={`Earlier (registered to later) · ${fmtDate(studies.find(s => s.study_id === earlier)?.study_date)}`}
              nz={nz} dims={dims} z={z} setZ={setZ} showHeat={heat}
              ctUrl={api.resultSliceUrl(temporal.result_id, z, 'ct', win[0], win[1])}
              heatUrl={api.resultSliceUrl(temporal.result_id, z, 'anomaly', win[0], win[1])}
              boxes={boxesFor(eCands, z, 'earlier')} />}
            <Viewer title={`${temporal ? 'Later' : 'Current'} · ${fmtDate(study.study_date)}`} nz={nz} dims={dims} z={z} setZ={setZ} showHeat={heat}
              ctUrl={api.sliceUrl(selected, z, 'ct', win[0], win[1])} heatUrl={api.sliceUrl(selected, z, 'anomaly', win[0], win[1])}
              boxes={boxesFor(cands, z)} />
          </div>
          <div className="lower">
            <div className="col">
              <div className="card">
                <div className="card-title">Ask a question {earlier ? <Badge kind="prior">temporal · prior selected</Badge> : <span className="muted small">select a prior exam of the same patient for temporal questions</span>}</div>
                <div className="ask">
                  <input value={question} onChange={e => setQuestion(e.target.value)} onKeyDown={e => e.key === 'Enter' && ask()} />
                  <button onClick={ask} disabled={busy}>{busy ? 'Analysing…' : 'Ask'}</button>
                </div>
                <div className="chips">{(earlier ? SUGGESTED.temporal : []).concat(SUGGESTED.single).map(q => <button key={q} className="chip" onClick={() => setQuestion(q)}>{q}</button>)}</div>
                {err && <div className="error-text small">{err}</div>}
              </div>
              <Answer a={answer} onJump={setZ} />
            </div>
            <div className="col">
              <CandidateTable cands={cands} onJump={setZ} />
              {temporal && <div className="card">
                <div className="card-title">Cross-time association</div>
                <div className="small muted">matched {temporal.counts.n_matched} · potentially new {temporal.counts.n_new} · potentially resolved {temporal.counts.n_resolved} · uncertain {temporal.counts.n_uncertain}</div>
                <table><thead><tr><th>Status</th><th>Earlier</th><th>Later</th><th>Vol earlier</th><th>Vol later</th><th>%Δ</th></tr></thead>
                  <tbody>{temporal.entities.map((e, i) => <tr key={i} className="clickable" onClick={() => setZ((e.later || e.earlier).key_slice)}>
                    <td>{e.match_status}</td><td className="mono">{e.earlier?.candidate_id || '—'}</td><td className="mono">{e.later?.candidate_id || '—'}</td>
                    <td className="num">{fmt(e.measurements.volume_mm3.earlier)}</td><td className="num">{fmt(e.measurements.volume_mm3.later)}</td>
                    <td className="num">{e.measurements.volume_mm3.percent == null ? '—' : fmt(e.measurements.volume_mm3.percent, 1, '%')}</td></tr>)}</tbody></table>
              </div>}
              <div className="card small muted">
                <div className="card-title">Preprocessing &amp; QC</div>
                <div>Lung volume {fmt(sm.qc.lung_volume_ml)} ml · HU range {sm.qc.hu_range?.map(v => fmt(v)).join(' … ')} · QC {sm.qc.ok ? 'passed' : 'flagged'}</div>
                <div>{sm.preprocessing_ops.map(o => o.op + (o.method ? ` (${o.method})` : '')).join(' → ')}</div>
              </div>
            </div>
          </div>
        </>}
      </main>
    </div>
  )
}
