// Thin API client. Study identifiers are random hex ids (no patient identifiers in URLs).
async function j(res) {
  if (!res.ok) {
    let msg = res.statusText
    try { const b = await res.json(); msg = typeof b.detail === 'string' ? b.detail : JSON.stringify(b.detail) } catch {}
    throw new Error(`${res.status}: ${msg}`)
  }
  return res.json()
}
const post = (url, body) => fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }).then(j)
export const api = {
  health: () => fetch('/health').then(j),
  studies: () => fetch('/studies').then(j),
  study: id => fetch(`/studies/${id}`).then(j),
  upload: files => { const f = new FormData(); [...files].forEach(x => f.append('files', x)); return fetch('/studies/upload', { method: 'POST', body: f }).then(j) },
  demo: () => fetch('/demo/synthetic?n_patients=2', { method: 'POST' }).then(j),
  anomaly: id => post('/anomaly-detection', { study_id: id }),
  ask: (id, question, earlier) => post('/vqa/ask', { study_id: id, question, earlier_study_id: earlier || null }),
  compare: (earlier, later, question) => post('/temporal-vqa/compare', { earlier_study_id: earlier, later_study_id: later, question }),
  sliceUrl: (id, z, layer, wc, ww) => `/studies/${id}/slices/${z}?layer=${layer}&wc=${wc}&ww=${ww}`,
  resultSliceUrl: (rid, z, layer, wc, ww) => `/results/${rid}/slices/${z}?layer=${layer}&wc=${wc}&ww=${ww}`,
  reportUrl: rid => `/results/${rid}/report`,
}
