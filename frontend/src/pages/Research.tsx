import { useEffect, useState } from 'react'
import { GroupedBars, Legend } from '../components/Charts'
import { ErrorLine, JobProgress, fmtTime, modelColor, useJob } from '../components/common'
import { api, MODEL_LABEL, MODEL_NAMES, type BenchRun } from '../lib/api'

const METRICS: Record<string, { label: string; datasets: string[] }> = {
  image_auroc: { label: 'Image AUROC', datasets: ['mvtec_ad2'] },
  pixel_auroc: { label: 'Pixel AUROC', datasets: ['mvtec_ad2'] },
  au_pro_30: { label: 'AU-PRO @30% FPR', datasets: ['mvtec_ad2'] },
  frame_auroc_micro: { label: 'Frame AUROC (micro)', datasets: ['msad'] },
  frame_auroc_macro: { label: 'Frame AUROC (macro)', datasets: ['msad'] },
  f1_at_threshold: { label: 'F1 @ calibrated threshold', datasets: ['mvtec_ad2', 'msad'] },
  best_f1: { label: 'Best F1', datasets: ['mvtec_ad2', 'msad'] },
}
const COLS = ['image_auroc', 'pixel_auroc', 'au_pro_30', 'frame_auroc_micro', 'frame_auroc_macro', 'f1_at_threshold', 'best_f1', 'latency_ms', 'train_s']

export default function Research() {
  const [runs, setRuns] = useState<BenchRun[]>([])
  const [dataset, setDataset] = useState<'mvtec_ad2' | 'msad'>('mvtec_ad2')
  const [metric, setMetric] = useState('image_auroc')
  const [root, setRoot] = useState('/datasets/mvtec_ad2')
  const [category, setCategory] = useState('all')
  const [labels, setLabels] = useState('')
  const [epochs, setEpochs] = useState('')
  const [jobId, setJobId] = useState<number | null>(null)
  const [error, setError] = useState<string | null>(null)
  const load = () => api.benchmarks().then(setRuns).catch((e) => setError(e.message))
  const job = useJob(jobId, load)
  useEffect(() => {
    load()
  }, [])

  // latest run per (dataset, category)
  const latest = new Map<string, BenchRun>()
  for (const r of runs) if (!latest.has(`${r.dataset}/${r.category}`)) latest.set(`${r.dataset}/${r.category}`, r)
  const shown = [...latest.values()].filter((r) => r.dataset === dataset).sort((a, b) => a.category.localeCompare(b.category))
  const metricsHere = Object.entries(METRICS).filter(([, m]) => m.datasets.includes(dataset))
  const models = MODEL_NAMES.filter((m) => shown.some((r) => r.results[m]))

  async function start() {
    setError(null)
    try {
      const body: Record<string, unknown> = { dataset, root, models: MODEL_NAMES, epochs: epochs ? Number(epochs) : null }
      if (dataset === 'mvtec_ad2') body.category = category
      else if (labels) body.labels = labels
      setJobId((await api.runBenchmark(body)).job_id)
    } catch (e) {
      setError((e as Error).message)
    }
  }

  return (
    <div className="stack">
      <div className="card stack">
        <h3>Model comparison</h3>
        <p className="small secondary" style={{ margin: 0 }}>
          Autoencoder (reconstruction) vs feature-embedding memory bank (PatchCore-style) vs temporal future-frame prediction.
          MVTec AD 2 measures static visual anomalies on still images (the temporal model does not apply); MSAD measures
          frame-level video anomalies across scenarios.
        </p>
        <div className="row" style={{ gap: 16 }}>
          <label className="field">Dataset
            <select value={dataset} onChange={(e) => {
              const d = e.target.value as 'mvtec_ad2' | 'msad'
              setDataset(d)
              setMetric(d === 'msad' ? 'frame_auroc_micro' : 'image_auroc')
              setRoot(d === 'msad' ? '/datasets/msad' : '/datasets/mvtec_ad2')
            }}>
              <option value="mvtec_ad2">MVTec AD 2</option>
              <option value="msad">MSAD</option>
            </select>
          </label>
          <label className="field">Dataset path on the server
            <input value={root} onChange={(e) => setRoot(e.target.value)} style={{ width: 240 }} />
          </label>
          {dataset === 'mvtec_ad2' ? (
            <label className="field">Category
              <input value={category} onChange={(e) => setCategory(e.target.value)} placeholder="all | can | fabric …" style={{ width: 140 }} />
            </label>
          ) : (
            <label className="field">Labels (JSON file or dir)
              <input value={labels} onChange={(e) => setLabels(e.target.value)} placeholder="<root>/test_labels" style={{ width: 220 }} />
            </label>
          )}
          <label className="field">Epochs
            <input type="number" value={epochs} placeholder="default" onChange={(e) => setEpochs(e.target.value)} style={{ width: 90 }} />
          </label>
          <button className="btn primary" disabled={job?.status === 'running'} onClick={start}>Run benchmark</button>
        </div>
        <JobProgress job={job} />
        <ErrorLine error={error} />
        <div className="small muted">Long runs are easier from the CLI: <code>python scripts/benchmark.py {dataset} --root … --save-db</code></div>
      </div>

      {shown.length ? (
        <>
          <div className="card">
            <div className="between" style={{ marginBottom: 8 }}>
              <select value={metric} onChange={(e) => setMetric(e.target.value)}>
                {metricsHere.map(([k, m]) => <option key={k} value={k}>{m.label}</option>)}
              </select>
              <Legend items={models.map((m) => ({ label: MODEL_LABEL[m], color: modelColor(m) }))} />
            </div>
            <GroupedBars
              categories={shown.map((r) => r.category)}
              series={models.map((m) => ({ key: m, label: MODEL_LABEL[m], color: modelColor(m), values: shown.map((r) => r.results[m]?.[metric] ?? null) }))}
            />
          </div>
          <div className="card table-wrap">
            <table className="data">
              <thead>
                <tr><th>Category</th><th>Detector</th>{COLS.map((c) => <th key={c} className="num">{METRICS[c]?.label ?? c}</th>)}<th>Run</th></tr>
              </thead>
              <tbody>
                {shown.flatMap((r) =>
                  models.filter((m) => r.results[m]).map((m) => (
                    <tr key={`${r.id}-${m}`}>
                      <td>{r.category}</td>
                      <td><span className="row" style={{ gap: 6 }}><span className="model-dot" style={{ background: modelColor(m) }} />{MODEL_LABEL[m]}</span></td>
                      {COLS.map((c) => {
                        const v = r.results[m][c]
                        return <td key={c} className="num mono">{v === null || v === undefined ? '–' : typeof v === 'number' ? (c.endsWith('_s') || c.endsWith('_ms') ? v.toFixed(1) : v.toFixed(3)) : String(v)}</td>
                      })}
                      <td className="small muted">{fmtTime(r.created_at)}</td>
                    </tr>
                  )),
                )}
              </tbody>
            </table>
          </div>
        </>
      ) : (
        <div className="card empty">No {dataset === 'msad' ? 'MSAD' : 'MVTec AD 2'} benchmark runs yet.</div>
      )}
    </div>
  )
}
