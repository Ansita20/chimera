import { useEffect, useState } from 'react'
import { ErrorLine, ExplanationCard, ModelScores, RegionBoxes, RegionLabels, Tile, VerdictBadge, modelColor } from '../components/common'
import { api, MODEL_LABEL, type FrameResult, type Scene } from '../lib/api'

export default function Inspect({ scene }: { scene: Scene | null }) {
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [res, setRes] = useState<FrameResult | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [threshold, setThreshold] = useState(1)
  const [fusion, setFusion] = useState('max')
  const [showHeat, setShowHeat] = useState(true)

  useEffect(() => {
    if (!file) return
    const u = URL.createObjectURL(file)
    setPreview(u)
    return () => URL.revokeObjectURL(u)
  }, [file])

  async function run(f = file) {
    if (!scene || !f) return
    setBusy(true)
    setError(null)
    try {
      setRes(await api.detect(scene.id, f, { threshold, fusion }))
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const appearance = scene ? (['embedding', 'autoencoder'] as const).filter((m) => scene.models[m]?.trained) : []

  return (
    <div className="stack">
      <div className="card row" style={{ gap: 16 }}>
        <label className="btn">
          Choose image…
          <input type="file" accept="image/*" hidden onChange={(e) => {
            const f = e.target.files?.[0] ?? null
            setFile(f)
            setRes(null)
            if (f) run(f)
          }} />
        </label>
        <label className="field">
          Threshold · {threshold.toFixed(2)}×
          <input type="range" min={0.5} max={3} step={0.05} value={threshold} onChange={(e) => setThreshold(Number(e.target.value))} />
        </label>
        <label className="field">
          Fusion
          <select value={fusion} onChange={(e) => setFusion(e.target.value)}>
            <option value="max">any detector (max)</option>
            <option value="mean">consensus (mean)</option>
          </select>
        </label>
        <button className="btn primary" disabled={!file || busy || !appearance.length} onClick={() => run()}>
          {busy ? 'Analysing…' : 'Analyse'}
        </button>
        <label className="row small" style={{ gap: 6 }}>
          <input type="checkbox" checked={showHeat} onChange={(e) => setShowHeat(e.target.checked)} /> Heatmap overlay
        </label>
        {scene && !appearance.length && <span className="muted">Train the embedding or autoencoder detector for this scene first.</span>}
      </div>
      <ErrorLine error={error} />

      {preview && (
        <div className="live-grid">
          <div className="stack">
            <div className="viewer">
              <img className="base" src={preview} alt="input" />
              {res && showHeat && <img className="heat" src={res.heatmap} alt="" />}
              {res && <RegionBoxes regions={res.regions} />}
              {res && <RegionLabels regions={res.regions} />}
            </div>
            {res?.model_heatmaps && (
              <div className="card">
                <h3>Per-detector heatmaps</h3>
                <div className="thumbs">
                  {Object.entries(res.model_heatmaps).map(([m, h]) => (
                    <figure key={m} style={{ margin: 0 }}>
                      <div className="viewer">
                        <img className="base" src={preview} alt="" />
                        <img className="heat" src={h} alt="" />
                      </div>
                      <figcaption className="row small" style={{ gap: 6, marginTop: 6 }}>
                        <span className="model-dot" style={{ background: modelColor(m) }} /> {MODEL_LABEL[m]} · {(res.models[m].score / threshold).toFixed(2)}×
                      </figcaption>
                    </figure>
                  ))}
                  {res.reconstruction && (
                    <figure style={{ margin: 0 }}>
                      <div className="viewer"><img className="base" src={res.reconstruction} alt="reconstruction" /></div>
                      <figcaption className="small muted" style={{ marginTop: 6 }}>Autoencoder reconstruction — what “normal” would look like</figcaption>
                    </figure>
                  )}
                </div>
              </div>
            )}
          </div>
          <div className="stack">
            {res ? (
              <>
                <div className="card stack">
                  <VerdictBadge verdict={res.verdict} big />
                  <div className="tiles">
                    <Tile label="Score" value={`${res.score.toFixed(2)}×`} />
                    <Tile label="Uncertainty" value={`${Math.round(res.uncertainty * 100)}%`} />
                    <Tile label="Regions" value={res.regions.length} />
                  </div>
                  <ModelScores models={res.models} threshold={threshold} />
                  <div className="small muted">
                    Uncertainty combines model uncertainty (Monte-Carlo dropout for the autoencoder, memory-bank disagreement
                    for the embedding detector) with how close the score sits to the threshold.
                  </div>
                </div>
                {res.regions.map((r, i) => (
                  <ExplanationCard key={i} e={r.explanation} title={`Region ${i + 1} · ${r.peak.toFixed(1)}× · ${(r.area_frac * 100).toFixed(1)}% of image`} />
                ))}
              </>
            ) : (
              <div className="card empty">{busy ? 'Analysing…' : 'Press Analyse.'}</div>
            )}
          </div>
        </div>
      )}
      {!preview && <div className="card empty">Upload a still image to localise anomalies and see why each region was flagged.</div>}
    </div>
  )
}
