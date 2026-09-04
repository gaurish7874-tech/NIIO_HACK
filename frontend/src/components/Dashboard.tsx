import { Activity, Eye, FileText, AlertCircle } from 'lucide-react';

export default function Dashboard({ result, isAnalyzing }: { result: any, isAnalyzing: boolean }) {
  if (isAnalyzing && !result) {
    return (
      <div className="surface" style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)' }}>
        <div style={{ textAlign: 'center' }}>
          <Activity size={32} style={{ margin: '0 auto 1rem', opacity: 0.5 }} />
          <p style={{ fontSize: '0.95rem' }}>Awaiting data...</p>
        </div>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="surface" style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)' }}>
        <div style={{ textAlign: 'center' }}>
          <Activity size={32} style={{ margin: '0 auto 1rem', opacity: 0.3 }} />
          <p style={{ fontSize: '0.95rem' }}>Provide a sample to view health metrics.</p>
        </div>
      </div>
    );
  }

  const { vitals, triage } = result;
  const physio = vitals?.physio || {};
  const behavioral = vitals?.behavioral || {};
  const score = triage?.wellness_score?.score || '--';
  const category = triage?.wellness_score?.category || 'Unknown';

  // Determine colors based on score
  let scoreColor = 'var(--accent)';
  if (score < 50) {
    scoreColor = 'var(--danger)';
  } else if (score < 75) {
    scoreColor = 'var(--warning)';
  }

  return (
    <div className="animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem', height: '100%' }}>
      
      {/* Top Banner: Score */}
      <div className="surface" style={{ padding: '1.5rem', display: 'flex', alignItems: 'center', justifyContent: 'space-between', borderLeft: `4px solid ${scoreColor}` }}>
        <div>
          <h2 style={{ fontSize: '1rem', color: 'var(--text-muted)', fontWeight: 500, marginBottom: '0.25rem' }}>Overall Wellness</h2>
          <div style={{ fontSize: '1.1rem', fontWeight: 600 }}>{category}</div>
        </div>
        <div style={{ 
          fontSize: '2.5rem', 
          fontWeight: 700, 
          color: scoreColor,
          lineHeight: 1
        }}>
          {score}
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
        {/* Physio Card */}
        <div className="surface" style={{ padding: '1.25rem' }}>
          <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.9rem', color: 'var(--text-main)', marginBottom: '1rem', fontWeight: 600 }}>
            <Activity size={16} color="var(--text-muted)" /> Physiology
          </h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', fontSize: '0.95rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: 'var(--text-muted)' }}>Heart Rate</span>
              <span style={{ fontWeight: 500 }}>{physio.hr?.toFixed(1) || '--'} bpm</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: 'var(--text-muted)' }}>Blood Pressure</span>
              <span style={{ fontWeight: 500 }}>
                {physio.bp_sys ? `${Math.round(physio.bp_sys)}/${Math.round(physio.bp_dia)}` : '--'} mmHg
              </span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: 'var(--text-muted)' }}>Respiration</span>
              <span style={{ fontWeight: 500 }}>{physio.respiration?.toFixed(1) || '--'} rpm</span>
            </div>
          </div>
        </div>

        {/* Behavioral Card */}
        <div className="surface" style={{ padding: '1.25rem' }}>
          <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.9rem', color: 'var(--text-main)', marginBottom: '1rem', fontWeight: 600 }}>
            <Eye size={16} color="var(--text-muted)" /> Behavior
          </h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', fontSize: '0.95rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: 'var(--text-muted)' }}>Blinks</span>
              <span style={{ fontWeight: 500 }}>{behavioral.blink_rate?.toFixed(1) || '--'} /min</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: 'var(--text-muted)' }}>Eye Closure</span>
              <span style={{ fontWeight: 500 }}>{behavioral.eye_closure ? (behavioral.eye_closure * 100).toFixed(1) : '--'}%</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: 'var(--text-muted)' }}>Gaze</span>
              <span style={{ fontWeight: 500 }}>{behavioral.gaze_stability?.toFixed(2) || '--'}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Summary / Emotion */}
      <div className="surface" style={{ padding: '1.5rem', flex: 1 }}>
        <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.9rem', marginBottom: '1rem', fontWeight: 600 }}>
          <FileText size={16} color="var(--text-muted)" /> Clinical Summary
        </h3>
        <p style={{ fontSize: '0.95rem', lineHeight: 1.6, color: 'var(--text-muted)' }}>
          {triage?.summary || "No summary available."}
        </p>
        
        {triage?.emotional_map && (
          <div style={{ marginTop: '1.5rem', padding: '1rem', background: 'var(--surface-hover)', borderRadius: 'var(--radius-sm)' }}>
            <div className="text-label" style={{ marginBottom: '0.25rem' }}>Mental State</div>
            <div style={{ fontWeight: 500, fontSize: '0.95rem' }}>{triage.emotional_map.quadrant_label}</div>
            <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>{triage.emotional_map.description}</div>
          </div>
        )}

        {triage?.contradiction?.has_contradiction && (
          <div style={{ marginTop: '1rem', padding: '1rem', background: 'var(--warning-light)', borderLeft: '3px solid var(--warning)', borderRadius: 'var(--radius-sm)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#b45309', fontWeight: 600, fontSize: '0.9rem', marginBottom: '0.25rem' }}>
              <AlertCircle size={16} /> Signal Mismatch
            </div>
            <div style={{ fontSize: '0.85rem', color: '#92400e' }}>{triage.contradiction.explanation}</div>
          </div>
        )}
      </div>
    </div>
  );
}
