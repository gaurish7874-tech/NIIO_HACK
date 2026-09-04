import { useEffect, useMemo, useState } from 'react';
import {
  Activity,
  ArrowRight,
  Check,
  Cpu,
  Eye,
  HeartPulse,
  Lock,
  Radio,
  ShieldCheck,
  SlidersHorizontal,
  User,
  X,
} from 'lucide-react';
import CameraRecord from './components/CameraRecord';
import Chat from './components/Chat';
import Dashboard from './components/Dashboard';
import './App.css';

const navItems = ['Technology', 'Methodology', 'Clinical Analysis', 'Data Security'];

const features = [
  {
    icon: Eye,
    tone: 'purple',
    title: 'Optical rPPG',
    description: 'Captures hemoglobin pulse changes across 68 facial mesh zones simultaneously.',
    link: 'Sub-pixel analysis',
  },
  {
    icon: SlidersHorizontal,
    tone: 'sky',
    title: 'Signal Pipeline',
    description: 'Isolates micro-motion and environmental flicker to maintain clinical accuracy.',
    link: 'rPPG Signal Filter',
  },
  {
    icon: HeartPulse,
    tone: 'blue',
    title: 'Dual Triage',
    description: 'Synthesizes pressure, stress, and respiration into actionable urgency scores.',
    link: 'Instant clinical report',
  },
];

const steps = [
  {
    number: '01',
    title: 'Capture',
    copy: 'Standard RGB camera stream acquires high-framerate ROI data.',
  },
  {
    number: '02',
    title: 'Isolate',
    copy: 'Component analysis filters out noise and extracts pure wave frequencies.',
  },
  {
    number: '03',
    title: 'Triage',
    copy: 'Instant reporting of validated wellness scores for practitioners.',
  },
];

function buildWavePath(phase: number) {
  const width = 520;
  const height = 150;
  const points = Array.from({ length: 72 }, (_, index) => {
    const x = (index / 71) * width;
    const pulse = Math.sin(index * 0.42 + phase) * 22;
    const micro = Math.sin(index * 1.25 + phase * 1.7) * 6;
    const peak = index % 18 === 0 ? -28 : 0;
    const y = height / 2 + pulse + micro + peak;
    return `${index === 0 ? 'M' : 'L'} ${x.toFixed(1)} ${y.toFixed(1)}`;
  });

  return points.join(' ');
}

function readNumber(value: unknown) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
}

function formatMetric(value: unknown, decimals = 0) {
  const numeric = readNumber(value);
  return numeric === null ? '--' : numeric.toFixed(decimals);
}

function formatVerdict(value: unknown) {
  if (typeof value !== 'string' || value.length === 0) {
    return 'Awaiting result';
  }

  return value
    .split('_')
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ');
}

function DiagnosticCard({
  result,
  isAnalyzing,
  sessionId,
}: {
  result: any;
  isAnalyzing: boolean;
  sessionId: string | null;
}) {
  const [phase, setPhase] = useState(0);
  const vitals = result?.vitals;
  const triage = result?.triage;
  const physio = vitals?.physio ?? {};
  const wellnessScore = readNumber(triage?.wellness_score?.score);
  const hrv = physio.hrv ?? physio.hrv_ms ?? physio.rmssd_ms;
  const hasResult = Boolean(result);
  const statusText = isAnalyzing ? 'Processing' : hasResult ? 'Results Ready' : 'Awaiting Assessment';
  const scanTitle = isAnalyzing
    ? 'Camera Judgement Running'
    : hasResult
      ? 'Latest Assessment Synced'
      : 'No Assessment Captured';
  const scanSubtitle = hasResult
    ? formatVerdict(triage?.verdict)
    : 'Launch camera capture to populate live values';
  const balanceLabel = triage?.emotional_map?.quadrant_label ?? triage?.wellness_score?.category ?? 'Pending';
  const metrics = [
    { label: 'Heart Rate', value: formatMetric(physio.hr, 1), unit: physio.hr == null ? '' : 'bpm' },
    { label: 'Respiration', value: formatMetric(physio.respiration, 1), unit: physio.respiration == null ? '' : 'rpm' },
    { label: 'HRV', value: formatMetric(hrv, 1), unit: hrv == null ? '' : 'ms' },
    {
      label: 'Wellness Score',
      value: wellnessScore === null ? '--' : String(Math.round(wellnessScore)),
      unit: wellnessScore === null ? '' : '/ 100',
    },
  ];

  useEffect(() => {
    let frameId = 0;
    const tick = () => {
      setPhase((current) => current + 0.08);
      frameId = requestAnimationFrame(tick);
    };

    frameId = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frameId);
  }, []);

  const wavePath = useMemo(() => buildWavePath(phase), [phase]);

  return (
    <aside className="diagnostic-card" aria-label="Live optical telemetry diagnostic card">
      <div className="diagnostic-topline">
        <span className={`status-pill ${hasResult ? 'is-ready' : isAnalyzing ? 'is-processing' : 'is-idle'}`}>
          <span aria-hidden="true" />
          {statusText}
        </span>
        <span className="subject-tag">
          {sessionId ? `Session ${sessionId.slice(0, 8)} | Latest Optical Telemetry` : 'No active subject | Capture required'}
        </span>
      </div>

      <div className="metrics-grid">
        {metrics.map((metric) => (
          <div className={metric.value === '--' ? 'metric-tile is-empty' : 'metric-tile'} key={metric.label}>
            <span>{metric.label}</span>
            <strong>
              {metric.value}
              <small>{metric.unit}</small>
            </strong>
          </div>
        ))}
      </div>

      <div className="scan-strip">
        <div className="scan-icon">
          <Radio size={18} />
        </div>
        <div>
          <span>{scanTitle}</span>
          <strong>{scanSubtitle}</strong>
        </div>
      </div>

      <div className="waveform-box">
        <div className="waveform-header">
          <span>RPPG WAVEFORM</span>
          <span>CH-01</span>
        </div>
        <svg viewBox="0 0 520 150" role="img" aria-label="Animated rPPG waveform">
          <defs>
            <linearGradient id="waveGlow" x1="0%" x2="100%" y1="0%" y2="0%">
              <stop offset="0%" stopColor="#38BDF8" />
              <stop offset="54%" stopColor="#2563EB" />
              <stop offset="100%" stopColor="#10B981" />
            </linearGradient>
          </defs>
          <path className="wave-shadow" d={wavePath} />
          <path className="wave-line" d={wavePath} />
        </svg>
        <div className="waveform-grid" aria-hidden="true" />
      </div>

      <div className="balance-row">
        <div>
          <span>Assessment Result</span>
          <strong>{wellnessScore === null ? balanceLabel : `${Math.round(wellnessScore)} / 100`}</strong>
        </div>
        <div className={wellnessScore === null ? 'balance-track is-empty' : 'balance-track'} aria-label="Wellness score progress">
          <span style={{ width: `${wellnessScore ?? 0}%` }} />
        </div>
      </div>

      <div className={hasResult ? 'home-result-summary' : 'home-result-summary is-empty'}>
        <span>{hasResult ? formatVerdict(triage?.verdict) : 'Waiting for camera assessment'}</span>
        <p>
          {hasResult
            ? triage?.summary ?? 'Assessment completed, but no clinical summary was returned.'
            : 'Your homepage vitals and wellness summary will appear here only after the camera analysis finishes.'}
        </p>
      </div>
    </aside>
  );
}

interface AssessmentModalProps {
  onClose: () => void;
  sessionId: string | null;
  setSessionId: (id: string) => void;
  analysisResult: unknown;
  setAnalysisResult: (result: unknown) => void;
  isAnalyzing: boolean;
  setIsAnalyzing: (analyzing: boolean) => void;
}

function AssessmentModal({
  onClose,
  sessionId,
  setSessionId,
  analysisResult,
  setAnalysisResult,
  isAnalyzing,
  setIsAnalyzing,
}: AssessmentModalProps) {
  return (
    <div className="modal-backdrop" role="dialog" aria-modal="true" aria-label="Live assessment">
      <div className="assessment-modal live-assessment-modal">
        <button className="modal-close" type="button" aria-label="Close assessment" onClick={onClose}>
          <X size={18} />
        </button>

        <div className="assessment-header">
          <div className="modal-copy">
            <span className="modal-kicker">Live 10s optical assessment</span>
            <h2>Camera vitals and wellness results</h2>
            <p>
              Real-time contactless capture with triage, signal summary, wellness scoring, and
              assistant context in one clinical workspace.
            </p>
          </div>
          <div className="assessment-session">
            <span className="status-pill">
              <span aria-hidden="true" />
              {isAnalyzing ? 'Processing' : analysisResult ? 'Results Ready' : 'Ready'}
            </span>
            <strong>{sessionId ? `Session ${sessionId.slice(0, 8)}` : 'No active session'}</strong>
          </div>
        </div>

        <div className="assessment-workspace">
          <section className="assessment-camera" aria-label="Camera capture">
            <CameraRecord
              sessionId={sessionId}
              setSessionId={setSessionId}
              setAnalysisResult={setAnalysisResult}
              isAnalyzing={isAnalyzing}
              setIsAnalyzing={setIsAnalyzing}
            />
          </section>
          <section className="assessment-dashboard" aria-label="Wellness results">
            <Dashboard result={analysisResult} isAnalyzing={isAnalyzing} />
          </section>
          <section className="assessment-chat" aria-label="Assessment assistant">
            <Chat sessionId={sessionId} result={analysisResult} setAnalysisResult={setAnalysisResult} />
          </section>
        </div>
      </div>
    </div>
  );
}

function App() {
  const [activeNav, setActiveNav] = useState(navItems[0]);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [analysisResult, setAnalysisResult] = useState<any>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  return (
    <div className="landing-page">
      <header className="nav-shell">
        <a className="brand-mark" href="#top" aria-label="OmniSight home">
          <span className="brand-icon">
            <Activity size={18} />
          </span>
          OmniSight
        </a>

        <nav className="nav-links" aria-label="Main navigation">
          {navItems.map((item) => (
            <button
              className={activeNav === item ? 'nav-link is-active' : 'nav-link'}
              key={item}
              type="button"
              onClick={() => setActiveNav(item)}
            >
              {item}
            </button>
          ))}
        </nav>

        <button className="profile-pill" type="button" aria-label="Open operator profile">
          <User size={18} />
        </button>
      </header>

      <main id="top">
        <section className="hero-section">
          <div className="hero-copy">
            <span className="hero-badge">
              <Cpu size={16} />
              Contactless Triage & Vitals
            </span>
            <h1>
              Multimodal Wellness Screening. <span>Zero Wearables.</span>
            </h1>
            <p>
              Extract real-time physiological markers through remote photoplethysmography,
              transforming standard video feeds into clinical-grade diagnostic streams instantly.
            </p>

            <div className="hero-actions">
              <button className="primary-cta" type="button" onClick={() => setIsModalOpen(true)}>
                Launch 10s Assessment
                <ArrowRight size={18} />
              </button>
              <a className="secondary-cta" href="#features">
                Explore Specs
              </a>
            </div>

            <div className="trust-label">
              <Lock size={14} />
              <span>100% on-device processing | Zero retention | HIPAA-compliant</span>
            </div>
          </div>

          <DiagnosticCard result={analysisResult} isAnalyzing={isAnalyzing} sessionId={sessionId} />
        </section>

        <section className="features-section" id="features">
          <div className="section-heading">
            <span>Computer vision telemetry</span>
            <h2>Precision Optical Tracking</h2>
            <p>Advanced computer vision maps micro-blood flow variations directly from facial optics.</p>
          </div>

          <div className="feature-grid">
            {features.map((feature) => {
              const Icon = feature.icon;
              return (
                <article className="feature-card" key={feature.title}>
                  <div className={`feature-icon ${feature.tone}`}>
                    <Icon size={22} />
                  </div>
                  <h3>{feature.title}</h3>
                  <p>{feature.description}</p>
                  <a href="#process">
                    {feature.link}
                    <ArrowRight size={15} />
                  </a>
                </article>
              );
            })}
          </div>
        </section>

        <section className="process-section" id="process">
          <div className="section-heading compact">
            <span>Live assessment flow</span>
            <h2>Three-Step Sequence</h2>
          </div>

          <div className="process-grid">
            {steps.map((step) => (
              <article className="process-card" key={step.number}>
                <strong>{step.number}</strong>
                <h3>{step.title}</h3>
                <p>{step.copy}</p>
              </article>
            ))}
          </div>
        </section>

        <section className="security-card" id="security">
          <div className="security-copy">
            <span>Security Architecture</span>
            <h2>Strict On-Device Processing</h2>
            <p>
              OmniSight operates entirely in local memory. Video streams are processed in real time
              and discarded instantly. No biometric data ever leaves your device.
            </p>
          </div>

          <div className="security-pills" aria-label="Security capabilities">
            {['Zero Cloud Storage', 'HIPAA & GDPR Ready', 'End-to-End Encryption'].map((item) => (
              <div className="security-pill" key={item}>
                <Check size={17} />
                {item}
              </div>
            ))}
          </div>

          <ShieldCheck className="security-watermark" size={240} aria-hidden="true" />
        </section>
      </main>

      <footer className="site-footer">
        <span>OMNISIGHT | Clinical Medical Tech Platform</span>
        <nav aria-label="Footer navigation">
          <a href="#top">Privacy</a>
          <a href="#top">Terms</a>
          <a href="#features">Clinical Validation</a>
        </nav>
        <span>Copyright 2025 OmniSight Inc. All rights reserved.</span>
      </footer>

      {isModalOpen && (
        <AssessmentModal
          onClose={() => setIsModalOpen(false)}
          sessionId={sessionId}
          setSessionId={setSessionId}
          analysisResult={analysisResult}
          setAnalysisResult={setAnalysisResult}
          isAnalyzing={isAnalyzing}
          setIsAnalyzing={setIsAnalyzing}
        />
      )}
    </div>
  );
}

export default App;
