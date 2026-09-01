import { useState } from 'react';
import './App.css';
import CameraRecord from './components/CameraRecord';
import Dashboard from './components/Dashboard';
import Chat from './components/Chat';
import { Activity } from 'lucide-react';

function App() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [analysisResult, setAnalysisResult] = useState<any>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  return (
    <div className="app-container">
      <header className="header animate-fade-in">
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <Activity size={24} color="var(--primary)" />
          <h1>Nexus Health</h1>
        </div>
        <div className="text-label">
          {sessionId ? `Session: ${sessionId.substring(0, 8)}...` : 'Ready'}
        </div>
      </header>

      <main className="main-content">
        <section className="left-panel">
          <CameraRecord 
            sessionId={sessionId}
            setSessionId={setSessionId}
            setAnalysisResult={setAnalysisResult}
            isAnalyzing={isAnalyzing}
            setIsAnalyzing={setIsAnalyzing}
          />
        </section>

        <section className="center-panel">
          <Dashboard result={analysisResult} isAnalyzing={isAnalyzing} />
        </section>

        <section className="right-panel">
          <Chat sessionId={sessionId} result={analysisResult} setAnalysisResult={setAnalysisResult} />
        </section>
      </main>
    </div>
  );
}

export default App;
