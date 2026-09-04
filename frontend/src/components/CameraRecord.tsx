import { useState, useRef, useEffect } from 'react';
import { Camera, Video, Square, Loader2 } from 'lucide-react';

interface CameraRecordProps {
  sessionId: string | null;
  setSessionId: (id: string) => void;
  setAnalysisResult: (result: any) => void;
  isAnalyzing: boolean;
  setIsAnalyzing: (analyzing: boolean) => void;
}

export default function CameraRecord({ sessionId, setSessionId, setAnalysisResult, isAnalyzing, setIsAnalyzing }: CameraRecordProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [errorMsg, setErrorMsg] = useState('');
  
  // Track if we are in continuous live mode
  const isLiveMode = useRef(false);
  // Keep track of the active recorder so we can stop it if needed
  const activeRecorder = useRef<MediaRecorder | null>(null);

  useEffect(() => {
    return () => {
      isLiveMode.current = false;
      if (activeRecorder.current && activeRecorder.current.state === 'recording') {
        activeRecorder.current.stop();
      }
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
    };
  }, [stream]);

  const startCamera = async () => {
    try {
      const s = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      setStream(s);
      if (videoRef.current) {
        videoRef.current.srcObject = s;
      }
      setErrorMsg('');
    } catch (err: any) {
      setErrorMsg(`Camera access failed: ${err.message}`);
    }
  };

  const startLiveMonitor = async () => {
    try {
      setErrorMsg('');
      let currentSessionId = sessionId;
      if (!currentSessionId) {
        const res = await fetch('/sessions', { method: 'POST' });
        if (!res.ok) throw new Error('Failed to create session');
        const data = await res.json();
        currentSessionId = data.session_id;
        setSessionId(currentSessionId as string);
      }

      if (!stream) return;
      
      isLiveMode.current = true;
      setIsRecording(true);
      
      const mimeType = MediaRecorder.isTypeSupported('video/webm;codecs=vp8')
        ? 'video/webm;codecs=vp8'
        : 'video/webm';
        
      recordNextChunk(currentSessionId as string, mimeType);
      
    } catch (err: any) {
      setErrorMsg(`Failed to start monitoring: ${err.message}`);
      setIsRecording(false);
      isLiveMode.current = false;
    }
  };

  const recordNextChunk = (sid: string, mimeType: string) => {
    if (!isLiveMode.current || !stream) return;

    const mediaRecorder = new MediaRecorder(stream, { mimeType });
    activeRecorder.current = mediaRecorder;
    const localChunks: Blob[] = [];

    mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) localChunks.push(e.data);
    };

    mediaRecorder.onstop = () => {
      if (localChunks.length > 0) {
        uploadChunk(sid, mimeType, localChunks);
      }
      // Start the next chunk immediately if still in live mode
      if (isLiveMode.current) {
        recordNextChunk(sid, mimeType);
      }
    };

    mediaRecorder.start();

    // Record for 5 seconds
    setTimeout(() => {
      if (mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
      }
    }, 5000);
  };

  const stopLiveMonitor = () => {
    isLiveMode.current = false;
    setIsRecording(false);
    if (activeRecorder.current && activeRecorder.current.state === 'recording') {
      activeRecorder.current.stop();
    }
  };

  const uploadChunk = async (sid: string, mimeType: string, chunkData: Blob[]) => {
    setIsAnalyzing(true);
    try {
      const videoBlob = new Blob(chunkData, { type: mimeType });
      const formData = new FormData();
      formData.append('file', videoBlob, 'camera-recording.webm');
      
      const res = await fetch(`/sessions/${sid}/analyze`, {
        method: 'POST',
        body: formData
      });
      
      const result = await res.json();
      if (!res.ok) throw new Error(result.detail || 'Analysis failed');
      
      setAnalysisResult(result);
    } catch (err: any) {
      setErrorMsg(err.message);
    } finally {
      // We don't set isAnalyzing to false immediately because the next chunk might be recording/uploading soon,
      // but if we stopped, we should clear it.
      if (!isLiveMode.current) {
        setIsAnalyzing(false);
      }
    }
  };

  return (
    <div className="surface animate-fade-in" style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h2 style={{ fontSize: '1rem', margin: 0, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Camera size={18} /> Sensor Input
        </h2>
        {isRecording && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', color: 'var(--danger)', fontSize: '0.8rem', fontWeight: 600 }}>
            <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: 'var(--danger)' }}></span>
            REC
          </div>
        )}
      </div>

      <div style={{ 
        position: 'relative', 
        width: '100%', 
        aspectRatio: '4/3', 
        background: 'var(--surface-hover)', 
        borderRadius: 'var(--radius-sm)', 
        overflow: 'hidden',
        border: '1px solid var(--border)'
      }}>
        <video 
          ref={videoRef} 
          autoPlay 
          muted 
          playsInline 
          style={{ width: '100%', height: '100%', objectFit: 'cover', transform: 'scaleX(-1)' }}
        />
        {!stream && (
          <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)', fontSize: '0.9rem' }}>
            Camera off
          </div>
        )}
      </div>

      {errorMsg && (
        <div style={{ padding: '0.75rem', background: 'var(--danger-light)', borderRadius: 'var(--radius-sm)', color: 'var(--danger)', fontSize: '0.85rem' }}>
          {errorMsg}
        </div>
      )}

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem' }}>
        {!stream ? (
          <button className="btn btn-outline" onClick={startCamera} style={{ gridColumn: 'span 2' }}>
            Enable Camera
          </button>
        ) : (
          <>
            <button 
              className="btn btn-primary" 
              onClick={startLiveMonitor} 
              disabled={isRecording}
            >
              <Video size={16} /> Live Monitor
            </button>
            <button 
              className="btn btn-outline" 
              onClick={stopLiveMonitor} 
              disabled={!isRecording}
            >
              <Square size={16} /> Stop
            </button>
          </>
        )}
      </div>

      {isAnalyzing && (
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem', padding: '1rem', background: 'var(--surface-hover)', borderRadius: 'var(--radius-sm)', color: 'var(--primary)' }}>
          <Loader2 size={16} className="animate-spin" style={{ animation: 'spin 1s linear infinite' }} />
          <span style={{ fontSize: '0.9rem', fontWeight: 500 }}>Processing signals...</span>
          <style>{`@keyframes spin { 100% { transform: rotate(360deg); } }`}</style>
        </div>
      )}
    </div>
  );
}
