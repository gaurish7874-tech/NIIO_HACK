import { useState } from 'react';
import { Send, MessageCircle } from 'lucide-react';

interface ChatProps {
  sessionId: string | null;
  result: any;
  setAnalysisResult: (result: any) => void;
}

interface Message {
  role: 'system' | 'user' | 'agent';
  content: string;
}

export default function Chat({ sessionId, result, setAnalysisResult }: ChatProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const triage = result?.triage;
  const needsClarification = triage?.needs_clarification;
  const clarifyingQuestion = triage?.clarifying_question;

  const sendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || !sessionId) return;
    
    const userMsg = input.trim();
    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: userMsg }]);
    setIsLoading(true);

    try {
      if (needsClarification) {
        // Submit as clarification
        const res = await fetch(`/sessions/${sessionId}/clarification`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ answer: userMsg })
        });
        const data = await res.json();
        if (res.ok) {
          setAnalysisResult(data);
          setMessages(prev => [...prev, { role: 'system', content: 'Clarification received. Assessment updated.' }]);
        } else {
           throw new Error(data.detail || 'Clarification failed');
        }
      } else {
        // Submit as general chat
        const res = await fetch(`/sessions/${sessionId}/chat`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: userMsg })
        });
        const data = await res.json();
        if (res.ok) {
          setMessages(prev => [...prev, { role: 'agent', content: data.response }]);
        } else {
           throw new Error(data.detail || 'Chat failed');
        }
      }
    } catch (err: any) {
      setMessages(prev => [...prev, { role: 'system', content: `Error: ${err.message}` }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="surface animate-fade-in" style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <div style={{ padding: '1.25rem', borderBottom: '1px solid var(--border)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
        <MessageCircle size={18} color="var(--primary)" /> 
        <h2 style={{ fontSize: '1rem', margin: 0, fontWeight: 600 }}>Nexus Assistant</h2>
      </div>

      <div style={{ flex: 1, padding: '1.25rem', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
        {messages.length === 0 && !needsClarification && (
          <div style={{ textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.9rem', marginTop: '3rem' }}>
            Awaiting analysis to provide context.
          </div>
        )}

        {needsClarification && (
          <div style={{ background: 'var(--info-light)', padding: '1rem', borderRadius: 'var(--radius-sm)', borderLeft: '3px solid var(--info)' }}>
            <div className="text-label" style={{ color: '#1e3a8a', marginBottom: '0.25rem' }}>Follow-up Question</div>
            <div style={{ fontSize: '0.95rem', color: '#1e3a8a' }}>{clarifyingQuestion}</div>
          </div>
        )}

        {messages.map((msg, i) => {
          const isUser = msg.role === 'user';
          const isSystem = msg.role === 'system';
          
          if (isSystem) {
             return (
               <div key={i} style={{ alignSelf: 'center', fontSize: '0.8rem', color: 'var(--text-muted)', margin: '0.5rem 0' }}>
                 {msg.content}
               </div>
             )
          }

          return (
            <div key={i} style={{ 
              alignSelf: isUser ? 'flex-end' : 'flex-start',
              maxWidth: '85%',
              background: isUser ? 'var(--primary)' : 'var(--surface-hover)',
              color: isUser ? 'var(--text-inverse)' : 'var(--text-main)',
              padding: '0.75rem 1rem',
              borderRadius: 'var(--radius-md)',
              borderBottomRightRadius: isUser ? '4px' : 'var(--radius-md)',
              borderBottomLeftRadius: isUser ? 'var(--radius-md)' : '4px',
              fontSize: '0.95rem',
              lineHeight: 1.5,
              border: isUser ? 'none' : '1px solid var(--border)'
            }}>
              {msg.content}
            </div>
          )
        })}
        {isLoading && (
          <div style={{ alignSelf: 'flex-start', color: 'var(--text-muted)', fontSize: '0.85rem' }}>Typing...</div>
        )}
      </div>

      <form onSubmit={sendMessage} style={{ padding: '1rem', borderTop: '1px solid var(--border)', display: 'flex', gap: '0.5rem', background: 'var(--surface)' }}>
        <input 
          type="text" 
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={needsClarification ? "Answer here..." : "Ask a question..."}
          disabled={!sessionId || (!result && !needsClarification)}
          style={{ 
            flex: 1, 
            background: 'var(--surface-hover)', 
            border: '1px solid var(--border)', 
            padding: '0.75rem 1rem', 
            borderRadius: 'var(--radius-full)', 
            color: 'var(--text-main)',
            outline: 'none',
            fontSize: '0.95rem',
            fontFamily: 'inherit'
          }}
        />
        <button 
          type="submit" 
          disabled={!input.trim() || isLoading || !sessionId}
          style={{ 
            background: 'var(--primary)', 
            color: 'var(--text-inverse)', 
            width: '42px', 
            height: '42px', 
            borderRadius: '50%', 
            display: 'flex', 
            alignItems: 'center', 
            justifyContent: 'center',
            transition: 'opacity 0.2s',
            cursor: (!input.trim() || isLoading || !sessionId) ? 'not-allowed' : 'pointer',
            opacity: (!input.trim() || isLoading || !sessionId) ? 0.5 : 1,
            border: 'none'
          }}
        >
          <Send size={18} />
        </button>
      </form>
    </div>
  );
}
