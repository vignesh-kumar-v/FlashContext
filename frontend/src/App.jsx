import { useState, useRef, useEffect, useCallback } from "react";
import katex from "katex";
import "katex/dist/katex.min.css";

const API_WS = "ws://localhost:8000/ws";

function generateId() {
  return Math.random().toString(36).slice(2, 10);
}

const STEP_LABELS = {
  extracting_keywords: "Extracting keywords",
  fetching_papers: "Searching Semantic Scholar",
  downloading_pdfs: "Downloading PDFs",
  initializing_qdrant: "Initializing vector DB",
  chunking_storing: "Chunking & embedding",
  done: "Done",
  error: "Error",
};

const STEP_ICONS = {
  extracting_keywords: "🔍",
  fetching_papers: "📚",
  downloading_pdfs: "⬇️",
  initializing_qdrant: "🗄️",
  chunking_storing: "🧩",
  done: "✅",
  error: "❌",
};

function renderLatex(text) {
  if (!text || !text.includes("$")) return text;
  const parts = [];
  let remaining = text;
  let key = 0;
  let iterations = 0;
  const MAX_ITERATIONS = 500;

  while (remaining.length > 0 && iterations < MAX_ITERATIONS) {
    iterations++;
    const displayMatch = remaining.match(/\$\$([\s\S]*?)\$\$/);
    const inlineMatch = remaining.match(/(?<!\$)\$(?!\$)([\s\S]*?)\$(?!\$)/);

    let match, isDisplay;
    if (displayMatch && (!inlineMatch || displayMatch.index <= inlineMatch.index)) {
      match = displayMatch;
      isDisplay = true;
    } else if (inlineMatch) {
      match = inlineMatch;
      isDisplay = false;
    } else {
      parts.push(<span key={key++}>{remaining}</span>);
      break;
    }

    if (match.index > 0) {
      parts.push(<span key={key++}>{remaining.slice(0, match.index)}</span>);
    }

    try {
      const html = katex.renderToString(match[1], {
        displayMode: isDisplay,
        throwOnError: false,
      });
      parts.push(
        <span
          key={key++}
          className={isDisplay ? "block text-center my-3" : ""}
          dangerouslySetInnerHTML={{ __html: html }}
        />
      );
    } catch {
      parts.push(<span key={key++} className="font-mono text-sm">{match[0]}</span>);
    }

    remaining = remaining.slice(match.index + match[0].length);
  }

  if (iterations >= MAX_ITERATIONS) {
    parts.push(<span key={key++}>{remaining}</span>);
  }

  return parts;
}

function createSession() {
  return {
    id: generateId(),
    messages: [],
    dbReady: false,
    showWelcome: true,
  };
}

export default function App() {
  const [sessions, setSessions] = useState(() => {
    const initial = createSession();
    return { [initial.id]: initial };
  });
  const [activeSession, setActiveSession] = useState(() => Object.keys(sessions)[0]);
  const [input, setInput] = useState("");
  const [status, setStatus] = useState("idle");
  const [progress, setProgress] = useState(null);
  const [streamingText, setStreamingText] = useState("");
  const [streamingSources, setStreamingSources] = useState(null);
  const wsRef = useRef(null);
  const chatRef = useRef(null);

  const session = sessions[activeSession];
  const sessionId = activeSession;

  const updateSession = useCallback((updater) => {
    setSessions((prev) => ({
      ...prev,
      [activeSession]: { ...prev[activeSession], ...updater },
    }));
  }, [activeSession]);

  const newSession = useCallback(() => {
    if (status !== "idle") return;
    const s = createSession();
    setSessions((prev) => ({ ...prev, [s.id]: s }));
    setActiveSession(s.id);
    setInput("");
    setProgress(null);
    setStreamingText("");
    setStreamingSources(null);
  }, [status]);

  const switchSession = useCallback((id) => {
    if (status !== "idle" || id === activeSession) return;
    setActiveSession(id);
    setInput("");
    setProgress(null);
    setStreamingText("");
    setStreamingSources(null);
  }, [status, activeSession]);

  useEffect(() => {
    chatRef.current?.scrollTo({ top: chatRef.current.scrollHeight, behavior: "smooth" });
  }, [session?.messages, streamingText, progress]);

  const sendMessage = useCallback(
    (type, query) => {
      if (!query.trim() || status !== "idle") return;

      const msgId = generateId();
      updateSession({ messages: [...session.messages, { id: msgId, role: "user", content: query }], showWelcome: false });
      setInput("");
      setStatus(type === "scrape" ? "scraping" : "querying");
      setProgress(null);
      setStreamingText("");
      setStreamingSources(null);

      const ws = new WebSocket(API_WS);
      wsRef.current = ws;

      ws.onopen = () => {
        ws.send(JSON.stringify({ type, query, session_id: sessionId }));
      };

      ws.onmessage = (e) => {
        const data = JSON.parse(e.data);

        if (data.token) {
          setStreamingText((prev) => prev + data.token);
        } else if (data.status === "PROGRESS") {
          setProgress(data.progress);
        } else if (data.status === "SUCCESS") {
          if (type === "scrape") {
            updateSession({
              dbReady: true,
              messages: [
                ...session.messages,
                { id: msgId, role: "user", content: query },
                {
                  id: generateId(),
                  role: "assistant",
                  content: "Papers fetched and indexed. You can now ask questions about this topic.",
                  sources: null,
                },
              ],
            });
          } else {
            const result = data.result;
            updateSession({
              messages: [
                ...session.messages,
                { id: msgId, role: "user", content: query },
                {
                  id: generateId(),
                  role: "assistant",
                  content: result.answer || streamingText,
                  sources: result.sources || [],
                },
              ],
            });
            setStreamingText("");
            setStreamingSources(null);
          }
          setStatus("idle");
          setProgress(null);
          ws.close();
        } else if (data.status === "FAILURE" || data.status === "ERROR") {
          updateSession({
            messages: [
              ...session.messages,
              { id: msgId, role: "user", content: query },
              {
                id: generateId(),
                role: "assistant",
                content: `Error: ${data.result?.error || "Something went wrong"}`,
                sources: null,
              },
            ],
          });
          setStatus("idle");
          setProgress(null);
          setStreamingText("");
          ws.close();
        }
      };

      ws.onerror = () => {
        updateSession({
          messages: [
            ...session.messages,
            { id: msgId, role: "user", content: query },
            { id: generateId(), role: "assistant", content: "Connection error. Is the backend running?", sources: null },
          ],
        });
        setStatus("idle");
        setProgress(null);
        setStreamingText("");
      };
    },
    [status, sessionId, session, updateSession, streamingText]
  );

  const handleScrape = () => sendMessage("scrape", input);
  const handleQuery = () => sendMessage("query", input);
  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      session.dbReady ? handleQuery() : handleScrape();
    }
  };

  return (
    <div className="h-screen flex flex-col overflow-hidden">
      <Header
        sessionId={sessionId}
        dbReady={session.dbReady}
        onNewSession={newSession}
        sessions={sessions}
        activeSession={activeSession}
        onSwitchSession={switchSession}
      />
      <ChatArea
        messages={session.messages}
        status={status}
        progress={progress}
        chatRef={chatRef}
        showWelcome={session.showWelcome}
        streamingText={streamingText}
        streamingSources={streamingSources}
      />
      <InputBar
        input={input}
        setInput={setInput}
        onSend={session.dbReady ? handleQuery : handleScrape}
        onKeyDown={handleKeyDown}
        status={status}
        dbReady={session.dbReady}
      />
    </div>
  );
}

function Header({ sessionId, dbReady, onNewSession, sessions, activeSession, onSwitchSession }) {
  const [open, setOpen] = useState(false);
  const sessionList = Object.values(sessions);

  return (
    <header className="glass shrink-0 px-6 py-3 flex items-center justify-between z-10">
      <div className="flex items-center gap-3">
        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center">
          <svg className="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M13 10V3L4 14h7v7l9-11h-7z" />
          </svg>
        </div>
        <h1 className="text-lg font-semibold tracking-tight">
          <span className="gradient-text">Flash</span>
          <span className="text-gray-300">Context</span>
        </h1>
      </div>
      <div className="flex items-center gap-3 text-sm">
        {sessionList.length > 1 && (
          <div className="relative">
            <button
              onClick={() => setOpen(!open)}
              className="glass-hover glass rounded-lg px-3 py-1.5 text-xs text-gray-400 hover:text-gray-200 transition-all flex items-center gap-1.5"
            >
              <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4" />
              </svg>
              Sessions ({sessionList.length})
            </button>
            {open && (
              <>
                <div className="fixed inset-0 z-10" onClick={() => setOpen(false)} />
                <div className="absolute right-0 top-full mt-1 w-64 glass rounded-xl py-1 z-20 shadow-2xl">
                  {sessionList.map((s) => (
                    <button
                      key={s.id}
                      onClick={() => { onSwitchSession(s.id); setOpen(false); }}
                      className={`w-full text-left px-4 py-2 text-xs flex items-center justify-between transition-colors ${
                        s.id === activeSession
                          ? "bg-indigo-500/20 text-indigo-300"
                          : "text-gray-400 hover:bg-gray-800/50 hover:text-gray-200"
                      }`}
                    >
                      <span className="flex items-center gap-2">
                        <code className="font-mono text-[10px] bg-gray-800/50 px-1.5 py-0.5 rounded">{s.id}</code>
                        {s.dbReady && <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />}
                      </span>
                      <span className="text-gray-600 text-[10px]">{s.messages.length} msgs</span>
                    </button>
                  ))}
                </div>
              </>
            )}
          </div>
        )}
        <div className="flex items-center gap-2 text-gray-400">
          <code className="font-mono text-[10px] bg-gray-800/50 px-2 py-0.5 rounded text-indigo-400">{sessionId}</code>
        </div>
        {dbReady && (
          <div className="flex items-center gap-1.5 text-emerald-400">
            <div className="w-2 h-2 rounded-full bg-emerald-400 pulse-dot" />
            <span className="text-xs font-medium">DB Ready</span>
          </div>
        )}
        <button
          onClick={onNewSession}
          className="glass-hover glass rounded-lg px-3 py-1.5 text-xs text-gray-400 hover:text-gray-200 transition-all flex items-center gap-1.5"
        >
          <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M12 4v16m8-8H4" />
          </svg>
          New Session
        </button>
      </div>
    </header>
  );
}

function ChatArea({ messages, status, progress, chatRef, showWelcome, streamingText, streamingSources }) {
  return (
    <main ref={chatRef} className="flex-1 overflow-y-auto px-4 py-6 space-y-4">
      {showWelcome && messages.length === 0 && <WelcomeMessage />}
      {messages.map((msg) => (
        <Message key={msg.id} msg={msg} />
      ))}
      {streamingText && (
        <StreamingMessage text={streamingText} sources={streamingSources} />
      )}
      {status !== "idle" && !streamingText && (
        status === "scraping" ? (
          <ProgressCard status={status} progress={progress} />
        ) : (
          <LoadingBubble />
        )
      )}
    </main>
  );
}

function WelcomeMessage() {
  return (
    <div className="flex flex-col items-center justify-center h-full text-center fade-in">
      <div className="w-20 h-20 rounded-2xl bg-gradient-to-br from-indigo-500/20 to-purple-600/20 border border-indigo-500/20 flex items-center justify-center mb-6">
        <svg className="w-10 h-10 text-indigo-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09zM18.259 8.715L18 9.75l-.259-1.035a3.375 3.375 0 00-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 002.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 002.455 2.456L21.75 6l-1.036.259a3.375 3.375 0 00-2.455 2.456z" />
        </svg>
      </div>
      <h2 className="text-2xl font-semibold text-gray-100 mb-3">
        What are you working on today?
      </h2>
      <p className="text-gray-500 max-w-md text-sm leading-relaxed">
        Tell me your research topic and I'll fetch relevant papers, index them, and answer your questions.
      </p>
      <div className="mt-8 flex flex-wrap gap-3 justify-center max-w-lg">
        {[
          "Diffusion models for image generation",
          "Transformer attention mechanisms",
          "Reinforcement learning from human feedback",
          "Graph neural networks for drug discovery",
        ].map((suggestion) => (
          <button
            key={suggestion}
            onClick={() => {
              const input = document.querySelector("textarea");
              if (input) {
                const nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, "value").set;
                nativeInputValueSetter.call(input, suggestion);
                input.dispatchEvent(new Event("input", { bubbles: true }));
                input.focus();
              }
            }}
            className="glass-hover glass rounded-full px-4 py-2 text-xs text-gray-400 hover:text-gray-200 transition-all cursor-pointer"
          >
            {suggestion}
          </button>
        ))}
      </div>
    </div>
  );
}

function Message({ msg }) {
  const isUser = msg.role === "user";
  return (
    <div className={`flex gap-3 slide-up ${isUser ? "justify-end" : "justify-start"}`}>
      {!isUser && (
        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center shrink-0 mt-0.5">
          <svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M13 10V3L4 14h7v7l9-11h-7z" />
          </svg>
        </div>
      )}
      <div className={`max-w-[75%] ${isUser ? "order-first" : ""}`}>
        <div
          className={`rounded-2xl px-4 py-3 text-sm leading-relaxed ${
            isUser
              ? "bg-indigo-600/30 border border-indigo-500/20 text-gray-100"
              : "glass text-gray-200"
          }`}
        >
          <div className="whitespace-pre-wrap">{isUser ? msg.content : renderLatex(msg.content)}</div>
        </div>
        {msg.sources && msg.sources.length > 0 && (
          <div className="mt-2 space-y-1">
            <p className="text-xs text-gray-500 font-medium mb-1">Sources</p>
            {msg.sources.map((s, i) => (
              <div key={i} className="glass rounded-lg px-3 py-1.5 text-xs flex items-center justify-between">
                <span className="text-gray-300 truncate mr-2">{s.title}</span>
                <span className="text-indigo-400 font-mono shrink-0">{s.score.toFixed(3)}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function StreamingMessage({ text, sources }) {
  return (
    <div className="flex gap-3 slide-up">
      <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center shrink-0 mt-0.5">
        <svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M13 10V3L4 14h7v7l9-11h-7z" />
        </svg>
      </div>
      <div className="max-w-[75%]">
        <div className="glass rounded-2xl px-4 py-3 text-sm leading-relaxed text-gray-200 relative overflow-hidden">
          <div className="whitespace-pre-wrap">
            {text}
            <span className="inline-block w-2 h-4 bg-indigo-400 ml-0.5 align-middle animate-pulse rounded-sm" />
          </div>
          <div
            className="absolute bottom-0 left-0 right-0 h-px"
            style={{
              background: "linear-gradient(90deg, transparent, #818cf8, #c084fc, transparent)",
              animation: "shimmer 2s ease-in-out infinite",
            }}
          />
        </div>
        {sources && sources.length > 0 && (
          <div className="mt-2 space-y-1">
            <p className="text-xs text-gray-500 font-medium mb-1">Sources</p>
            {sources.map((s, i) => (
              <div key={i} className="glass rounded-lg px-3 py-1.5 text-xs flex items-center justify-between">
                <span className="text-gray-300 truncate mr-2">{s.title}</span>
                <span className="text-indigo-400 font-mono shrink-0">{s.score.toFixed(3)}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function ProgressCard({ status, progress }) {
  const step = progress?.step || "";
  const detail = progress?.detail || "";

  const steps = ["extracting_keywords", "fetching_papers", "downloading_pdfs", "initializing_qdrant", "chunking_storing", "done"];

  const currentIdx = steps.indexOf(step);
  const progressPct = step === "done" ? 100 : step === "error" ? 0 : Math.max(5, ((currentIdx) / steps.length) * 100);

  return (
    <div className="flex gap-3 slide-up">
      <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center shrink-0 mt-0.5">
        <svg className="w-4 h-4 text-white animate-spin" fill="none" viewBox="0 0 24 24">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
        </svg>
      </div>
      <div className="glass rounded-2xl px-5 py-4 flex-1 max-w-md">
        <p className="text-sm font-medium text-gray-200 mb-3">Fetching & indexing papers</p>

        <div className="w-full h-2 bg-gray-800 rounded-full overflow-hidden mb-3">
          <div
            className="h-full rounded-full transition-all duration-700 ease-out"
            style={{
              width: `${progressPct}%`,
              background: "linear-gradient(90deg, #818cf8, #c084fc, #f472b6)",
            }}
          />
        </div>

        <div className="flex gap-1.5 mb-3">
          {steps.map((s, i) => {
            const done = i < currentIdx;
            const active = i === currentIdx;
            return (
              <div
                key={s}
                className="flex-1 h-1 rounded-full transition-all duration-500"
                style={{
                  background: done
                    ? "linear-gradient(90deg, #818cf8, #c084fc)"
                    : active
                    ? "#6366f1"
                    : "#374151",
                }}
              />
            );
          })}
        </div>

        <div className="flex items-center gap-2">
          <span className="text-sm">{STEP_ICONS[step] || "⏳"}</span>
          <span className="text-sm text-gray-300">{STEP_LABELS[step] || step}</span>
        </div>
        {detail && (
          <p className="text-xs text-indigo-400 mt-1.5 ml-7 typing-cursor">{detail}</p>
        )}
      </div>
    </div>
  );
}

function LoadingBubble() {
  return (
    <div className="flex gap-3 slide-up">
      <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center shrink-0 mt-0.5">
        <svg className="w-4 h-4 text-white animate-spin" fill="none" viewBox="0 0 24 24">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
        </svg>
      </div>
      <div className="glass rounded-2xl px-4 py-3">
        <p className="text-sm text-gray-300">Searching & generating...</p>
      </div>
    </div>
  );
}

function InputBar({ input, setInput, onSend, onKeyDown, status, dbReady }) {
  const busy = status !== "idle";
  const placeholder = dbReady
    ? "Ask a question about the papers..."
    : "Tell me what you're working on...";

  return (
    <footer className="glass shrink-0 px-4 py-3 z-10">
      <div className="max-w-3xl mx-auto flex gap-2">
        <div className="flex-1 relative gradient-border rounded-xl">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder={placeholder}
            disabled={busy}
            rows={1}
            className="w-full bg-gray-900/80 rounded-xl px-4 py-3 text-sm text-gray-100 placeholder-gray-600 resize-none outline-none disabled:opacity-50"
          />
        </div>
        <button
          onClick={onSend}
          disabled={busy || !input.trim()}
          className="shrink-0 w-11 h-11 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center disabled:opacity-40 disabled:cursor-not-allowed hover:from-indigo-400 hover:to-purple-500 transition-all"
        >
          {busy ? (
            <svg className="w-5 h-5 text-white animate-spin" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
            </svg>
          ) : (
            <svg className="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 12L3.269 3.126A59.768 59.768 0 0121.485 12 59.77 59.77 0 013.27 20.876L5.999 12zm0 0h7.5" />
            </svg>
          )}
        </button>
      </div>
      <p className="text-center text-xs text-gray-600 mt-2">
        {dbReady ? "DB populated — asking questions" : "First message will fetch and index papers"}
      </p>
    </footer>
  );
}
