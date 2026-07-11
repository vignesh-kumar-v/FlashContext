import { useState, useRef, useEffect, useCallback } from "react";

const API_WS = "ws://localhost:8000/ws";

function generateId() {
  return Math.random().toString(36).slice(2, 10);
}

export default function App() {
  const [sessionId] = useState(() => generateId());
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [status, setStatus] = useState("idle"); // idle | scraping | querying
  const [progress, setProgress] = useState(null);
  const [dbReady, setDbReady] = useState(false);
  const wsRef = useRef(null);
  const chatRef = useRef(null);

  useEffect(() => {
    chatRef.current?.scrollTo({ top: chatRef.current.scrollHeight, behavior: "smooth" });
  }, [messages]);

  const sendMessage = useCallback(
    (type, query) => {
      if (!query.trim() || status !== "idle") return;

      const msgId = generateId();
      setMessages((prev) => [...prev, { id: msgId, role: "user", content: query }]);
      setInput("");
      setStatus(type === "scrape" ? "scraping" : "querying");
      setProgress(null);

      const ws = new WebSocket(API_WS);
      wsRef.current = ws;

      ws.onopen = () => {
        ws.send(JSON.stringify({ type, query, session_id: sessionId }));
      };

      ws.onmessage = (e) => {
        const data = JSON.parse(e.data);
        if (data.status === "PROGRESS") {
          setProgress(data.progress);
        } else if (data.status === "SUCCESS") {
          if (type === "scrape") {
            setDbReady(true);
            setMessages((prev) => [
              ...prev,
              {
                id: generateId(),
                role: "assistant",
                content: "Papers fetched and indexed. You can now ask questions about this topic.",
                sources: null,
              },
            ]);
          } else {
            setMessages((prev) => [
              ...prev,
              {
                id: generateId(),
                role: "assistant",
                content: data.result.answer,
                sources: data.result.sources || [],
              },
            ]);
          }
          setStatus("idle");
          setProgress(null);
          ws.close();
        } else if (data.status === "FAILURE" || data.status === "ERROR") {
          setMessages((prev) => [
            ...prev,
            {
              id: generateId(),
              role: "assistant",
              content: `Error: ${data.result?.error || "Something went wrong"}`,
              sources: null,
            },
          ]);
          setStatus("idle");
          setProgress(null);
          ws.close();
        }
      };

      ws.onerror = () => {
        setMessages((prev) => [
          ...prev,
          { id: generateId(), role: "assistant", content: "Connection error. Is the backend running?", sources: null },
        ]);
        setStatus("idle");
        setProgress(null);
      };
    },
    [status, sessionId]
  );

  const handleScrape = () => sendMessage("scrape", input);
  const handleQuery = () => sendMessage("query", input);
  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      dbReady ? handleQuery() : handleScrape();
    }
  };

  return (
    <div className="h-screen flex flex-col overflow-hidden">
      <Header sessionId={sessionId} dbReady={dbReady} />
      <ChatArea messages={messages} status={status} progress={progress} chatRef={chatRef} />
      <InputBar
        input={input}
        setInput={setInput}
        onSend={dbReady ? handleQuery : handleScrape}
        onKeyDown={handleKeyDown}
        status={status}
        dbReady={dbReady}
      />
    </div>
  );
}

function Header({ sessionId, dbReady }) {
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
      <div className="flex items-center gap-4 text-sm">
        <div className="flex items-center gap-2 text-gray-400">
          <span className="text-gray-600">session</span>
          <code className="font-mono text-xs bg-gray-800/50 px-2 py-0.5 rounded text-indigo-400">{sessionId}</code>
        </div>
        {dbReady && (
          <div className="flex items-center gap-1.5 text-emerald-400">
            <div className="w-2 h-2 rounded-full bg-emerald-400 pulse-dot" />
            <span className="text-xs font-medium">DB Ready</span>
          </div>
        )}
      </div>
    </header>
  );
}

function ChatArea({ messages, status, progress, chatRef }) {
  return (
    <main ref={chatRef} className="flex-1 overflow-y-auto px-4 py-6 space-y-4">
      {messages.length === 0 && (
        <div className="flex flex-col items-center justify-center h-full text-center fade-in">
          <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-indigo-500/20 to-purple-600/20 border border-indigo-500/20 flex items-center justify-center mb-6">
            <svg className="w-8 h-8 text-indigo-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09zM18.259 8.715L18 9.75l-.259-1.035a3.375 3.375 0 00-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 002.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 002.455 2.456L21.75 6l-1.036.259a3.375 3.375 0 00-2.455 2.456z" />
            </svg>
          </div>
          <h2 className="text-xl font-semibold text-gray-200 mb-2">Research RAG Assistant</h2>
          <p className="text-gray-500 max-w-md text-sm leading-relaxed">
            Enter a research topic to fetch and index papers, then ask questions about them.
            Powered by Semantic Scholar, ArXiv, and GLM-5.2.
          </p>
        </div>
      )}
      {messages.map((msg) => (
        <Message key={msg.id} msg={msg} />
      ))}
      {status !== "idle" && <LoadingBubble status={status} progress={progress} />}
    </main>
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
          <div className="whitespace-pre-wrap">{msg.content}</div>
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

function LoadingBubble({ status, progress }) {
  const label = status === "scraping" ? "Fetching & indexing papers" : "Searching & generating";
  const step = progress?.step || "";
  const stepLabels = {
    extracting_keywords: "Extracting keywords...",
    searching_qdrant: "Searching knowledge base...",
    generating_response: "Generating answer...",
  };

  return (
    <div className="flex gap-3 slide-up">
      <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center shrink-0 mt-0.5">
        <svg className="w-4 h-4 text-white animate-spin" fill="none" viewBox="0 0 24 24">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
        </svg>
      </div>
      <div className="glass rounded-2xl px-4 py-3">
        <p className="text-sm text-gray-300">{label}</p>
        {step && (
          <p className="text-xs text-indigo-400 mt-1 typing-cursor">
            {stepLabels[step] || step}
          </p>
        )}
      </div>
    </div>
  );
}

function InputBar({ input, setInput, onSend, onKeyDown, status, dbReady }) {
  const busy = status !== "idle";
  const placeholder = dbReady
    ? "Ask a question about the papers..."
    : "Enter a research topic to fetch papers...";

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
