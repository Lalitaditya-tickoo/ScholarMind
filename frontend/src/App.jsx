import { useState, useRef, useEffect } from "react";
import {
  Upload, Search, FileText, Trash2, BookOpen,
  Send, Loader2, ChevronDown, ChevronUp, Sparkles,
  AlertCircle, CheckCircle2, Brain, X,
} from "lucide-react";

const API = "http://localhost:8000";

export default function App() {
  const [papers, setPapers] = useState([]);
  const [conversations, setConversations] = useState([]);
  const [query, setQuery] = useState("");
  const [uploading, setUploading] = useState(false);
  const [querying, setQuerying] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [selectedPapers, setSelectedPapers] = useState([]);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);
  const chatEndRef = useRef(null);

  // ── Auto-scroll chat ──
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [conversations]);

  // ── Fetch papers on mount ──
  useEffect(() => {
    fetchPapers();
  }, []);

  async function fetchPapers() {
    try {
      const res = await fetch(`${API}/papers`);
      const data = await res.json();
      setPapers(data.papers || []);
    } catch (e) {
      console.error("Failed to fetch papers:", e);
    }
  }

  // ── Upload PDF ──
  async function handleUpload(files) {
    if (!files?.length) return;
    setUploading(true);
    setError(null);

    for (const file of files) {
      if (!file.name.toLowerCase().endsWith(".pdf")) {
        setError(`${file.name} is not a PDF file`);
        continue;
      }

      const formData = new FormData();
      formData.append("file", file);

      try {
        const res = await fetch(`${API}/papers/upload`, {
          method: "POST",
          body: formData,
        });

        if (!res.ok) {
          const err = await res.json();
          throw new Error(err.detail || "Upload failed");
        }

        const paper = await res.json();
        setPapers((prev) => [...prev, paper]);
      } catch (e) {
        setError(`Failed to upload ${file.name}: ${e.message}`);
      }
    }
    setUploading(false);
  }

  // ── Delete paper ──
  async function handleDelete(paperId) {
    try {
      await fetch(`${API}/papers/${paperId}`, { method: "DELETE" });
      setPapers((prev) => prev.filter((p) => p.paper_id !== paperId));
      setSelectedPapers((prev) => prev.filter((id) => id !== paperId));
    } catch (e) {
      setError("Failed to delete paper");
    }
  }

  // ── Ask question ──
  async function handleQuery(e) {
    e.preventDefault();
    if (!query.trim() || querying) return;
    if (papers.length === 0) {
      setError("Upload at least one paper first!");
      return;
    }

    const question = query.trim();
    setQuery("");
    setQuerying(true);
    setError(null);

    // Add user message
    setConversations((prev) => [...prev, { role: "user", content: question }]);

    try {
      const res = await fetch(`${API}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question,
          paper_ids: selectedPapers.length > 0 ? selectedPapers : null,
          top_k: 8,
        }),
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Query failed");
      }

      const data = await res.json();
      setConversations((prev) => [
        ...prev,
        {
          role: "assistant",
          content: data.answer,
          citations: data.citations,
          figures: data.figures_used,
        },
      ]);
    } catch (e) {
      setConversations((prev) => [
        ...prev,
        { role: "error", content: e.message },
      ]);
    }

    setQuerying(false);
  }

  // ── Toggle paper selection ──
  function togglePaper(paperId) {
    setSelectedPapers((prev) =>
      prev.includes(paperId)
        ? prev.filter((id) => id !== paperId)
        : [...prev, paperId]
    );
  }

  // ── Drag & Drop ──
  function handleDrop(e) {
    e.preventDefault();
    setDragOver(false);
    handleUpload(e.dataTransfer.files);
  }

  return (
    <div className="flex h-screen overflow-hidden">
      {/* ═══════════ SIDEBAR ═══════════ */}
      <aside className="w-80 border-r border-white/5 bg-[#0d0d14] flex flex-col">
        {/* Logo */}
        <div className="p-6 border-b border-white/5">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-violet-500 to-amber-400 flex items-center justify-center">
              <Brain className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="font-display text-xl text-white">ScholarMind</h1>
              <p className="text-xs text-white/40 font-mono">multi-modal RAG</p>
            </div>
          </div>
        </div>

        {/* Upload Zone */}
        <div className="p-4">
          <div
            className={`relative border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition-all duration-300 ${
              dragOver
                ? "border-violet-400 bg-violet-500/10"
                : "border-white/10 hover:border-white/20 hover:bg-white/[0.02]"
            }`}
            onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf"
              multiple
              className="hidden"
              onChange={(e) => handleUpload(e.target.files)}
            />
            {uploading ? (
              <Loader2 className="w-8 h-8 mx-auto text-violet-400 animate-spin" />
            ) : (
              <Upload className="w-8 h-8 mx-auto text-white/30" />
            )}
            <p className="text-sm text-white/50 mt-3">
              {uploading ? "Processing..." : "Drop PDFs here"}
            </p>
            <p className="text-xs text-white/25 mt-1">or click to browse</p>
          </div>
        </div>

        {/* Paper List */}
        <div className="flex-1 overflow-y-auto px-4 pb-4">
          <p className="text-xs font-mono text-white/30 uppercase tracking-widest mb-3">
            Papers ({papers.length})
          </p>

          {papers.length === 0 ? (
            <p className="text-sm text-white/20 text-center py-8">
              No papers uploaded yet
            </p>
          ) : (
            <div className="space-y-2">
              {papers.map((paper) => (
                <PaperCard
                  key={paper.paper_id}
                  paper={paper}
                  selected={selectedPapers.includes(paper.paper_id)}
                  onToggle={() => togglePaper(paper.paper_id)}
                  onDelete={() => handleDelete(paper.paper_id)}
                />
              ))}
            </div>
          )}

          {selectedPapers.length > 0 && (
            <div className="mt-3 px-3 py-2 rounded-lg bg-violet-500/10 border border-violet-500/20">
              <p className="text-xs text-violet-300">
                Searching {selectedPapers.length} selected paper{selectedPapers.length > 1 ? "s" : ""}
              </p>
              <button
                onClick={() => setSelectedPapers([])}
                className="text-xs text-violet-400 hover:text-violet-300 mt-1 underline"
              >
                Clear selection (search all)
              </button>
            </div>
          )}
        </div>

        {/* Stats */}
        <div className="p-4 border-t border-white/5 text-xs font-mono text-white/20">
          ChromaDB • sentence-transformers • Claude Sonnet
        </div>
      </aside>

      {/* ═══════════ MAIN CHAT ═══════════ */}
      <main className="flex-1 flex flex-col bg-[#0a0a0f]">
        {/* Error Banner */}
        {error && (
          <div className="mx-6 mt-4 px-4 py-3 rounded-lg bg-red-500/10 border border-red-500/20 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <AlertCircle className="w-4 h-4 text-red-400" />
              <p className="text-sm text-red-300">{error}</p>
            </div>
            <button onClick={() => setError(null)}>
              <X className="w-4 h-4 text-red-400" />
            </button>
          </div>
        )}

        {/* Chat Area */}
        <div className="flex-1 overflow-y-auto px-6 py-8">
          {conversations.length === 0 ? (
            <EmptyState />
          ) : (
            <div className="max-w-3xl mx-auto space-y-6">
              {conversations.map((msg, i) => (
                <ChatMessage key={i} message={msg} />
              ))}
              {querying && (
                <div className="flex items-center gap-3 text-white/40">
                  <Loader2 className="w-5 h-5 animate-spin text-violet-400" />
                  <span className="text-sm animate-pulse-glow">
                    Searching papers & generating answer...
                  </span>
                </div>
              )}
              <div ref={chatEndRef} />
            </div>
          )}
        </div>

        {/* Input Bar */}
        <div className="border-t border-white/5 p-4">
          <form
            onSubmit={handleQuery}
            className="max-w-3xl mx-auto flex items-center gap-3"
          >
            <div className="flex-1 relative">
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Ask about your papers..."
                className="w-full px-5 py-3.5 rounded-xl bg-white/[0.04] border border-white/10 text-white placeholder-white/25 focus:outline-none focus:border-violet-500/50 focus:bg-white/[0.06] transition-all font-sans"
                disabled={querying}
              />
            </div>
            <button
              type="submit"
              disabled={querying || !query.trim()}
              className="p-3.5 rounded-xl bg-gradient-to-r from-violet-600 to-violet-500 text-white disabled:opacity-30 hover:from-violet-500 hover:to-violet-400 transition-all"
            >
              <Send className="w-5 h-5" />
            </button>
          </form>
          <p className="text-center text-xs text-white/15 mt-2 font-mono">
            Powered by Claude Sonnet • Answers cite specific papers and pages
          </p>
        </div>
      </main>
    </div>
  );
}

// ── Paper Card ──
function PaperCard({ paper, selected, onToggle, onDelete }) {
  return (
    <div
      className={`group relative rounded-xl p-3 cursor-pointer transition-all duration-200 ${
        selected
          ? "bg-violet-500/10 border border-violet-500/30"
          : "bg-white/[0.02] border border-transparent hover:bg-white/[0.04] hover:border-white/5"
      }`}
      onClick={onToggle}
    >
      <div className="flex items-start gap-3">
        <div className={`mt-0.5 w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 ${
          selected ? "bg-violet-500/20" : "bg-white/5"
        }`}>
          <FileText className={`w-4 h-4 ${selected ? "text-violet-400" : "text-white/30"}`} />
        </div>
        <div className="flex-1 min-w-0">
          <p className="text-sm text-white/80 font-medium truncate">
            {paper.title || paper.filename}
          </p>
          <p className="text-xs text-white/30 mt-1 font-mono">
            {paper.num_pages} pages • {paper.num_chunks} chunks • {paper.num_figures} figures
          </p>
        </div>
        <button
          onClick={(e) => { e.stopPropagation(); onDelete(); }}
          className="opacity-0 group-hover:opacity-100 p-1.5 rounded-lg hover:bg-red-500/10 transition-all"
        >
          <Trash2 className="w-3.5 h-3.5 text-red-400" />
        </button>
      </div>
    </div>
  );
}

// ── Chat Message ──
function ChatMessage({ message }) {
  const [showCitations, setShowCitations] = useState(false);

  if (message.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-lg px-5 py-3 rounded-2xl rounded-br-md bg-violet-600/20 border border-violet-500/20 text-white/90">
          {message.content}
        </div>
      </div>
    );
  }

  if (message.role === "error") {
    return (
      <div className="flex items-center gap-2 px-4 py-3 rounded-xl bg-red-500/10 border border-red-500/20">
        <AlertCircle className="w-4 h-4 text-red-400 flex-shrink-0" />
        <p className="text-sm text-red-300">{message.content}</p>
      </div>
    );
  }

  // Assistant message
  return (
    <div className="space-y-3">
      <div className="flex items-start gap-3">
        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-violet-500 to-amber-400 flex items-center justify-center flex-shrink-0 mt-1">
          <Sparkles className="w-4 h-4 text-white" />
        </div>
        <div className="flex-1 min-w-0">
          <div className="answer-content text-white/80 text-sm leading-relaxed whitespace-pre-wrap">
            {message.content}
          </div>

          {/* Figures */}
          {message.figures?.length > 0 && (
            <div className="flex gap-3 mt-4 overflow-x-auto pb-2">
              {message.figures.map((fig, i) => (
                <img
                  key={i}
                  src={`data:image/jpeg;base64,${fig}`}
                  alt={`Figure ${i + 1}`}
                  className="h-40 rounded-lg border border-white/10 object-contain bg-white/5"
                />
              ))}
            </div>
          )}

          {/* Citations toggle */}
          {message.citations?.length > 0 && (
            <div className="mt-3">
              <button
                onClick={() => setShowCitations(!showCitations)}
                className="flex items-center gap-2 text-xs font-mono text-violet-400 hover:text-violet-300 transition-colors"
              >
                <BookOpen className="w-3.5 h-3.5" />
                {message.citations.length} source{message.citations.length > 1 ? "s" : ""}
                {showCitations ? (
                  <ChevronUp className="w-3.5 h-3.5" />
                ) : (
                  <ChevronDown className="w-3.5 h-3.5" />
                )}
              </button>

              {showCitations && (
                <div className="mt-2 space-y-2">
                  {message.citations.map((cite, i) => (
                    <div
                      key={i}
                      className="px-3 py-2.5 rounded-lg bg-white/[0.03] border border-white/5 text-xs"
                    >
                      <div className="flex items-center justify-between mb-1">
                        <span className="font-mono text-violet-400">
                          {cite.paper_name} — Page {cite.page_number}
                        </span>
                        <span className="text-white/20">
                          {(cite.relevance_score * 100).toFixed(1)}% match
                        </span>
                      </div>
                      <p className="text-white/40 line-clamp-2">
                        {cite.chunk_text}
                      </p>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ── Empty State ──
function EmptyState() {
  return (
    <div className="flex flex-col items-center justify-center h-full text-center">
      <div className="w-20 h-20 rounded-2xl bg-gradient-to-br from-violet-500/20 to-amber-400/20 flex items-center justify-center mb-6">
        <Brain className="w-10 h-10 text-violet-400" />
      </div>
      <h2 className="font-display text-3xl text-white mb-2">ScholarMind</h2>
      <p className="text-white/40 max-w-md mb-8">
        Upload scientific papers and ask questions. Get cited answers with page references, 
        powered by multi-modal RAG and Claude.
      </p>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 max-w-lg">
        {[
          "What methodology did they use?",
          "Compare results across papers",
          "Summarize the key findings",
          "What are the limitations?",
        ].map((q) => (
          <div
            key={q}
            className="px-4 py-3 rounded-xl bg-white/[0.03] border border-white/5 text-sm text-white/40 hover:bg-white/[0.05] hover:text-white/60 cursor-pointer transition-all"
          >
            {q}
          </div>
        ))}
      </div>
    </div>
  );
}
