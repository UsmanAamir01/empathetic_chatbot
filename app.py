# streamlit_app.py
# ============================================================
# Task 6 — Inference & UI (Streamlit): Greedy/Beam + Attention Heatmap
# ============================================================

import os, json, math, copy
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import urllib.request
from pathlib import Path

# --------------------- Model Auto-Download ------------------
def download_model_from_release(model_name="best_model2.pt", version="v1.0.0"):
    """
    Download model from GitHub releases if not present locally.
    This enables Streamlit Cloud deployment without Git LFS.
    """
    model_path = Path(model_name)
    
    # If model already exists, skip download
    if model_path.exists():
        return str(model_path)
    
    # GitHub release URL
    base_url = "https://github.com/UsmanAamir01/empathetic_chatbot/releases/download"
    download_url = f"{base_url}/{version}/{model_name}"
    
    st.info(f"📥 Downloading model from GitHub releases... ({model_name})")
    st.info(f"🔗 URL: {download_url}")
    st.warning("⏱️ This may take 2-3 minutes (171 MB). Please wait...")
    
    try:
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def show_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(int((downloaded / total_size) * 100), 100)
                progress_bar.progress(percent)
                status_text.text(f"Downloaded: {downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB ({percent}%)")
        
        # Download with progress
        urllib.request.urlretrieve(download_url, model_path, show_progress)
        
        progress_bar.empty()
        status_text.empty()
        st.success(f"✅ Model downloaded successfully: {model_name}")
        
        return str(model_path)
        
    except Exception as e:
        st.error(f"❌ Failed to download model: {str(e)}")
        st.error(f"Please download manually from: {download_url}")
        st.info("💡 Place the downloaded file in the project root directory and refresh the page.")
        st.stop()

# --------------------- Paths & Defaults ---------------------
IS_KAGGLE = os.path.exists("/kaggle/input")
DATA_DIR  = "/kaggle/working" if IS_KAGGLE else "."

# Auto-detect model and vocab paths
def find_best_model():
    """Find best_model.pt in current dir or checkpoints subdirectory, or download if missing"""
    candidates = [
        os.path.join(DATA_DIR, "best_model2.pt"),
        os.path.join(DATA_DIR, "best_model.pt"),
        os.path.join(DATA_DIR, "checkpoints", "best_model2.pt"),
        os.path.join(DATA_DIR, "checkpoints", "best_model.pt"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    
    # If no model found, try to download from GitHub releases
    st.warning("⚠️ Model file not found locally. Attempting to download from GitHub releases...")
    try:
        return download_model_from_release("best_model2.pt", "v1.0.0")
    except:
        # Fallback to best_model.pt
        try:
            return download_model_from_release("best_model.pt", "v1.0.0")
        except:
            return None

def find_vocab():
    """Find vocab.json in current dir or checkpoints subdirectory"""
    candidates = [
        os.path.join(DATA_DIR, "vocab.json"),
        os.path.join(DATA_DIR, "checkpoints", "vocab.json"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None

CKPT_PATH = find_best_model()
VOCAB_PATH = find_vocab()

# --------------------- Repro & Device -----------------------
torch.manual_seed(42)
np.random.seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------- Model (with Attn for UI) -------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

def make_pad_mask(ids, pad_idx): return (ids != pad_idx)
def make_attn_mask(valid_q, valid_k): return valid_q.unsqueeze(2) & valid_k.unsqueeze(1)
def make_causal_mask(L, device): return torch.tril(torch.ones(L, L, dtype=torch.bool, device=device))

def ensure_nonempty_rows(mask):
    has_true = mask.any(dim=-1, keepdim=True)
    mask = mask.clone()
    mask[:, :, 0] = mask[:, :, 0] | (~has_true.squeeze(-1))
    return mask

class MultiHeadAttentionReturn(nn.Module):
    """
    MHA that returns both (output, attn_probs).
    Used only for UI/inference to visualize attention.
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.dk = d_model // n_heads
        self.Wq, self.Wk, self.Wv = nn.Linear(d_model, d_model), nn.Linear(d_model, d_model), nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def _split(self, x):
        B,L,D = x.size()
        return x.view(B, L, self.h, self.dk).transpose(1,2)  # [B,h,L,dk]
    def _merge(self, x):
        B,h,L,dk = x.size()
        return x.transpose(1,2).contiguous().view(B, L, h*dk)

    def forward(self, q, k, v, mask=None):
        Q, K, V = self._split(self.Wq(q)), self._split(self.Wk(k)), self._split(self.Wv(v))
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.dk)  # [B,h,Lq,Lk]
        if mask is not None:
            if mask.dtype == torch.bool:
                mask = ensure_nonempty_rows(mask)
                scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))
            else:
                scores = scores + mask.unsqueeze(1)
        attn = F.softmax(scores, dim=-1)  # [B,h,Lq,Lk]
        attn = torch.nan_to_num(attn, nan=0.0)
        out = attn @ V
        out = self._merge(out)
        out = self.Wo(self.drop(out))
        return out, attn  # << return probs for visualization

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout)
        )
    def forward(self, x): return self.ff(x)

class EncoderLayerVis(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=2048, dropout=0.1):
        super().__init__()
        self.sa = MultiHeadAttentionReturn(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.n1, self.n2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
    def forward(self, x, attn_mask=None):
        sa_out, _ = self.sa(x, x, x, attn_mask)
        x = self.n1(x + sa_out)
        x = self.n2(x + self.ff(x))
        return x

class DecoderLayerVis(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=2048, dropout=0.1):
        super().__init__()
        self.msa  = MultiHeadAttentionReturn(d_model, n_heads, dropout)
        self.xattn = MultiHeadAttentionReturn(d_model, n_heads, dropout)  # Match checkpoint naming
        self.ff   = FeedForward(d_model, d_ff, dropout)
        self.n1, self.n2, self.n3 = nn.LayerNorm(d_model), nn.LayerNorm(d_model), nn.LayerNorm(d_model)
    def forward(self, x, mem, self_mask=None, mem_mask=None):
        msa_out, _        = self.msa(x, x, x, self_mask)
        x                 = self.n1(x + msa_out)
        xatt_out, attn_x  = self.xattn(x, mem, mem, mem_mask)  # cross-attn probs
        x                 = self.n2(x + xatt_out)
        x                 = self.n3(x + self.ff(x))
        return x, attn_x  # return cross-attn (per layer)

class EncoderVis(nn.Module):
    def __init__(self, d_model, n_heads, num_layers, d_ff=2048, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayerVis(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)])
    def forward(self, x, attn_mask=None):
        for lyr in self.layers:
            x = lyr(x, attn_mask)
        return x

class DecoderVis(nn.Module):
    def __init__(self, d_model, n_heads, num_layers, d_ff=2048, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayerVis(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)])
    def forward(self, x, mem, self_mask=None, mem_mask=None):
        cross_attns = []
        for lyr in self.layers:
            x, attn_x = lyr(x, mem, self_mask, mem_mask)
            cross_attns.append(attn_x)  # [B,h,Lt,Ls]
        return x, cross_attns

class TransformerUI(nn.Module):
    """
    Same shapes/names as your training model → loads the same checkpoint.
    Adds ability to return cross-attention probabilities.
    """
    def __init__(self, vocab_size, d_model=256, n_heads=2, num_layers=2, d_ff=2048, dropout=0.1, pad_idx=0):
        super().__init__()
        self.pad_idx = pad_idx
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos   = PositionalEncoding(d_model)
        self.encoder = EncoderVis(d_model, n_heads, num_layers, d_ff, dropout)
        self.decoder = DecoderVis(d_model, n_heads, num_layers, d_ff, dropout)
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        device = src.device
        B, Ls = src.size()
        Lt = tgt.size(1)
        src_valid = make_pad_mask(src, self.pad_idx)
        tgt_valid = make_pad_mask(tgt, self.pad_idx)
        enc_self  = make_attn_mask(src_valid, src_valid)
        dec_caus  = make_causal_mask(Lt, device)
        dec_self  = dec_caus.unsqueeze(0).expand(B,-1,-1) & make_attn_mask(tgt_valid, tgt_valid)
        cross     = make_attn_mask(tgt_valid, src_valid)

        src_e = self.pos(self.embed(src) * math.sqrt(self.d_model))
        tgt_e = self.pos(self.embed(tgt) * math.sqrt(self.d_model))

        mem   = self.encoder(src_e, enc_self)
        dec, cross_attn_list = self.decoder(tgt_e, mem, dec_self, cross) # list len = num_layers
        logits = self.out(dec)
        # Return last layer’s cross-attention (avg over heads outside)
        return logits, cross_attn_list[-1]  # [B,h,Lt,Ls]

# --------------------- Vocab & Template ---------------------
@st.cache_data(show_spinner=False)
def load_vocab(vocab_path):
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    # sanity
    for tok in ["<pad>","<bos>","<eos>","<unk>"]:
        assert tok in vocab, f"{tok} missing in vocab.json"
    return vocab

def tok(s: str): return str(s).strip().split()

def make_input_text(emotion, situation, customer):
    # EXACT Task 2 template:
    # "Emotion: {emotion} | Situation: {situation} | Customer: {customer} Agent:"
    return f"Emotion: {emotion} | Situation: {situation} | Customer: {customer} Agent:".strip()

def encode(tokens, vocab, unk_id):
    return [int(vocab.get(t, unk_id)) for t in tokens]

def maybe_add_emotion_token(tokens, emotion, vocab):
    # If Task 1/2 added <emotion_<emo>>, prepend it + <sep> when present
    emo_tok = f"<emotion_{emotion.lower().strip().replace(' ','_')}>"
    if emo_tok in vocab:
        tokens = [emo_tok, "<sep>"] + tokens if "<sep>" in vocab else [emo_tok] + tokens
    return tokens

def ids_to_tokens(ids, vocab):
    inv = {v:k for k,v in vocab.items()}
    out = []
    for i in ids:
        if i == vocab.get("<eos>"): break
        if i == vocab.get("<bos>"): continue
        if i == vocab.get("<pad>"): continue
        out.append(inv.get(int(i), "<unk>"))
    return out

# --------------------- Load Model ---------------------------
@st.cache_resource(show_spinner=True)
def load_model_and_config(ckpt_path, vocab_from_file):
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    
    # Prefer vocab from checkpoint (ensures ID consistency)
    if "vocab" in ckpt and isinstance(ckpt["vocab"], dict):
        vocab = ckpt["vocab"]
        vocab_source = "checkpoint"
    else:
        vocab = vocab_from_file
        vocab_source = "file"
    
    cfg  = ckpt.get("config", {})
    d_model   = cfg.get("d_model", 256)
    n_heads   = cfg.get("n_heads", 2)
    num_layers= cfg.get("num_layers", 2)
    d_ff      = cfg.get("d_ff", 2048)
    dropout   = cfg.get("dropout", 0.1)
    pad_idx   = vocab["<pad>"]

    model = TransformerUI(
        vocab_size=len(vocab),
        d_model=d_model, n_heads=n_heads, num_layers=num_layers,
        d_ff=d_ff, dropout=dropout, pad_idx=pad_idx
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt, vocab, vocab_source

# --------------------- Decoding ------------------------------
@torch.no_grad()
def greedy_decode_with_attn(model, src_ids, vocab, max_len=128):
    bos, eos = vocab["<bos>"], vocab["<eos>"]
    tgt = torch.full((1,1), bos, dtype=torch.long, device=DEVICE)
    # generate tokens
    for _ in range(max_len-1):
        logits, _ = model(src_ids, tgt)      # [1,T,V], [1,h,T,Ls]
        next_id = logits[:,-1,:].argmax(-1, keepdim=True)
        tgt = torch.cat([tgt, next_id], dim=1)
        if int(next_id.item()) == eos:
            break
    # one more forward to get full cross-attn matrix for visualization
    _, cross_attn = model(src_ids, tgt)
    # Average heads: [1,h,T,Ls] -> [T,Ls]
    attn_avg = cross_attn.mean(dim=1).squeeze(0).detach().cpu().numpy()
    return tgt.squeeze(0).tolist(), attn_avg

@torch.no_grad()
def beam_search_decode_with_attn(model, src_ids, vocab, beam_size=4, max_len=128, length_penalty=1.0):
    bos, eos = vocab["<bos>"], vocab["<eos>"]
    beams = [(torch.tensor([[bos]], device=DEVICE), 0.0, False)]  # (tgt_ids, logprob, ended)
    for _ in range(max_len-1):
        new_beams = []
        all_ended = True
        for tgt, logp, ended in beams:
            if ended:
                new_beams.append((tgt, logp, True))
                continue
            logits, _ = model(src_ids, tgt)  # [1,T,V]
            log_probs = F.log_softmax(logits[:,-1,:], dim=-1).squeeze(0)  # [V]
            topk_logp, topk_ids = torch.topk(log_probs, beam_size)
            for k in range(beam_size):
                nid = topk_ids[k].view(1,1)
                nlp = logp + float(topk_logp[k].item())
                ntgt = torch.cat([tgt, nid], dim=1)
                ended_k = (int(nid.item()) == eos)
                new_beams.append((ntgt, nlp, ended_k))
                if not ended_k:
                    all_ended = False
        # select top beams with length penalty
        def score_fn(tgt, logp, ended):
            T = tgt.size(1)
            lp = ((5 + T)**length_penalty) / ((5 + 1)**length_penalty)
            return logp / lp
        new_beams.sort(key=lambda x: score_fn(*x), reverse=True)
        beams = new_beams[:beam_size]
        if all_ended: break

    best_tgt, _, _ = beams[0]
    # one more forward to grab full cross-attn
    _, cross_attn = model(src_ids, best_tgt)
    attn_avg = cross_attn.mean(dim=1).squeeze(0).detach().cpu().numpy()
    return best_tgt.squeeze(0).tolist(), attn_avg

# --------------------- Heatmap Plot --------------------------
def plot_attention_heatmap(src_tokens, tgt_tokens, attn_matrix):
    # attn_matrix shape [T_tgt, T_src]
    fig, ax = plt.subplots(figsize=(min(12, 1 + 0.5*len(src_tokens)),
                                    min(10, 1 + 0.5*len(tgt_tokens))))
    im = ax.imshow(attn_matrix, aspect="auto")
    ax.set_xticks(range(len(src_tokens)))
    ax.set_yticks(range(len(tgt_tokens)))
    ax.set_xticklabels(src_tokens, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(tgt_tokens, fontsize=9)
    ax.set_xlabel("Source tokens")
    ax.set_ylabel("Generated tokens")
    ax.set_title("Decoder Cross-Attention (last layer, avg heads)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    st.pyplot(fig)

# --------------------- Streamlit UI --------------------------
st.set_page_config(
    page_title="Empathetic AI Chatbot",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern chatbot UI
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #4A90E2;
        --secondary-color: #50C878;
        --bg-light: #F8F9FA;
        --bg-dark: #2C3E50;
        --text-dark: #2C3E50;
        --text-light: #FFFFFF;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    /* Chat container styling */
    .chat-container {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        max-height: 600px;
        overflow-y: auto;
    }
    
    /* Message bubbles */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 18px 18px 5px 18px;
        margin: 0.5rem 0;
        margin-left: 20%;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
    }
    
    .agent-message {
        background: #F0F2F6;
        color: #2C3E50;
        padding: 1rem 1.5rem;
        border-radius: 18px 18px 18px 5px;
        margin: 0.5rem 0;
        margin-right: 20%;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
    }
    
    .message-label {
        font-weight: 600;
        font-size: 0.85rem;
        margin-bottom: 0.3rem;
        opacity: 0.8;
    }
    
    .message-text {
        font-size: 1rem;
        line-height: 1.5;
    }
    
    .emotion-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.2);
        padding: 0.2rem 0.8rem;
        border-radius: 12px;
        font-size: 0.85rem;
        margin-top: 0.5rem;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Input form styling */
    .stTextArea textarea {
        border-radius: 10px;
        border: 2px solid #E0E0E0;
        padding: 0.8rem;
    }
    
    .stTextArea textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2);
    }
    
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    
    /* Stats cards */
    .stat-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .stat-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #667eea;
    }
    
    .stat-label {
        font-size: 0.9rem;
        color: #666;
        margin-top: 0.3rem;
    }
    
    /* Attention heatmap container */
    .heatmap-container {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        margin-top: 1.5rem;
    }
    
    /* Empty state */
    .empty-state {
        text-align: center;
        padding: 3rem;
        color: #999;
    }
    
    .empty-state-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
    }
    
    /* Scrollbar styling */
    .chat-container::-webkit-scrollbar {
        width: 8px;
    }
    
    .chat-container::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    .chat-container::-webkit-scrollbar-thumb {
        background: #667eea;
        border-radius: 10px;
    }
    
    .chat-container::-webkit-scrollbar-thumb:hover {
        background: #764ba2;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>💬 Empathetic AI Chatbot</h1>
    <p>An intelligent conversational assistant that understands your emotions and provides empathetic responses</p>
</div>
""", unsafe_allow_html=True)

# Validate files exist before showing UI
if CKPT_PATH is None:
    st.error("❌ **Model file not found!**")
    st.info("📁 Please ensure the trained model file exists:")
    st.code(f"{os.path.abspath(DATA_DIR)}/best_model.pt")
    st.info("💡 **Tip**: Train the model first using task4.py")
    st.stop()

if VOCAB_PATH is None:
    st.error("❌ **Vocabulary file not found!**")
    st.info("📁 Please ensure the vocabulary file exists:")
    st.code(f"{os.path.abspath(DATA_DIR)}/vocab.json")
    st.info("💡 **Tip**: Run preprocessing first using task1.py")
    st.stop()

# ===================== SIDEBAR =====================
st.sidebar.markdown("### 🔧 Configuration")
st.sidebar.markdown("---")

# Model info
st.sidebar.markdown("#### 📦 Model Status")
st.sidebar.info(f"🖥️ Running on: {DEVICE}")

st.sidebar.markdown("---")

# Decoding settings
st.sidebar.markdown("#### 🎛️ Generation Settings")
decoding = st.sidebar.radio(
    "Decoding Strategy",
    ["Greedy", "Beam Search"],
    help="Greedy is faster, Beam Search produces higher quality responses"
)

max_len = st.sidebar.slider(
    "Max Length",
    min_value=16,
    max_value=256,
    value=128,
    step=8,
    help="Maximum number of tokens to generate"
)

if decoding == "Beam Search":
    st.sidebar.markdown("##### Beam Search Parameters")
    beam_size = st.sidebar.slider(
        "Beam Size",
        min_value=2,
        max_value=8,
        value=4,
        step=1,
        help="Number of beams (higher = better quality but slower)"
    )
    len_pen = st.sidebar.slider(
        "Length Penalty",
        min_value=0.6,
        max_value=2.0,
        value=1.0,
        step=0.1,
        help="Penalty for sequence length (>1 = longer, <1 = shorter)"
    )
else:
    beam_size = 4
    len_pen = 1.0

st.sidebar.markdown("---")

# Display options
st.sidebar.markdown("#### 👁️ Display Options")
show_attention = st.sidebar.checkbox("Show Attention Heatmap", value=True, help="Visualize cross-attention weights")
show_tokens = st.sidebar.checkbox("Show Token Details", value=False, help="Display tokenized input/output")

st.sidebar.markdown("---")

# Load resources
if not (os.path.exists(VOCAB_PATH) and os.path.exists(CKPT_PATH)):
    st.error("❌ Required files are not accessible.")
    st.info("Please check that both model and vocabulary files exist.")
    st.stop()

vocab_from_file = load_vocab(VOCAB_PATH)
try:
    model, ckpt, vocab, vocab_source = load_model_and_config(CKPT_PATH, vocab_from_file)
    
    # Model info in sidebar
    st.sidebar.markdown("#### 📊 Model Configuration")
    config = ckpt.get('config', {})
    st.sidebar.text(f"📚 Vocabulary: {len(vocab):,} tokens")
    st.sidebar.text(f"📏 Embedding size: {config.get('d_model', 256)}")
    st.sidebar.text(f"🏗️ Layers: {config.get('num_layers', 2)}")
    st.sidebar.text(f"👁️ Attention heads: {config.get('n_heads', 2)}")
    st.sidebar.caption(f"Vocab loaded from: {vocab_source}")
    
except Exception as e:
    st.error(f"❌ Failed to load the model")
    st.error(f"**Error details:** {str(e)}")
    with st.expander("🔍 View full error trace"):
        import traceback
        st.code(traceback.format_exc())
    st.stop()

pad_id, bos_id, eos_id, unk_id = vocab["<pad>"], vocab["<bos>"], vocab["<eos>"], vocab["<unk>"]

# Session state for conversation
if "history" not in st.session_state:
    st.session_state.history = []
if "show_heatmap" not in st.session_state:
    st.session_state.show_heatmap = None

# Statistics in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("#### 📈 Session Stats")
st.sidebar.metric("Total Messages", len(st.session_state.history))
if st.session_state.history:
    emotions_used = [h['emotion'] for h in st.session_state.history if h.get('emotion')]
    st.sidebar.metric("Emotions Used", len(set(emotions_used)))

# Clear conversation button
if st.sidebar.button("🗑️ Clear Conversation", use_container_width=True):
    st.session_state.history = []
    st.session_state.show_heatmap = None
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("#### ℹ️ About")
with st.sidebar.expander("How to use"):
    st.markdown("""
    1. **Enter your message** in the text area
    2. **Optionally specify emotion** (e.g., sad, happy, anxious)
    3. **Add context/situation** if relevant
    4. **Click Send** to generate response
    5. **View attention heatmap** to understand model focus
    
    **Emotions supported**: joyful, sad, angry, anxious, grateful, etc.
    """)

with st.sidebar.expander("Technical Details"):
    st.markdown("""
    - **Architecture**: Transformer Encoder-Decoder
    - **Implementation**: From scratch (PyTorch)
    - **Training**: Custom loss + optimizer
    - **Attention**: Multi-head cross-attention
    - **Template**: `emotion: {e} | situation: {s} | customer: {c} | agent:`
    """)

# ===================== MAIN CHAT AREA =====================

# Chat history display
if st.session_state.history:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for i, turn in enumerate(st.session_state.history):
        # User message
        emotion_badge = f'<span class="emotion-badge">😊 {turn["emotion"]}</span>' if turn.get("emotion") else ""
        situation_text = f'<br><small>📍 Context: {turn["situation"]}</small>' if turn.get("situation") else ""
        
        st.markdown(f"""
        <div class="user-message">
            <div class="message-label">You</div>
            <div class="message-text">{turn['user']}</div>
            {emotion_badge}{situation_text}
        </div>
        """, unsafe_allow_html=True)
        
        # Agent message
        st.markdown(f"""
        <div class="agent-message">
            <div class="message-label">🤖 Empathetic AI</div>
            <div class="message-text">{turn['bot']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
else:
    # Empty state
    st.markdown("""
    <div class="empty-state">
        <div class="empty-state-icon">💬</div>
        <h3>Start a conversation!</h3>
        <p>Share how you're feeling or what's on your mind.<br>I'm here to listen and provide empathetic support.</p>
    </div>
    """, unsafe_allow_html=True)

# Input form
st.markdown("### 💭 Your Message")

col1, col2 = st.columns([3, 1])

with col1:
    user_text = st.text_area(
        "Message",
        height=120,
        placeholder="Type your message here... (e.g., 'I'm feeling overwhelmed with work')",
        label_visibility="collapsed",
        key="user_input"
    )

with col2:
    emotion = st.selectbox(
        "Emotion (optional)",
        ["", "joyful", "sad", "angry", "anxious", "grateful", "fearful", "surprised", "disgusted", "lonely"],
        help="Select the emotion you're experiencing"
    )
    
    situation = st.text_input(
        "Context",
        placeholder="Brief context (optional)",
        help="Provide additional context if needed",
        label_visibility="collapsed"
    )

col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    submitted = st.button("🚀 Send Message", use_container_width=True, type="primary")

# Helper functions
def build_src_ids(emotion_str, situation_str, customer_str, vocab, max_len):
    x_text = make_input_text(emotion_str, situation_str, customer_str)
    x_tokens = tok(x_text)
    x_tokens = maybe_add_emotion_token(x_tokens, emotion_str, vocab) if emotion_str else x_tokens
    ids = encode(x_tokens, vocab, unk_id)[:max_len]
    return torch.tensor(ids, dtype=torch.long, device=DEVICE).unsqueeze(0), x_tokens

def run_one_turn(user_text, emotion, situation):
    src_ids, src_tokens = build_src_ids(emotion, situation, user_text, vocab, max_len)
    if decoding == "Greedy":
        tgt_ids, attn = greedy_decode_with_attn(model, src_ids, vocab, max_len=max_len)
    else:
        tgt_ids, attn = beam_search_decode_with_attn(model, src_ids, vocab, beam_size=beam_size, max_len=max_len, length_penalty=len_pen)
    gen_tokens = ids_to_tokens(tgt_ids, vocab)
    return " ".join(gen_tokens).strip(), src_tokens, gen_tokens, attn

# Generate response
if submitted:
    if not user_text.strip():
        st.warning("⚠️ Please enter a message to continue.")
    else:
        with st.spinner("🤔 Generating empathetic response..."):
            reply, src_toks, gen_toks, attn = run_one_turn(user_text.strip(), emotion.strip(), situation.strip())
        
        # Add to history
        st.session_state.history.append({
            "user": user_text.strip(),
            "emotion": emotion.strip(),
            "situation": situation.strip(),
            "bot": reply
        })
        
        # Store heatmap data
        st.session_state.show_heatmap = {
            'src_tokens': src_toks,
            'gen_tokens': gen_toks,
            'attention': attn
        }
        
        st.success("✅ Response generated successfully!")
        st.rerun()

# Show attention heatmap for latest response
if show_attention and st.session_state.show_heatmap:
    st.markdown("---")
    st.markdown("### 🎯 Attention Visualization")
    st.markdown("This heatmap shows which input tokens the model focuses on when generating each output token.")
    
    heatmap_data = st.session_state.show_heatmap
    
    # Shorten token labels for display
    def shorten(ts, cap=20):
        return [t if len(t)<=cap else (t[:cap-1]+"…") for t in ts]
    
    src_labels = shorten(heatmap_data['src_tokens'])
    tgt_labels = shorten(heatmap_data['gen_tokens'])
    attn = heatmap_data['attention']
    
    # Create two columns for heatmap and explanation
    col_heat1, col_heat2 = st.columns([3, 1])
    
    with col_heat1:
        plot_attention_heatmap(src_labels, tgt_labels, attn[:len(tgt_labels), :len(src_labels)])
    
    with col_heat2:
        st.markdown("#### 📖 Reading the Heatmap")
        st.markdown("""
        - **Brighter colors** = stronger attention
        - **Rows**: Generated tokens
        - **Columns**: Input tokens
        - The model "looks at" input tokens when generating output
        """)
        
        if st.button("❌ Hide Heatmap"):
            st.session_state.show_heatmap = None
            st.rerun()

# Show token details if enabled
if show_tokens and st.session_state.show_heatmap:
    st.markdown("---")
    st.markdown("### 🔤 Token Details")
    
    col_tok1, col_tok2 = st.columns(2)
    
    with col_tok1:
        st.markdown("#### Input Tokens")
        src_tokens = st.session_state.show_heatmap['src_tokens']
        st.code(" | ".join(src_tokens[:50]) + ("..." if len(src_tokens) > 50 else ""))
        st.caption(f"Total: {len(src_tokens)} tokens")
    
    with col_tok2:
        st.markdown("#### Output Tokens")
        gen_tokens = st.session_state.show_heatmap['gen_tokens']
        st.code(" | ".join(gen_tokens))
        st.caption(f"Total: {len(gen_tokens)} tokens")

# Footer
st.markdown("---")
col_foot1, col_foot2, col_foot3 = st.columns(3)

with col_foot1:
    st.markdown("**🏗️ Architecture**")
    st.caption("Transformer Encoder-Decoder")

with col_foot2:
    st.markdown("**📝 Implementation**")
    st.caption("Built from scratch with PyTorch")

with col_foot3:
    st.markdown("**⚡ Performance**")
    st.caption(f"{decoding} Decoding")

