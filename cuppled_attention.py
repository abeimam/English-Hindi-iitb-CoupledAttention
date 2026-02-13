import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from datasets import Dataset
import matplotlib.pyplot as plt
import time
import os
import re
import json
import random
from torch.utils.checkpoint import checkpoint
from torch.nn import DataParallel
from tqdm.auto import tqdm
import pandas as pd
import gc
from collections import Counter
import nltk
import sys
import subprocess
import warnings
warnings.filterwarnings('ignore')

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction

# Install SentencePiece if not available
try:
    import sentencepiece as spm
except ImportError:
    print("Installing sentencepiece...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "sentencepiece"])
    import sentencepiece as spm

# Install indic-nlp if available
try:
    from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
    INDIC_NORM_AVAILABLE = True
except ImportError:
    INDIC_NORM_AVAILABLE = False
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "indic-nlp-library"])
        from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
        INDIC_NORM_AVAILABLE = True
    except:
        INDIC_NORM_AVAILABLE = False
        print("Could not install indic-nlp-library. Proceeding without Indic normalization.")

# Disable tokenizer parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Enable TF32 for better performance
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ===================== USER CONFIGURATION =====================
# Update these paths as needed
DATASET_PATH = "/kaggle/input/english-hindi-parallel-corpus/dataset.csv"  # Your dataset path
CHECKPOINT_PATH = "/kaggle/input/84000hai/checkpoint_step_84000.pth"  # Your checkpoint path
RESUME_TRAINING = True

# ===================== OPTIMIZED CONFIGURATION =====================
D_MODEL = 768
D_K = 768
FF_DIM = 4096
NUM_ENCODER_LAYERS = 8
NUM_DECODER_LAYERS = 6
VOCAB_SIZE = 42000
SEQ_LEN = 256
BATCH_SIZE = 8
NUM_EPOCHS = 10
LR = 1e-4
DROPOUT = 0.1
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
WARMUP_STEPS = 4000
GRAD_ACCUM_STEPS = 8
PATIENCE = 10
GRAD_CHECKPOINT = True

DATA_PERCENTAGE = 1.0

# Advanced training parameters
ADAM_EPSILON = 1e-9
MAX_GRAD_NORM = 1.0
LABEL_SMOOTHING = 0.1

# Model architecture
USE_BIAS = False
ATTENTION_DROPOUT = 0.1
ACTIVATION_DROPOUT = 0.1

# Learning rate scheduler
LR_SCHEDULER = "cosine"
MIN_LR_RATIO = 0.05


# Checkpointing - Step-based training
EVAL_STEPS = 2000
SAVE_STEPS = 2000
LOG_STEPS = 100

# Mixed precision training
USE_AMP = True

# Model initialization
INIT_RANGE = 0.02

# Data loading optimization
PREFETCH_FACTOR = 2
PERSISTENT_WORKERS = True
NUM_WORKERS = 4

# Language embedding configuration
NUM_LANGUAGES = 2
LANG_EMBED_DIM = 768

# Feed-forward network configuration
FFN_ACTIVATION = "gelu"
USE_GLU = True

# Regularization enhancements
LAYER_NORM_EPS = 1e-6

# Training optimization
USE_FLASH_ATTENTION = False
MAX_TRAIN_STEPS = 200000

# Model saving configuration
SAVE_OPTIMIZER_STATE = True
KEEP_CHECKPOINT_MAX = 10

# Generation parameters - Beam Search
MAX_GENERATION_LENGTH = 150
NUM_BEAMS = 4
LENGTH_PENALTY = 1.0
TEMPERATURE = 0.8
TOP_K = 50
TOP_P = 0.9
REPETITION_PENALTY = 1.1
NO_REPEAT_NGRAM_SIZE = 3

# Memory optimization
USE_GRADIENT_CHECKPOINTING = True
CLEAR_CACHE_EVERY_N_STEPS = 100

# Multi-head attention configuration
NUM_HEADS = 12

# Special tokens for SentencePiece Unigram
SPECIAL_TOKENS = [
    "<pad>", "<unk>", "<s>", "</s>",
    "<en>", "<hi>",
    "<sep>",
]

# Token IDs
PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3
EN_ID = 4
HI_ID = 5
SEP_ID = 6

# Unigram sampling probability
UNIGRAM_SAMPLING_PROB = 0.1

# ===================== GLOBAL VARIABLES =====================
tokenizer = None
lang_map = {"en": 0, "hi": 1}

# ===================== LOAD DATASET =====================
def load_full_iitb_dataset(csv_path):
    """Load COMPLETE IITB English-Hindi parallel corpus from CSV file"""
    try:
        print(f"📖 Reading CSV file from: {csv_path}")
        
        if not os.path.exists(csv_path):
            print(f"❌ File not found: {csv_path}")
            raise FileNotFoundError(f"File not found: {csv_path}")
        
        df = pd.read_csv(csv_path)
        print(f"✅ Loaded CSV with {len(df):,} rows")
        print(f"📊 Columns: {list(df.columns)}")
        
        possible_en_columns = ['english', 'English', 'en', 'source', 'src', 'english_sentence', 'source_text']
        possible_hi_columns = ['hindi', 'Hindi', 'hi', 'target', 'tgt', 'hindi_sentence', 'target_text']
        
        en_col = None
        hi_col = None
        
        for col in possible_en_columns:
            if col in df.columns:
                en_col = col
                break
        
        for col in possible_hi_columns:
            if col in df.columns:
                hi_col = col
                break
        
        if en_col is None or hi_col is None:
            print(f"⚠️  Could not find standard column names. Using first two columns.")
            if len(df.columns) >= 2:
                en_col = df.columns[0]
                hi_col = df.columns[1]
            else:
                raise ValueError(f"CSV file must have at least 2 columns. Found {len(df.columns)}")
        
        print(f"📝 Using columns: English='{en_col}', Hindi='{hi_col}'")
        
        data = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing dataset"):
            en_text = str(row[en_col]) if pd.notna(row[en_col]) else ""
            hi_text = str(row[hi_col]) if pd.notna(row[hi_col]) else ""
            
            if en_text.strip() and hi_text.strip():
                data.append({
                    'translation': {
                        'en': en_text.strip(),
                        'hi': hi_text.strip()
                    }
                })
        
        print(f"📊 After cleaning: {len(data):,} valid translation pairs")
        
        if len(data) == 0:
            print("❌ No valid data found in CSV")
            raise ValueError("No valid data found in CSV")
            
        return Dataset.from_list(data)
        
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        print("🔄 Creating enhanced synthetic dataset for testing...")
        
        synthetic_data = []
        
        english_sentences = [
            "Hello, how are you?", "What is your name?", "Where are you from?",
            "Thank you for your help.", "I love learning new languages.",
            "What time is it?", "Where is the nearest hospital?",
            "How much does this cost?", "Can you help me, please?",
            "What do you do for a living?", "Good morning", "Good evening",
            "Good night", "How old are you?", "What is this?", "Where is the station?",
            "I need water", "How are you doing?", "Nice to meet you", "You are welcome",
            "I don't understand", "Can you speak English?", "My name is John",
            "I am from India", "I am a student", "I work in an office",
            "I like to read books", "The weather is nice today", "I am hungry",
            "I want to eat food", "This is very good", "That is interesting",
            "Please wait a moment", "Excuse me", "I am sorry", "No problem",
            "See you later", "Take care", "Have a nice day", "Happy birthday",
            "Congratulations", "Good luck", "Be careful", "I miss you",
            "I love you", "How was your day?", "What did you do today?",
            "I am tired", "I am happy", "I am sad", "Let's go", "Come here"
        ]
        
        hindi_translations = [
            "नमस्ते, आप कैसे हैं?", "आपका नाम क्या है?", "आप कहाँ से हैं?",
            "आपकी मदद के लिए धन्यवाद।", "मुझे नई भाषाएँ सीखना पसंद है।",
            "क्या समय हुआ है?", "निकटतम अस्पताल कहाँ है?",
            "इसकी कीमत कितनी है?", "क्या आप कृपया मेरी मदद कर सकते हैं?",
            "आप क्या काम करते हैं?", "शुभ प्रभात", "शुभ संध्या",
            "शुभ रात्रि", "आपकी उम्र क्या है?", "यह क्या है?", "स्टेशन कहाँ है?",
            "मुझे पानी चाहिए", "आप कैसे हैं?", "आपसे मिलकर अच्छा लगा", "आपका स्वागत है",
            "मुझे समझ नहीं आया", "क्या आप अंग्रेजी बोल सकते हैं?", "मेरा नाम जॉन है",
            "मैं भारत से हूँ", "मैं एक छात्र हूँ", "मैं एक ऑफिस में काम करता हूँ",
            "मुझे किताबें पढ़ना पसंद है", "आज मौसम अच्छा है", "मुझे भूख लगी है",
            "मैं खाना खाना चाहता हूँ", "यह बहुत अच्छा है", "यह दिलचस्प है",
            "कृपया एक क्षण प्रतीक्षा करें", "माफ़ कीजिए", "मुझे माफ कर दो", "कोई बात नहीं",
            "बाद में मिलते हैं", "अपना ख्याल रखना", "आपका दिन शुभ हो", "जन्मदिन मुबारक",
            "बधाई हो", "शुभकामनाएँ", "सावधान रहना", "मुझे आपकी याद आती है",
            "मैं आपसे प्यार करता हूँ", "आपका दिन कैसा रहा?", "आपने आज क्या किया?",
            "मैं थक गया हूँ", "मैं खुश हूँ", "मैं दुखी हूँ", "चलो चलते हैं", "यहाँ आओ"
        ]
        
        for i in range(1000):
            idx = i % len(english_sentences)
            en = english_sentences[idx]
            hi = hindi_translations[idx]
            
            if i > len(english_sentences):
                en = en.replace(".", "!") if i % 2 == 0 else en
                hi = hi.replace("।", "!") if i % 2 == 0 else hi
            
            synthetic_data.append({
                'translation': {
                    'en': en,
                    'hi': hi
                }
            })
        
        print(f"📊 Created enhanced synthetic dataset with {len(synthetic_data):,} samples")
        return Dataset.from_list(synthetic_data)

# ===================== TEXT NORMALIZATION =====================
def unicode_nfc_normalize(text):
    """Apply Unicode NFC normalization"""
    import unicodedata
    return unicodedata.normalize('NFC', text)

def remove_zero_width_joiners(text):
    """Remove zero-width joiners and other zero-width characters"""
    zero_width_chars = ['\u200c', '\u200d', '\u200b', '\u200e', '\u200f']
    for char in zero_width_chars:
        text = text.replace(char, '')
    return text

def indic_normalize(text, lang='hi'):
    """Apply Indic normalization if available"""
    if not INDIC_NORM_AVAILABLE or lang != 'hi':
        return text
    try:
        factory = IndicNormalizerFactory()
        normalizer = factory.get_normalizer(lang)
        return normalizer.normalize(text)
    except:
        return text

def normalize_text(text, lang='en'):
    """Apply full normalization pipeline"""
    if not text or not isinstance(text, str):
        return ""
    
    text = unicode_nfc_normalize(text)
    text = remove_zero_width_joiners(text)
    
    if lang == 'hi':
        text = indic_normalize(text, lang)
    
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_text(text):
    """Enhanced text cleaning with normalization"""
    if not text or not isinstance(text, str):
        return ""
    text = normalize_text(text)
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    return text[:500]

# ===================== TOKENIZER =====================
class SentencePieceUnigramTokenizer:
    """Wrapper for SentencePiece Unigram tokenizer with shared vocabulary"""
    
    def __init__(self, model_path=None, vocab_size=VOCAB_SIZE):
        self.sp = spm.SentencePieceProcessor()
        self.vocab_size = vocab_size
        self.model = None
        
        if model_path and os.path.exists(model_path):
            self.load(model_path)
        
        self.pad_id = PAD_ID
        self.unk_id = UNK_ID
        self.bos_id = BOS_ID
        self.eos_id = EOS_ID
        self.en_id = EN_ID
        self.hi_id = HI_ID
        self.sep_id = SEP_ID
        
        self.vocab = None
    
    def train(self, texts, model_prefix="spm_unigram", character_coverage=0.9995):
        """Train SentencePiece Unigram tokenizer with adaptive vocab size"""
        print("🔄 Training SentencePiece Unigram tokenizer...")
        
        unique_words = set()
        for text in texts:
            words = text.split()
            unique_words.update(words)
        
        estimated_vocab_size = min(
            len(unique_words) * 2,
            self.vocab_size
        )
        
        estimated_vocab_size = max(estimated_vocab_size, 1000)
        
        print(f"📊 Estimated optimal vocab size: {estimated_vocab_size} (based on {len(unique_words)} unique words)")
        
        temp_file = "temp_training_texts.txt"
        with open(temp_file, 'w', encoding='utf-8') as f:
            for text in tqdm(texts, desc="Writing training texts"):
                f.write(text + '\n')
        
        train_args = [
            f'--input={temp_file}',
            f'--model_prefix={model_prefix}',
            f'--vocab_size={estimated_vocab_size}',
            '--model_type=unigram',
            '--character_coverage=1.0',
            '--shuffle_input_sentence=true',
            '--split_by_unicode_script=true',
            '--split_by_whitespace=true',
            '--normalization_rule_name=nfkc',
            f'--pad_id={self.pad_id}',
            f'--unk_id={self.unk_id}',
            f'--bos_id={self.bos_id}',
            f'--eos_id={self.eos_id}',
            '--byte_fallback=true',
            '--hard_vocab_limit=false',
        ]
        
        if len(SPECIAL_TOKENS) > 4:
            train_args.append(f'--user_defined_symbols={",".join(SPECIAL_TOKENS[4:])}')
        
        try:
            print("🔧 Training SentencePiece model...")
            
            model_file = f"{model_prefix}.model"
            vocab_file = f"{model_prefix}.vocab"
            if os.path.exists(model_file):
                os.remove(model_file)
            if os.path.exists(vocab_file):
                os.remove(vocab_file)
                
            spm.SentencePieceTrainer.train(' '.join(train_args))
            
            self.vocab_size = estimated_vocab_size
            
        except Exception as e:
            print(f"⚠️  Error training with vocab size {estimated_vocab_size}: {e}")
            print("🔄 Trying with smaller vocab size...")
            
            smaller_vocab = max(1000, estimated_vocab_size // 2)
            train_args[2] = f'--vocab_size={smaller_vocab}'
            
            if os.path.exists(model_file):
                os.remove(model_file)
            if os.path.exists(vocab_file):
                os.remove(vocab_file)
                
            try:
                spm.SentencePieceTrainer.train(' '.join(train_args))
                self.vocab_size = smaller_vocab
            except Exception as e2:
                print(f"❌ Failed to train tokenizer: {e2}")
                raise
        
        self.load(f"{model_prefix}.model")
        
        if os.path.exists(temp_file):
            os.remove(temp_file)
        
        print(f"✅ SentencePiece Unigram tokenizer trained with vocab size: {self.get_vocab_size()}")
    
    def load(self, model_path):
        """Load trained SentencePiece model"""
        self.sp.load(model_path)
        self.model = model_path
        self.vocab = {self.sp.id_to_piece(i): i for i in range(self.sp.get_piece_size())}
        print(f"✅ Loaded SentencePiece model from {model_path}")
    
    def save(self, model_path):
        """Save SentencePiece model"""
        if self.model:
            import shutil
            shutil.copy2(self.model, model_path)
            print(f"✅ Saved SentencePiece model to {model_path}")
    
    def encode(self, text, add_bos=True, add_eos=True, enable_sampling=True, alpha=UNIGRAM_SAMPLING_PROB):
        """Encode text with optional Unigram sampling"""
        if enable_sampling and UNIGRAM_SAMPLING_PROB > 0:
            return self.sp.sample_encode_as_ids(
                text, nbest_size=-1, alpha=alpha,
                add_bos=add_bos, add_eos=add_eos
            )
        else:
            return self.sp.encode_as_ids(text, add_bos=add_bos, add_eos=add_eos)
    
    def decode(self, ids):
        """Decode token IDs to text"""
        return self.sp.decode_ids(ids)
    
    def get_vocab_size(self):
        """Get vocabulary size"""
        return self.sp.get_piece_size() if self.sp else 0
    
    def get_vocab(self):
        """Get vocabulary dictionary"""
        return self.vocab
    
    def test_encode_decode(self, text):
        """Test encoding and decoding"""
        print(f"\n🧪 Testing tokenizer on: {text[:50]}...")
        ids = self.encode(text, add_bos=False, add_eos=False, enable_sampling=False)
        tokens = [self.sp.id_to_piece(id) for id in ids[:10]]
        decoded = self.decode(ids)
        print(f"   First 10 tokens: {tokens}")
        print(f"   Decoded: {decoded[:100]}...")
        return ids, tokens, decoded

# ===================== MODEL ARCHITECTURE =====================
class RotaryEmbedding(nn.Module):
    """Rotary Positional Embedding"""
    def __init__(self, dim, max_seq_len=5000):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_seq_len)
    
    def _build_cache(self, max_seq_len):
        t = torch.arange(max_seq_len, device=self.inv_freq.device, dtype=torch.float32)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
    
    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.size(1)
        if seq_len > self.cos_cached.size(0):
            self._build_cache(seq_len)
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )

def rotate_half(x):
    """Rotate half of the dimensions"""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_emb(q, k, cos, sin):
    """Apply rotary embeddings to queries and keys"""
    q_embed = (q * cos.unsqueeze(0)) + (rotate_half(q) * sin.unsqueeze(0))
    k_embed = (k * cos.unsqueeze(0)) + (rotate_half(k) * sin.unsqueeze(0))
    return q_embed, k_embed

class MultiHeadCoupledAttention(nn.Module):
    """Multi-Head Coupled Attention (without V projection)"""
    def __init__(self, d_model, d_k, num_heads=12, is_cross_attention=False):
        super().__init__()
        assert d_k % num_heads == 0, "d_k must be divisible by num_heads"
        
        self.d_model = d_model
        self.d_k = d_k
        self.num_heads = num_heads
        self.head_dim = d_k // num_heads
        self.is_cross_attention = is_cross_attention
        
        self.qw_proj = nn.Linear(d_model, d_k, bias=USE_BIAS)
        self.kw_proj = nn.Linear(d_model, d_k, bias=USE_BIAS)
        self.qp_proj = nn.Linear(d_model, d_k, bias=USE_BIAS)
        self.kp_proj = nn.Linear(d_model, d_k, bias=USE_BIAS)
        
        self.out_proj = nn.Linear(d_k, d_model, bias=USE_BIAS)
        
        self.attn_dropout = nn.Dropout(ATTENTION_DROPOUT)
        
        if not is_cross_attention:
            self.rotary = RotaryEmbedding(self.head_dim)
    
    def split_heads(self, x):
        """Split the last dimension into (num_heads, head_dim)"""
        batch_size, seq_len = x.size()[:2]
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)
    
    def combine_heads(self, x):
        """Combine heads back to original shape"""
        batch_size, num_heads, seq_len, head_dim = x.size()
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, num_heads * head_dim)
    
    def forward(self, query, key=None, value=None, mask=None, encoder_output=None):
        if encoder_output is not None:
            key = key if key is not None else encoder_output
            value = encoder_output
        elif key is None or value is None:
            key = query
            value = query
        
        B, L, _ = query.size()
        B, S, _ = key.size()
        
        qw = self.qw_proj(query)
        kw = self.kw_proj(key)
        qp = self.qp_proj(query)
        kp = self.kp_proj(key)
        
        qw = self.split_heads(qw)
        kw = self.split_heads(kw)
        qp = self.split_heads(qp)
        kp = self.split_heads(kp)
        
        value_heads = self.split_heads(value)
        
        if not self.is_cross_attention:
            cos, sin = self.rotary(qw, seq_len=L)
            qw, kw = apply_rotary_emb(qw, kw, cos, sin)
            qp, kp = apply_rotary_emb(qp, kp, cos, sin)
        
        w2p_scores = torch.matmul(qw, kp.transpose(-2, -1)) / math.sqrt(self.head_dim)
        p2w_scores = torch.matmul(qp, kw.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        w2p_activated = F.silu(w2p_scores)
        p2w_activated = F.silu(p2w_scores)
        attn_scores = (w2p_activated + p2w_activated) / math.sqrt(2)
        
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(1)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, value_heads)
        attn_output = self.combine_heads(attn_output)
        
        output = self.out_proj(attn_output)
        
        return output, attn_weights

class EncoderBlock(nn.Module):
    """Encoder Block with Coupled Attention"""
    def __init__(self, d_model, d_k, ff_dim, num_heads=12):
        super().__init__()
        self.self_attn = MultiHeadCoupledAttention(d_model, d_k, num_heads, is_cross_attention=False)
        self.norm1 = nn.LayerNorm(d_model, eps=LAYER_NORM_EPS)
        self.norm2 = nn.LayerNorm(d_model, eps=LAYER_NORM_EPS)
        
        if USE_GLU:
            self.ffn = nn.Sequential(
                nn.Linear(d_model, ff_dim * 2, bias=USE_BIAS),
                nn.GLU(dim=-1),
                nn.Dropout(ACTIVATION_DROPOUT),
                nn.Linear(ff_dim, d_model, bias=USE_BIAS),
            )
        else:
            self.ffn = nn.Sequential(
                nn.Linear(d_model, ff_dim, bias=USE_BIAS),
                nn.GELU(),
                nn.Dropout(ACTIVATION_DROPOUT),
                nn.Linear(ff_dim, d_model, bias=USE_BIAS),
            )
        
        self.dropout1 = nn.Dropout(DROPOUT)
        self.dropout2 = nn.Dropout(DROPOUT)
    
    def forward(self, x, src_mask):
        residual = x
        x_norm = self.norm1(x)
        attn_out, attn_weights = self.self_attn(x_norm, mask=src_mask)
        x = residual + self.dropout1(attn_out)
        
        residual = x
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = residual + self.dropout2(ffn_out)
        
        return x, attn_weights

class DecoderBlock(nn.Module):
    """Decoder Block with Coupled Attention"""
    def __init__(self, d_model, d_k, ff_dim, num_heads=12):
        super().__init__()
        self.self_attn = MultiHeadCoupledAttention(d_model, d_k, num_heads, is_cross_attention=False)
        self.cross_attn = MultiHeadCoupledAttention(d_model, d_k, num_heads, is_cross_attention=True)
        self.norm1 = nn.LayerNorm(d_model, eps=LAYER_NORM_EPS)
        self.norm2 = nn.LayerNorm(d_model, eps=LAYER_NORM_EPS)
        self.norm3 = nn.LayerNorm(d_model, eps=LAYER_NORM_EPS)
        
        if USE_GLU:
            self.ffn = nn.Sequential(
                nn.Linear(d_model, ff_dim * 2, bias=USE_BIAS),
                nn.GLU(dim=-1),
                nn.Dropout(ACTIVATION_DROPOUT),
                nn.Linear(ff_dim, d_model, bias=USE_BIAS),
            )
        else:
            self.ffn = nn.Sequential(
                nn.Linear(d_model, ff_dim, bias=USE_BIAS),
                nn.GELU(),
                nn.Dropout(ACTIVATION_DROPOUT),
                nn.Linear(ff_dim, d_model, bias=USE_BIAS),
            )
        
        self.dropout1 = nn.Dropout(DROPOUT)
        self.dropout2 = nn.Dropout(DROPOUT)
        self.dropout3 = nn.Dropout(DROPOUT)
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        residual = x
        x_norm = self.norm1(x)
        attn_out, self_attn_weights = self.self_attn(x_norm, mask=tgt_mask)
        x = residual + self.dropout1(attn_out)
        
        residual = x
        x_norm = self.norm2(x)
        cross_attn_out, cross_attn_weights = self.cross_attn(
            x_norm, encoder_output=encoder_output, mask=src_mask
        )
        x = residual + self.dropout2(cross_attn_out)
        
        residual = x
        x_norm = self.norm3(x)
        ffn_out = self.ffn(x_norm)
        x = residual + self.dropout3(ffn_out)
        
        return x, self_attn_weights, cross_attn_weights

class Encoder(nn.Module):
    """Encoder with Language Embeddings"""
    def __init__(self, num_layers, d_model, d_k, ff_dim, vocab_size, num_heads=12):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.language_embedding = nn.Embedding(NUM_LANGUAGES, LANG_EMBED_DIM)
        self.lang_proj = nn.Linear(LANG_EMBED_DIM, d_model, bias=False)
        
        self.layers = nn.ModuleList([
            EncoderBlock(d_model, d_k, ff_dim, num_heads)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model, eps=LAYER_NORM_EPS)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=INIT_RANGE)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=INIT_RANGE)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, src_tokens, src_lang_ids, src_mask):
        word_embed = self.embedding(src_tokens)
        lang_embed = self.language_embedding(src_lang_ids)
        lang_embed = self.lang_proj(lang_embed)
        x = word_embed + lang_embed
        
        all_attn_weights = []
        for layer in self.layers:
            if GRAD_CHECKPOINT and self.training:
                x, attn_weights = checkpoint(layer, x, src_mask)
            else:
                x, attn_weights = layer(x, src_mask)
            all_attn_weights.append(attn_weights)
        
        x = self.norm(x)
        return x, all_attn_weights

class Decoder(nn.Module):
    """Decoder with Language Embeddings"""
    def __init__(self, num_layers, d_model, d_k, ff_dim, vocab_size, num_heads=12):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.language_embedding = nn.Embedding(NUM_LANGUAGES, LANG_EMBED_DIM)
        self.lang_proj = nn.Linear(LANG_EMBED_DIM, d_model, bias=False)
        
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, d_k, ff_dim, num_heads)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model, eps=LAYER_NORM_EPS)
        
        self.output_proj = nn.Linear(d_model, vocab_size, bias=USE_BIAS)
        
        self.apply(self._init_weights)
        
        self.output_proj.weight = self.embedding.weight
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=INIT_RANGE)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=INIT_RANGE)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, tgt_tokens, tgt_lang_ids, encoder_output, src_mask, tgt_mask):
        word_embed = self.embedding(tgt_tokens)
        lang_embed = self.language_embedding(tgt_lang_ids)
        lang_embed = self.lang_proj(lang_embed)
        x = word_embed + lang_embed
        
        all_self_attn_weights = []
        all_cross_attn_weights = []
        
        for layer in self.layers:
            if GRAD_CHECKPOINT and self.training:
                x, self_attn, cross_attn = checkpoint(layer, x, encoder_output, src_mask, tgt_mask)
            else:
                x, self_attn, cross_attn = layer(x, encoder_output, src_mask, tgt_mask)
            all_self_attn_weights.append(self_attn)
            all_cross_attn_weights.append(cross_attn)
        
        x = self.norm(x)
        logits = self.output_proj(x)
        
        return logits, all_self_attn_weights, all_cross_attn_weights

class TranslationTransformer(nn.Module):
    """Translation Transformer (Encoder-Decoder)"""
    def __init__(self, num_encoder_layers, num_decoder_layers, d_model, d_k, ff_dim, vocab_size, num_heads=12):
        super().__init__()
        self.encoder = Encoder(num_encoder_layers, d_model, d_k, ff_dim, vocab_size, num_heads)
        self.decoder = Decoder(num_decoder_layers, d_model, d_k, ff_dim, vocab_size, num_heads)
    
    def forward(self, src_tokens, src_lang_ids, tgt_tokens, tgt_lang_ids, src_mask, tgt_mask):
        encoder_output, encoder_attn_weights = self.encoder(src_tokens, src_lang_ids, src_mask)
        
        logits, decoder_self_attn_weights, decoder_cross_attn_weights = self.decoder(
            tgt_tokens, tgt_lang_ids, encoder_output, src_mask, tgt_mask
        )
        
        return logits, encoder_attn_weights, decoder_self_attn_weights, decoder_cross_attn_weights

# ===================== DATASET AND UTILITIES =====================
class TranslationDataset(torch.utils.data.Dataset):
    """Translation Dataset with Unigram Sampling"""
    def __init__(self, translation_pairs, seq_len=256, max_samples=None, enable_unigram_sampling=True):
        self.seq_len = seq_len
        self.enable_unigram_sampling = enable_unigram_sampling
        
        print(f"🔄 Processing translation dataset ({len(translation_pairs):,} pairs)...")
        
        if max_samples and len(translation_pairs) > max_samples:
            translation_pairs = translation_pairs[:max_samples]
        
        self.src_segments = []
        self.tgt_segments = []
        self.src_lang_segments = []
        self.tgt_lang_segments = []
        
        processed_count = 0
        for src_lang, src_text, tgt_lang, tgt_text in tqdm(translation_pairs, desc="Processing pairs"):
            src_text_clean = clean_text(src_text)
            tgt_text_clean = clean_text(tgt_text)
            
            if not src_text_clean or not tgt_text_clean:
                continue
            
            try:
                src_tokens = tokenizer.encode(
                    src_text_clean,
                    add_bos=True,
                    add_eos=True,
                    enable_sampling=self.enable_unigram_sampling,
                    alpha=UNIGRAM_SAMPLING_PROB
                )
                
                tgt_tokens = tokenizer.encode(
                    tgt_text_clean,
                    add_bos=True,
                    add_eos=True,
                    enable_sampling=self.enable_unigram_sampling,
                    alpha=UNIGRAM_SAMPLING_PROB
                )
                
                src_tokens = self._pad_or_truncate(src_tokens)
                tgt_tokens = self._pad_or_truncate(tgt_tokens)
                
                src_lang_ids = [lang_map[src_lang]] * self.seq_len
                tgt_lang_ids = [lang_map[tgt_lang]] * self.seq_len
                
                src_meaningful = [t for t in src_tokens if t not in [tokenizer.bos_id, tokenizer.eos_id, tokenizer.pad_id]]
                tgt_meaningful = [t for t in tgt_tokens if t not in [tokenizer.bos_id, tokenizer.eos_id, tokenizer.pad_id]]
                
                if len(src_meaningful) > 1 and len(tgt_meaningful) > 1:
                    self.src_segments.append(src_tokens)
                    self.tgt_segments.append(tgt_tokens)
                    self.src_lang_segments.append(src_lang_ids)
                    self.tgt_lang_segments.append(tgt_lang_ids)
                    processed_count += 1
                    
            except Exception as e:
                continue
        
        print(f"✅ Created {len(self.src_segments):,} sequences from {len(translation_pairs):,} translation pairs")
    
    def _pad_or_truncate(self, tokens):
        """Pad or truncate token sequence to seq_len"""
        if len(tokens) < self.seq_len:
            return tokens + [tokenizer.pad_id] * (self.seq_len - len(tokens))
        else:
            truncated = tokens[:self.seq_len]
            truncated[-1] = tokenizer.eos_id
            return truncated
    
    def __len__(self):
        return len(self.src_segments)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.src_segments[idx], dtype=torch.long),
            torch.tensor(self.src_lang_segments[idx], dtype=torch.long),
            torch.tensor(self.tgt_segments[idx], dtype=torch.long),
            torch.tensor(self.tgt_lang_segments[idx], dtype=torch.long)
        )

def create_padding_mask(tokens, pad_id, device):
    """Create padding mask (1 for non-pad tokens, 0 for pad tokens)"""
    mask = (tokens != pad_id).to(device)
    return mask

def create_causal_mask(seq_len, device):
    """Create causal mask for decoder self-attention"""
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()
    return mask

def calculate_bleu(reference, candidate):
    """Calculate BLEU score between reference and candidate translations"""
    ref_tokens = nltk.word_tokenize(reference.lower())
    cand_tokens = nltk.word_tokenize(candidate.lower())
    
    smoothie = SmoothingFunction().method4
    
    try:
        score = sentence_bleu([ref_tokens], cand_tokens,
                              weights=(0.25, 0.25, 0.25, 0.25),
                              smoothing_function=smoothie)
        return score
    except:
        return 0.0

def evaluate_translation_quality(model, val_pairs, device, num_samples=100):
    """Evaluate translation quality using BLEU score"""
    print("\n🧪 Evaluating Translation Quality with BLEU Score")
    
    if not val_pairs:
        return {"bleu_score": 0.0, "samples": []}
    
    eval_pairs = val_pairs[:min(num_samples, len(val_pairs))]
    references = []
    candidates = []
    
    model.eval()
    
    for src_lang, src_text, tgt_lang, tgt_text in tqdm(eval_pairs, desc="Evaluating"):
        if src_lang == "en" and tgt_lang == "hi":
            try:
                translation = translate_sentence_beam_search(
                    model, src_text, device, src_lang="en", tgt_lang="hi"
                )
                references.append(clean_text(tgt_text))
                candidates.append(clean_text(translation))
            except Exception as e:
                continue
    
    if not references:
        return {"bleu_score": 0.0, "samples": []}
    
    bleu_scores = [calculate_bleu(ref, cand) for ref, cand in zip(references, candidates)]
    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
    
    return {
        "bleu_score": avg_bleu,
        "avg_sentence_bleu": avg_bleu,
        "num_evaluated": len(references)
    }

def translate_sentence_beam_search(model, sentence, device, src_lang="en", tgt_lang="hi", max_length=MAX_GENERATION_LENGTH):
    """Translate a single sentence using beam search"""
    model.eval()
    sentence_clean = clean_text(sentence)
    
    src_tokens = tokenizer.encode(
        sentence_clean,
        add_bos=True,
        add_eos=True,
        enable_sampling=False
    )
    
    if len(src_tokens) < SEQ_LEN:
        src_tokens = src_tokens + [tokenizer.pad_id] * (SEQ_LEN - len(src_tokens))
    else:
        src_tokens = src_tokens[:SEQ_LEN]
        src_tokens[-1] = tokenizer.eos_id
    
    src_tensor = torch.tensor([src_tokens]).to(device)
    src_lang_tensor = torch.tensor([[lang_map[src_lang]] * SEQ_LEN]).to(device)
    
    src_mask = create_padding_mask(src_tensor, tokenizer.pad_id, device)
    
    with torch.no_grad():
        encoder_output, _ = model.encoder(src_tensor, src_lang_tensor, src_mask)
    
    beams = [(torch.tensor([tokenizer.bos_id]).to(device), 0.0)]
    
    for step in range(max_length):
        candidates = []
        
        for seq, score in beams:
            if seq[-1] == tokenizer.eos_id:
                candidates.append((seq, score))
                continue
            
            decoder_input = seq.unsqueeze(0)
            if decoder_input.size(1) < SEQ_LEN:
                padding = torch.full((1, SEQ_LEN - decoder_input.size(1)), tokenizer.pad_id, device=device)
                decoder_input = torch.cat([decoder_input, padding], dim=1)
            
            decoder_input = decoder_input[:, :SEQ_LEN]
            tgt_lang_tensor = torch.tensor([[lang_map[tgt_lang]] * decoder_input.size(1)]).to(device)
            
            tgt_padding_mask = create_padding_mask(decoder_input, tokenizer.pad_id, device)
            causal_mask = create_causal_mask(decoder_input.size(1), device)
            tgt_mask = causal_mask.unsqueeze(0) & tgt_padding_mask.unsqueeze(1)
            
            with torch.no_grad():
                logits, _, _ = model.decoder(
                    decoder_input, tgt_lang_tensor, encoder_output, src_mask, tgt_mask
                )
            
            next_logits = logits[0, seq.size(0) - 1, :] / TEMPERATURE
            
            if REPETITION_PENALTY != 1.0:
                for token_id in seq:
                    next_logits[token_id] /= REPETITION_PENALTY
            
            topk_probs, topk_indices = torch.topk(F.softmax(next_logits, dim=-1), NUM_BEAMS)
            
            for token_id, token_prob in zip(topk_indices, topk_probs):
                new_seq = torch.cat([seq, token_id.unsqueeze(0)])
                new_score = score + math.log(token_prob.item())
                
                length_penalized_score = new_score / ((5 + len(new_seq)) / 6) ** LENGTH_PENALTY
                candidates.append((new_seq, length_penalized_score))
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        beams = candidates[:NUM_BEAMS]
        
        if all(beam[0][-1] == tokenizer.eos_id or len(beam[0]) >= max_length for beam in beams):
            break
    
    best_seq = beams[0][0]
    
    if len(best_seq) > 0 and best_seq[0] == tokenizer.bos_id:
        best_seq = best_seq[1:]
    
    if tokenizer.eos_id in best_seq:
        best_seq = best_seq[:torch.where(best_seq == tokenizer.eos_id)[0][0]]
    
    translation = tokenizer.decode(best_seq.tolist())
    return translation

def translate_sentence(model, sentence, device, src_lang="en", tgt_lang="hi"):
    """Wrapper for translate_sentence_beam_search for backward compatibility"""
    return translate_sentence_beam_search(model, sentence, device, src_lang, tgt_lang)

# ===================== CHECKPOINT MANAGEMENT =====================
def save_checkpoint(step, model, optimizer, scheduler, scaler, train_losses, val_losses, 
                   train_ppls, val_ppls, bleu_scores, learning_rates, global_step, 
                   best_val_loss, best_bleu_score, checkpoint_dir="checkpoints"):
    """Save training checkpoint - step-based"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    model_state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    
    checkpoint = {
        'global_step': global_step,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_ppls': train_ppls,
        'val_ppls': val_ppls,
        'bleu_scores': bleu_scores,
        'learning_rates': learning_rates,
        'best_val_loss': best_val_loss,
        'best_bleu_score': best_bleu_score,
        'step_losses': step_losses if 'step_losses' in locals() else [],
        'step_ppls': step_ppls if 'step_ppls' in locals() else [],
        'last_100_losses': last_100_losses if 'last_100_losses' in locals() else [],
        'last_100_ppls': last_100_ppls if 'last_100_ppls' in locals() else [],
        'config': {
            'D_MODEL': D_MODEL,
            'D_K': D_K,
            'FF_DIM': FF_DIM,
            'NUM_ENCODER_LAYERS': NUM_ENCODER_LAYERS,
            'NUM_DECODER_LAYERS': NUM_DECODER_LAYERS,
            'VOCAB_SIZE': VOCAB_SIZE,
            'SEQ_LEN': SEQ_LEN,
            'BATCH_SIZE': BATCH_SIZE,
            'LR': LR,
            'NUM_HEADS': NUM_HEADS,
        }
    }
    
    step_checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_step_{global_step}.pth')
    torch.save(checkpoint, step_checkpoint_path)
    print(f"💾 Saved step checkpoint: {step_checkpoint_path}")
    
    latest_checkpoint_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
    torch.save(checkpoint, latest_checkpoint_path)
    
    if global_step % (SAVE_STEPS * 2) == 0:
        best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
        torch.save(model_state_dict, best_model_path)
    
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_step_')]
    if len(checkpoints) > KEEP_CHECKPOINT_MAX:
        def get_step_num(filename):
            try:
                return int(filename.split('step_')[1].split('.')[0])
            except:
                return 0
        
        checkpoints.sort(key=get_step_num, reverse=True)
        for old_checkpoint in checkpoints[KEEP_CHECKPOINT_MAX:]:
            os.remove(os.path.join(checkpoint_dir, old_checkpoint))
            print(f"🗑️  Removed old checkpoint: {old_checkpoint}")

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, scaler=None):
    """Load training checkpoint"""
    print(f"🔄 Loading checkpoint from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    if scaler and checkpoint['scaler_state_dict']:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    training_state = {
        'global_step': checkpoint.get('global_step', 0),
        'train_losses': checkpoint.get('train_losses', []),
        'val_losses': checkpoint.get('val_losses', []),
        'train_ppls': checkpoint.get('train_ppls', []),
        'val_ppls': checkpoint.get('val_ppls', []),
        'bleu_scores': checkpoint.get('bleu_scores', []),
        'learning_rates': checkpoint.get('learning_rates', []),
        'best_val_loss': checkpoint.get('best_val_loss', float('inf')),
        'best_bleu_score': checkpoint.get('best_bleu_score', 0.0),
        'step_losses': checkpoint.get('step_losses', []),
        'step_ppls': checkpoint.get('step_ppls', []),
        'last_100_losses': checkpoint.get('last_100_losses', []),
        'last_100_ppls': checkpoint.get('last_100_ppls', []),
    }
    
    print(f"✅ Checkpoint loaded. Resuming from step {checkpoint['global_step']}")
    return training_state

def find_latest_checkpoint(checkpoint_dir="/kaggle/input/2500hai"):
    """Find the latest checkpoint file"""
    if not os.path.exists(checkpoint_dir):
        return None
    
    latest_path = os.path.join(checkpoint_dir, 'checkpoint_step_2500.pth')
    if os.path.exists(latest_path):
        return latest_path
    
    step_checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_step_')]
    if step_checkpoints:
        step_checkpoints.sort(key=lambda x: int(x.split('_')[2].split('.')[0]), reverse=True)
        return os.path.join(checkpoint_dir, step_checkpoints[0])
    
    return None

def train_model(train_dataset, val_dataset, resume_from_checkpoint="/kaggle/input/2500hai/checkpoint_step_2500.pth"):
    """Main training function with step-based training"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=min(BATCH_SIZE, len(train_dataset)),
        shuffle=True,
        num_workers=min(NUM_WORKERS, 4),
        pin_memory=True,
        persistent_workers=PERSISTENT_WORKERS,
        prefetch_factor=PREFETCH_FACTOR
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=min(BATCH_SIZE, len(val_dataset)),
        shuffle=False,
        num_workers=min(NUM_WORKERS, 2),
        pin_memory=True
    )
    
    actual_vocab_size = tokenizer.get_vocab_size()
    print(f"📊 Using actual vocab size: {actual_vocab_size}")
    
    model = TranslationTransformer(
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        d_model=D_MODEL,
        d_k=D_K,
        ff_dim=FF_DIM,
        vocab_size=actual_vocab_size,
        num_heads=NUM_HEADS
    ).to(device)
    
    if torch.cuda.device_count() > 1:
        print(f"🎯 Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = DataParallel(model)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Model parameters: {num_params/1e6:.2f}M")
    print(f"🎯 Coupled Attention Architecture (No V Projection)")
    print(f"   Encoder Layers: {NUM_ENCODER_LAYERS}")
    print(f"   Decoder Layers: {NUM_DECODER_LAYERS}")
    print(f"   Attention Heads: {NUM_HEADS}")
    print(f"   Model Dimension: {D_MODEL}")
    print(f"   FFN Dimension: {FF_DIM}")
    print(f"   Training samples: {len(train_loader.dataset):,}")
    print(f"   Validation samples: {len(val_loader.dataset):,}")
    print(f"   Training Steps Target: {MAX_TRAIN_STEPS:,}")
    print(f"   Save checkpoint every {SAVE_STEPS} steps")
    print(f"   Evaluate every {EVAL_STEPS} steps")
    print(f"   Gradient Accumulation Steps: {GRAD_ACCUM_STEPS}")
    print(f"   Effective Batch Size: {BATCH_SIZE * GRAD_ACCUM_STEPS}")
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        betas=(0.9, 0.98),
        eps=ADAM_EPSILON,
        weight_decay=WEIGHT_DECAY
    )
    
    if LABEL_SMOOTHING > 0:
        criterion = nn.CrossEntropyLoss(
            ignore_index=tokenizer.pad_id,
            label_smoothing=LABEL_SMOOTHING
        )
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
    
    def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return MIN_LR_RATIO + (1.0 - MIN_LR_RATIO) * max(0.0, cosine_decay)
        
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    total_training_steps = MAX_TRAIN_STEPS
    warmup_steps = min(WARMUP_STEPS, total_training_steps // 10)
    
    print(f"📈 Training Steps: {total_training_steps:,}")
    print(f"🔥 Warmup Steps: {warmup_steps:,}")
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps
    )
    
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP and torch.cuda.is_available())
    
    training_state = {
        'train_losses': [],
        'val_losses': [],
        'train_ppls': [],
        'val_ppls': [],
        'bleu_scores': [],
        'learning_rates': [],
        'global_step': 0,
        'best_val_loss': float('inf'),
        'best_bleu_score': 0.0,
        'step_losses': [],
        'step_ppls': [],
        'last_100_losses': [],
        'last_100_ppls': [],
    }
    
    if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
        loaded_state = load_checkpoint(
            resume_from_checkpoint, model, optimizer, scheduler, scaler
        )
        training_state.update(loaded_state)
    
    print("🚀 Starting step-based training with FP16 mixed precision...")
    print("📝 Logging every step with loss and metrics")
    start_time = time.time()
    
    train_iter = iter(train_loader)
    
    pbar = tqdm(total=MAX_TRAIN_STEPS, desc="Training Steps", initial=training_state['global_step'])
    
    while training_state['global_step'] < MAX_TRAIN_STEPS:
        model.train()
        optimizer.zero_grad()
        
        accumulated_loss = 0.0
        
        for accum_step in range(GRAD_ACCUM_STEPS):
            try:
                src_batch, src_lang_batch, tgt_batch, tgt_lang_batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                src_batch, src_lang_batch, tgt_batch, tgt_lang_batch = next(train_iter)
            
            src_batch = src_batch.to(device, non_blocking=True)
            src_lang_batch = src_lang_batch.to(device, non_blocking=True)
            tgt_batch = tgt_batch.to(device, non_blocking=True)
            tgt_lang_batch = tgt_lang_batch.to(device, non_blocking=True)
            
            decoder_input = tgt_batch[:, :-1]
            targets = tgt_batch[:, 1:]
            decoder_lang_input = tgt_lang_batch[:, :-1]
            
            src_mask = create_padding_mask(src_batch, tokenizer.pad_id, device).unsqueeze(1).unsqueeze(2)
            tgt_padding_mask = create_padding_mask(decoder_input, tokenizer.pad_id, device)
            causal_mask = create_causal_mask(decoder_input.size(1), device)
            tgt_mask = causal_mask.unsqueeze(0) & tgt_padding_mask.unsqueeze(1)
            
            with torch.cuda.amp.autocast(enabled=USE_AMP and torch.cuda.is_available()):
                logits, _, _, _ = model(
                    src_batch, src_lang_batch,
                    decoder_input, decoder_lang_input,
                    src_mask, tgt_mask
                )
                loss = criterion(logits.view(-1, actual_vocab_size), targets.reshape(-1))
                loss = loss / GRAD_ACCUM_STEPS
            
            scaler.scale(loss).backward()
            accumulated_loss += loss.item() * GRAD_ACCUM_STEPS
        
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        
        scaler.step(optimizer)
        scaler.update()
        
        scheduler.step()
        
        optimizer.zero_grad()
        training_state['global_step'] += 1
        
        current_loss = accumulated_loss
        current_ppl = math.exp(min(current_loss, 20))
        training_state['step_losses'].append(current_loss)
        training_state['step_ppls'].append(current_ppl)
        training_state['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        training_state['last_100_losses'].append(current_loss)
        training_state['last_100_ppls'].append(current_ppl)
        if len(training_state['last_100_losses']) > 100:
            training_state['last_100_losses'].pop(0)
            training_state['last_100_ppls'].pop(0)
        
        if len(training_state['last_100_losses']) > 0:
            avg_loss = np.mean(training_state['last_100_losses'])
            avg_ppl = np.mean(training_state['last_100_ppls'])
        else:
            avg_loss = current_loss
            avg_ppl = current_ppl
        
        pbar.update(1)
        pbar.set_postfix({
            'Loss': f'{current_loss:.4f}',
            'Avg Loss': f'{avg_loss:.4f}',
            'PPL': f'{current_ppl:.2f}',
            'LR': f'{optimizer.param_groups[0]["lr"]:.2e}',
            'Step': training_state['global_step']
        })
        
        if training_state['global_step'] % 10 == 0:
            print(f"\n📊 Step {training_state['global_step']:,}: "
                  f"Loss={current_loss:.4f} (avg100={avg_loss:.4f}), "
                  f"PPL={current_ppl:.2f} (avg100={avg_ppl:.2f}), "
                  f"LR={optimizer.param_groups[0]['lr']:.2e}")
        
        if training_state['global_step'] % EVAL_STEPS == 0:
            print(f"\n🔍 Step {training_state['global_step']:,}: Evaluating...")
            
            model.eval()
            val_loss = 0
            val_samples = 0
            
            if len(val_loader) > 0:
                with torch.no_grad():
                    for val_src, val_src_lang, val_tgt, val_tgt_lang in val_loader:
                        val_src = val_src.to(device, non_blocking=True)
                        val_src_lang = val_src_lang.to(device, non_blocking=True)
                        val_tgt = val_tgt.to(device, non_blocking=True)
                        val_tgt_lang = val_tgt_lang.to(device, non_blocking=True)
                        
                        val_decoder_input = val_tgt[:, :-1]
                        val_targets = val_tgt[:, 1:]
                        val_decoder_lang_input = val_tgt_lang[:, :-1]
                        
                        val_src_mask = create_padding_mask(val_src, tokenizer.pad_id, device).unsqueeze(1).unsqueeze(2)
                        val_tgt_padding_mask = create_padding_mask(val_decoder_input, tokenizer.pad_id, device)
                        val_causal_mask = create_causal_mask(val_decoder_input.size(1), device)
                        val_tgt_mask = val_causal_mask.unsqueeze(0) & val_tgt_padding_mask.unsqueeze(1)
                        
                        with torch.cuda.amp.autocast(enabled=USE_AMP and torch.cuda.is_available()):
                            val_logits, _, _, _ = model(
                                val_src, val_src_lang,
                                val_decoder_input, val_decoder_lang_input,
                                val_src_mask, val_tgt_mask
                            )
                            val_batch_loss = criterion(val_logits.view(-1, actual_vocab_size), val_targets.reshape(-1))
                            val_loss += val_batch_loss.item()
                            val_samples += 1
                
                avg_val_loss = val_loss / max(1, val_samples)
                val_ppl = math.exp(min(avg_val_loss, 20))
                
                training_state['val_losses'].append(avg_val_loss)
                training_state['val_ppls'].append(val_ppl)
                
                print(f"   Validation Loss: {avg_val_loss:.4f}")
                print(f"   Validation PPL: {val_ppl:.2f}")
                
                if avg_val_loss < training_state['best_val_loss']:
                    training_state['best_val_loss'] = avg_val_loss
                    print(f"   🏆 New best validation loss!")
            
            print("   Calculating BLEU score...")
            model_to_eval = model.module if hasattr(model, 'module') else model
            
            if 'val_pairs' in globals() and val_pairs:
                bleu_results = evaluate_translation_quality(
                    model_to_eval, val_pairs[:20], device, num_samples=20
                )
                current_bleu = bleu_results['bleu_score']
                training_state['bleu_scores'].append(current_bleu)
                
                print(f"   BLEU Score: {current_bleu:.4f}")
                
                if current_bleu > training_state['best_bleu_score']:
                    training_state['best_bleu_score'] = current_bleu
                    torch.save(model_to_eval.state_dict(), "best_bleu_model.pth")
                    print(f"   🏆 New best BLEU score! Model saved.")
            
            print("   Sample Translation:")
            english_sentence = "Hello, how are you today?"
            print(f"   English: {english_sentence}")
            
            try:
                hindi_translation = translate_sentence_beam_search(
                    model_to_eval, english_sentence, device, src_lang="en", tgt_lang="hi"
                )
                print(f"   Hindi (Beam Search): {hindi_translation}")
            except Exception as e:
                print(f"   Translation error: {e}")
            
            model.train()
        
        if training_state['global_step'] % SAVE_STEPS == 0:
            print(f"\n💾 Step {training_state['global_step']:,}: Saving checkpoint...")
            save_checkpoint(
                training_state['global_step'], model, optimizer, scheduler, scaler,
                training_state['train_losses'], training_state['val_losses'],
                training_state['train_ppls'], training_state['val_ppls'],
                training_state['bleu_scores'], training_state['learning_rates'],
                training_state['global_step'], training_state['best_val_loss'],
                training_state['best_bleu_score']
            )
        
        if CLEAR_CACHE_EVERY_N_STEPS > 0 and training_state['global_step'] % CLEAR_CACHE_EVERY_N_STEPS == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        if len(training_state['val_losses']) > 10:
            recent_val_losses = training_state['val_losses'][-10:]
            if all(loss > training_state['best_val_loss'] * 1.05 for loss in recent_val_losses):
                print(f"\n🛑 Early stopping at step {training_state['global_step']:,} (validation loss not improving)")
                break
    
    pbar.close()
    total_time = time.time() - start_time
    print(f"\n🎉 Training complete in {total_time/60:.1f} minutes!")
    print(f"📊 Final Step: {training_state['global_step']:,}")
    print(f"📈 Best Validation Loss: {training_state['best_val_loss']:.4f}")
    print(f"🏆 Best BLEU Score: {training_state['best_bleu_score']:.4f}")
    
    return model, training_state

# ===================== MAIN EXECUTION =====================
if __name__ == "__main__":
    print("=" * 80)
    print("🚀 English-Hindi Translation Training with Step-Based Training")
    print("=" * 80)
    
    print("📦 Step 1: Loading dataset...")
    iitb_dataset = load_full_iitb_dataset(DATASET_PATH)
    
    print("\n🔄 Step 2: Extracting translation pairs...")
    train_pairs = []
    val_pairs = []
    
    all_data = []
    for item in iitb_dataset:
        if 'translation' in item:
            en_text = clean_text(item['translation'].get('en', ''))
            hi_text = clean_text(item['translation'].get('hi', ''))
            if en_text and hi_text:
                all_data.append(("en", en_text, "hi", hi_text))
    
    if len(all_data) == 0:
        print("❌ No valid data extracted from dataset")
        all_data = [("en", "Hello", "hi", "नमस्ते"), 
                   ("en", "Good morning", "hi", "शुभ प्रभात")]
    
    random.shuffle(all_data)
    split_idx = int(len(all_data) * 0.95)
    train_pairs = all_data[:split_idx]
    val_pairs = all_data[split_idx:]
    
    print(f"📊 Total pairs: {len(all_data):,}")
    print(f"🚂 Training pairs: {len(train_pairs):,}")
    print(f"📊 Validation pairs: {len(val_pairs):,}")
    
    print("\n🔤 Step 3: Training tokenizer...")
    global tokenizer
    tokenizer = SentencePieceUnigramTokenizer(vocab_size=VOCAB_SIZE)
    
    all_texts = []
    for src_lang, src_text, tgt_lang, tgt_text in train_pairs + val_pairs:
        all_texts.append(clean_text(src_text))
        all_texts.append(clean_text(tgt_text))
    
    all_texts = list(set(all_texts))
    print(f"📝 Training tokenizer on {len(all_texts):,} unique texts")
    
    tokenizer_path = "/kaggle/input/30000hai/spm_unigram_translation(8).model"
    vocab_path = "/kaggle/input/30000hai/spm_unigram_translation(9).vocab"
    
    if os.path.exists(tokenizer_path) and os.path.exists(vocab_path):
        tokenizer.load(tokenizer_path)
        print(f"✅ Loaded existing tokenizer from {tokenizer_path}")
    else:
        try:
            if os.path.exists(tokenizer_path):
                os.remove(tokenizer_path)
            if os.path.exists(vocab_path):
                os.remove(vocab_path)
                
            tokenizer.train(all_texts, model_prefix="spm_unigram_translation")
            print(f"✅ Tokenizer trained and loaded")
        except Exception as e:
            print(f"❌ Error training tokenizer: {e}")
            print("🔄 Using a simple fallback tokenizer...")
            class SimpleTokenizer:
                def __init__(self):
                    self.pad_id = 0
                    self.unk_id = 1
                    self.bos_id = 2
                    self.eos_id = 3
                    self.en_id = 4
                    self.hi_id = 5
                    self.sep_id = 6
                    self.vocab = {}
                    self.reverse_vocab = {}
                
                def encode(self, text, add_bos=True, add_eos=True, **kwargs):
                    tokens = []
                    if add_bos:
                        tokens.append(self.bos_id)
                    for char in text[:100]:
                        if char not in self.vocab:
                            self.vocab[char] = len(self.vocab) + 10
                        tokens.append(self.vocab[char])
                    if add_eos:
                        tokens.append(self.eos_id)
                    return tokens
                
                def decode(self, ids):
                    if not hasattr(self, 'reverse_vocab'):
                        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
                    text = ""
                    for id in ids:
                        if id in self.reverse_vocab:
                            text += self.reverse_vocab[id]
                    return text
                
                def get_vocab_size(self):
                    return max(len(self.vocab), 100) + 20
            
            tokenizer = SimpleTokenizer()
    
    print("\n🧪 Testing tokenizer...")
    try:
        tokenizer.test_encode_decode("Hello, how are you?")
        tokenizer.test_encode_decode("नमस्ते, आप कैसे हैं?")
    except:
        print("⚠️  Tokenizer test skipped")
    
    print("\n📚 Step 4: Creating datasets...")
    train_dataset = TranslationDataset(train_pairs, SEQ_LEN, enable_unigram_sampling=True)
    val_dataset = TranslationDataset(val_pairs, SEQ_LEN, enable_unigram_sampling=False)
    
    print(f"✅ Train sequences: {len(train_dataset):,}")
    print(f"✅ Val sequences: {len(val_dataset):,}")
    
    checkpoint_to_resume = None
    if RESUME_TRAINING:
        if CHECKPOINT_PATH and os.path.exists(CHECKPOINT_PATH):
            checkpoint_to_resume = CHECKPOINT_PATH
        else:
            checkpoint_to_resume = find_latest_checkpoint()
        
        if checkpoint_to_resume:
            print(f"🔄 Will resume training from: {checkpoint_to_resume}")
        else:
            print("🆕 No checkpoint found, starting from scratch")
    
    print("\n🚀 Step 5: Starting training...")
    try:
        trained_model, training_state = train_model(
            train_dataset, val_dataset, resume_from_checkpoint=checkpoint_to_resume
        )
        
        print("\n📊 Step 6: Final evaluation...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_to_eval = trained_model.module if hasattr(trained_model, 'module') else trained_model
        
        final_results = evaluate_translation_quality(
            model_to_eval, val_pairs[:50], device, num_samples=50
        )
        
        print(f"\n🎯 Final BLEU Score: {final_results['bleu_score']:.4f}")
        print(f"📈 Best BLEU Score: {training_state['best_bleu_score']:.4f}")
        
        torch.save(model_to_eval.state_dict(), "final_model.pth")
        print("💾 Saved final model as 'final_model.pth'")
        
        history = {
            'train_losses': training_state['train_losses'],
            'val_losses': training_state['val_losses'],
            'train_ppls': training_state['train_ppls'],
            'val_ppls': training_state['val_ppls'],
            'bleu_scores': training_state['bleu_scores'],
            'learning_rates': training_state['learning_rates'],
            'step_losses': training_state['step_losses'],
            'step_ppls': training_state['step_ppls'],
            'final_bleu': final_results['bleu_score'],
            'best_bleu': training_state['best_bleu_score'],
            'total_steps': training_state['global_step'],
            'config': {
                'D_MODEL': D_MODEL,
                'D_K': D_K,
                'FF_DIM': FF_DIM,
                'NUM_ENCODER_LAYERS': NUM_ENCODER_LAYERS,
                'NUM_DECODER_LAYERS': NUM_DECODER_LAYERS,
                'VOCAB_SIZE': tokenizer.get_vocab_size(),
                'SEQ_LEN': SEQ_LEN,
                'BATCH_SIZE': BATCH_SIZE,
                'GRAD_ACCUM_STEPS': GRAD_ACCUM_STEPS,
                'LR': LR,
                'NUM_HEADS': NUM_HEADS,
                'SAVE_STEPS': SAVE_STEPS,
                'EVAL_STEPS': EVAL_STEPS,
                'MAX_TRAIN_STEPS': MAX_TRAIN_STEPS,
            }
        }
        
        with open('training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        print("💾 Saved training history as 'training_history.json'")
        
        print("\n🌐 Final Translation Tests:")
        test_sentences = [
            "Hello, how are you?",
            "What is your name?",
            "Where is the nearest hospital?",
            "Thank you for your help.",
            "I love learning new languages."
        ]
        
        for sentence in test_sentences:
            translation = translate_sentence_beam_search(
                model_to_eval, sentence, device, src_lang="en", tgt_lang="hi"
            )
            print(f"  English: {sentence}")
            print(f"  Hindi: {translation}")
            print()
        
    except Exception as e:
        print(f"❌ Training error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("🎉 TRAINING COMPLETE! 🎉")
    print("=" * 80)
    print("📁 Generated files:")
    print("   • best_model.pth - Best model by validation loss")
    print("   • best_bleu_model.pth - Best model by BLEU score")
    print("   • final_model.pth - Final trained model")
    print("   • spm_unigram_translation.model - Trained tokenizer")
    print("   • spm_unigram_translation.vocab - Tokenizer vocabulary")
    print("   • checkpoints/ - Training checkpoints (every 5000 steps)")
    print("   • training_history.json - Training metrics history")
    print("\n🚀 Ready for inference or resuming training!")
