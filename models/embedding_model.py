import numpy as np
import os
import json

# ── GloVe fallback cache (module-level, loaded lazily) ───────────────────────
_glove_cache: dict = {}
_glove_loaded: bool = False

def _load_full_glove(glove_path: str):
    """Load the ENTIRE glove file into memory once."""
    global _glove_loaded
    if _glove_loaded or not os.path.exists(glove_path):
        return
    print(f"Loading full GloVe dictionary from {os.path.basename(glove_path)} into memory...", flush=True)
    with open(glove_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.rstrip().split(' ')
            try:
                _glove_cache[parts[0]] = np.array(parts[1:], dtype=np.float32)
            except ValueError:
                pass
    _glove_loaded = True
    print(f"Loaded {len(_glove_cache)} GloVe vectors.", flush=True)


class SimpleSkipGram:
    """
    Pure NumPy Skip-Gram model with Negative Sampling.
    OOV words fall back to GloVe 6B 100d, then to zero vector.
    """
    def __init__(self, vocab_size, embed_dim=64):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        # W_in: shape (vocab_size, embed_dim)
        self.W_in = np.random.uniform(-0.1, 0.1, (vocab_size, embed_dim))
        # W_out: shape (vocab_size, embed_dim)
        self.W_out = np.random.uniform(-0.1, 0.1, (vocab_size, embed_dim))
        
        self.word2id = {}
        self.id2word = {}
        
        # Path to GloVe fallback — set after load() or build_vocab() when
        # config is available.
        self._glove_path: str = ""
        self._glove_cache: dict = {}  # word → np.ndarray (100d)
        
    def set_glove_fallback(self, glove_path: str):
        """Register a GloVe file to use for OOV words."""
        self._glove_path = glove_path

    def build_vocab(self, tokens, min_freq=2):
        freqs = {}
        for token in tokens:
            freqs[token] = freqs.get(token, 0) + 1
            
        # Add <UNK> and <PAD>
        self.word2id = {"<PAD>": 0, "<UNK>": 1}
        self.id2word = {0: "<PAD>", 1: "<UNK>"}
        
        idx = 2
        for token, count in freqs.items():
            if count >= min_freq:
                self.word2id[token] = idx
                self.id2word[idx] = token
                idx += 1
                
        self.vocab_size = len(self.word2id)
        # Reinitialize weights based on exact vocab size
        self.W_in = np.random.uniform(-0.1, 0.1, (self.vocab_size, self.embed_dim))
        self.W_out = np.random.uniform(-0.1, 0.1, (self.vocab_size, self.embed_dim))
        
    def _sigmoid(self, x):
        # Clip x to avoid overflow
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))
        
    def train_step(self, target_idx, context_idx, negative_indices, lr=0.01):
        """
        Single manual SGD step using negative sampling.
        """
        # Get vectors
        v_target = self.W_in[target_idx]
        
        # Positive sample
        v_pos = self.W_out[context_idx]
        z_pos = np.dot(v_target, v_pos)
        p_pos = self._sigmoid(z_pos)
        
        # Loss component: -log(p_pos)
        loss = -np.log(p_pos + 1e-8)
        
        # Gradients for positive
        g_pos = p_pos - 1.0
        grad_v_target = g_pos * v_pos
        grad_v_pos = g_pos * v_target
        
        self.W_out[context_idx] -= lr * grad_v_pos
        
        # Negative samples
        for neg_idx in negative_indices:
            v_neg = self.W_out[neg_idx]
            z_neg = np.dot(v_target, v_neg)
            p_neg = self._sigmoid(z_neg)
            
            # Loss component: -log(1 - p_neg)
            loss += -np.log(1.0 - p_neg + 1e-8)
            
            g_neg = p_neg
            grad_v_target += g_neg * v_neg
            grad_v_neg = g_neg * v_target
            
            self.W_out[neg_idx] -= lr * grad_v_neg
            
        # Update target word embedding
        self.W_in[target_idx] -= lr * grad_v_target
        
        return loss

    def _glove_lookup(self, oov_words: set) -> dict:
        """Looks up words in the globally loaded GloVe dictionary."""
        if not self._glove_path:
            return {}
        _load_full_glove(self._glove_path)
        return {w: _glove_cache[w] for w in oov_words if w in _glove_cache}

    def get_embedding(self, word):
        """Returns embedding for a single word. Falls back to GloVe then zero."""
        if word in self.word2id:
            return self.W_in[self.word2id[word]]
        
        # GloVe fallback
        if self._glove_path:
            glove_hits = self._glove_lookup({word})
            if word in glove_hits:
                vec = glove_hits[word]
                # Project if dimensions differ (GloVe is 100d, NN1 might be 64d)
                if len(vec) != self.embed_dim:
                    vec = vec[:self.embed_dim] if len(vec) > self.embed_dim else np.pad(vec, (0, self.embed_dim - len(vec)))
                return vec
        
        # Final fallback: zero vector
        return np.zeros(self.embed_dim, dtype=np.float32)
        
    def get_text_embedding(self, tokens):
        """Returns average embedding for a list of tokens."""
        if not tokens:
            return np.zeros(self.embed_dim)
        
        in_vocab = [w for w in tokens if w in self.word2id]
        oov = [w for w in tokens if w not in self.word2id]
        
        embeddings = [self.W_in[self.word2id[w]] for w in in_vocab]
        
        if oov and self._glove_path:
            glove_hits = self._glove_lookup(set(oov))
            for w in oov:
                if w in glove_hits:
                    vec = glove_hits[w]
                    if len(vec) != self.embed_dim:
                        vec = vec[:self.embed_dim] if len(vec) > self.embed_dim else np.pad(vec, (0, self.embed_dim - len(vec)))
                    embeddings.append(vec)
                # else: silently skip (zero contribution via mean)
        
        if not embeddings:
            return np.zeros(self.embed_dim, dtype=np.float32)
        return np.mean(embeddings, axis=0).astype(np.float32)
        
    def save(self, filepath, vocab_path):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        np.save(filepath, self.W_in)
        with open(vocab_path, 'w') as f:
            json.dump(self.word2id, f)
            
    def load(self, filepath, vocab_path):
        self.W_in = np.load(filepath)
        with open(vocab_path, 'r') as f:
            self.word2id = json.load(f)
        self.id2word = {int(v): k for k, v in self.word2id.items()}
        self.vocab_size = len(self.word2id)
        self.embed_dim = self.W_in.shape[1]
