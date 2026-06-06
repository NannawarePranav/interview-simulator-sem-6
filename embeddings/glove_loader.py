import numpy as np

def load_glove(path: str, dim=100) -> dict:
    glove = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= dim + 1:
                word = " ".join(parts[:-dim])
                try:
                    vec = np.array([float(x) for x in parts[-dim:]], dtype=np.float32)
                    glove[word] = vec
                except ValueError:
                    pass
    return glove

def get_glove_embedding(tokens: list, glove: dict, fallback_model=None) -> np.ndarray:
    vecs = []
    for t in tokens:
        if t in glove:
            vecs.append(glove[t])
        elif fallback_model is not None:
            # Fall back to NN1 (SimpleSkipGram)
            fallback_vec = fallback_model.get_text_embedding([t])
            # If NN1 is 64-dim, pad to 100-dim
            if fallback_vec.shape[0] < 100:
                padded = np.zeros(100, dtype=np.float32)
                padded[:fallback_vec.shape[0]] = fallback_vec
                vecs.append(padded)
            else:
                vecs.append(fallback_vec[:100])
                
    if not vecs:
        return np.zeros(100, dtype=np.float32)
        
    return np.mean(vecs, axis=0)
