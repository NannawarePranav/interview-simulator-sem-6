import torch
import torch.nn as nn

class QuestionRanker(nn.Module):
    """
    GRU-based question ranker with difficulty-awareness.
    
    Inputs:
      context_seq : (batch, seq_len, embed_dim)  — past question embeddings
      candidate_q : (batch, embed_dim)            — candidate question embedding
      difficulty  : (batch, 1)                    — scalar 0.0/0.5/1.0
    """
    def __init__(self, embed_dim=64, hidden_dim=64):
        super(QuestionRanker, self).__init__()
        # GRU replaces vanilla RNN — better long-range memory retention
        self.gru = nn.GRU(input_size=embed_dim, hidden_size=hidden_dim,
                          num_layers=1, batch_first=True)
        # Scorer: context_hidden + candidate_embed + difficulty_scalar
        self.fc      = nn.Linear(hidden_dim + embed_dim + 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, context_seq, candidate_q, difficulty=None):
        """
        context_seq : (batch, seq_len, embed_dim)
        candidate_q : (batch, embed_dim)
        difficulty  : (batch, 1)  — optional; zero-filled if not provided
        """
        _, h_n = self.gru(context_seq)          # h_n: (1, batch, hidden_dim)
        context_vec = h_n.squeeze(0)             # (batch, hidden_dim)

        if difficulty is None:
            difficulty = torch.zeros(context_vec.size(0), 1,
                                     device=context_vec.device,
                                     dtype=context_vec.dtype)

        x     = torch.cat([context_vec, candidate_q, difficulty], dim=1)
        score = self.fc(x)
        return self.sigmoid(score)
