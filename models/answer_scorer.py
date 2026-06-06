import torch
import torch.nn as nn

class SiameseAnswerScorer(nn.Module):
    """Siamese MLP scorer with Dropout to prevent overfitting."""
    def __init__(self, embed_dim=64):
        super(SiameseAnswerScorer, self).__init__()
        self.q_encoder = nn.Linear(embed_dim, 64)
        self.a_encoder = nn.Linear(embed_dim, 64)

        # MLP with Dropout between layers
        self.fc1     = nn.Linear(128, 64)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        self.fc2     = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, q_emb, a_emb):
        """
        q_emb: (batch, embed_dim)
        a_emb: (batch, embed_dim)
        """
        q_encoded = self.q_encoder(q_emb)
        a_encoded = self.a_encoder(a_emb)

        combined = torch.cat([q_encoded, a_encoded], dim=1)  # (batch, 128)
        out = self.fc1(combined)
        out = self.relu(out)
        out = self.dropout(out)   # Dropout added here
        out = self.fc2(out)
        return self.sigmoid(out)


class DistilBERTAnswerScorer(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_encoder = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU()
        )
        self.a_encoder = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU()
        )
        self.scorer = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, q_vec, a_vec):
        q_enc    = self.q_encoder(q_vec)
        a_enc    = self.a_encoder(a_vec)
        combined = torch.cat((q_enc, a_enc), dim=1)
        score    = self.scorer(combined)
        return score
