import torch
import torch.nn as nn

class TopicClassifier(nn.Module):
    def __init__(self, embed_dim=64, hidden_dim=128, num_topics=6):
        super(TopicClassifier, self).__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_topics)
        self.log_softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.log_softmax(x)
        return x
