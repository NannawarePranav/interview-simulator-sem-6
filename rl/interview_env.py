import numpy as np

class InterviewEnv:
    def __init__(self, candidate_profile: dict, topics: list):
        self.profile = candidate_profile
        self.topics = topics
        self.num_topics = len(topics)
        self.max_questions = 15
        self.reset()
        
    def reset(self) -> np.ndarray:
        self.topics_covered = set()
        self.scores_per_topic = {t: [] for t in self.topics}
        self.questions_asked = 0
        self.consecutive_weak = 0
        return self._encode_state()
        
    def step(self, action: int) -> tuple:
        topic = self.topics[action]
        
        base_quality = self.profile.get(topic, 0.5)
        score = float(np.clip(np.random.normal(base_quality, 0.1), 0.0, 1.0))
        
        self.scores_per_topic[topic].append(score)
        self.questions_asked += 1
        
        if score < 0.4:
            self.consecutive_weak += 1
        else:
            self.consecutive_weak = 0
            
        reward = score
        if topic in self.topics_covered:
            reward -= 0.1
            
        self.topics_covered.add(topic)
        
        done = (len(self.topics_covered) == self.num_topics) or (self.questions_asked >= self.max_questions)
        
        return self._encode_state(), reward, done
        
    def _encode_state(self) -> np.ndarray:
        one_hot_covered = [1.0 if t in self.topics_covered else 0.0 for t in self.topics]
        avg_scores = []
        for t in self.topics:
            if self.scores_per_topic[t]:
                avg_scores.append(sum(self.scores_per_topic[t]) / len(self.scores_per_topic[t]))
            else:
                avg_scores.append(0.0)
                
        state = one_hot_covered + avg_scores + [self.consecutive_weak / 5.0, self.questions_asked / self.max_questions]
        return np.array(state, dtype=np.float32)
