from collections import defaultdict
from numpy import log2

class MutualInformation:
    def __init__(self, tokens: list[str]) -> None:
        self.tokens: list[str] = tokens
        self.tokens_length = len(self.tokens)
        self.tokens_counts = self.count_tokens()
    
    def count_tokens(self):
        counts = defaultdict(int)

        for token in self.tokens:
            counts[token] += 1
        
        return counts
    
    def count_joint_tokens(self, lag: int):
        joint_counts = defaultdict(int)
        if lag <= 0:
            raise ValueError("Lag must be a positive integer.")
        
        for i in range(self.tokens_length - lag):
            x = self.tokens[i]
            y = self.tokens[i + 1]
            joint_counts[(x, y)] += 1

        return joint_counts
    
    def get_marginal_probability(self, word: str) -> float:
        return self.tokens_counts[word] / self.tokens_length
    
    def get_joint_probability(self, word_x: str, word_y: str, join_counts) -> float:
        number_of_pairs = sum(join_counts.values())
        return join_counts[(word_x, word_y)] / number_of_pairs if number_of_pairs > 0 else 0.0
    
    def calculate_mutual_information(self, lag: int):
        joint_counts = self.count_joint_tokens(lag = lag)
        number_of_pairs = sum(joint_counts.values())

        if number_of_pairs == 0:
            return 0.0
        
        mi = 0.0
        for (x, y), count_xy in joint_counts.items():
            p_xy = count_xy / number_of_pairs
            p_x = self.get_marginal_probability(word = x)
            p_y = self.get_marginal_probability(word = y)
            if (p_xy > 0) and (p_x > 0) and (p_y > 0):
                mi += p_xy * log2(p_xy / (p_x * p_y))
        
        return mi
