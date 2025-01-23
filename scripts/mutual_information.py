from collections import defaultdict, Counter
from numpy import log2, log
from scipy.special import digamma

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
            y = self.tokens[i + lag]
            joint_counts[(x, y)] += 1

        return joint_counts
    
    def get_marginal_probability(self, word: str) -> float:
        return self.tokens_counts[word] / self.tokens_length
    
    def get_joint_probability(self, word_x: str, word_y: str, joint_counts) -> float:
        number_of_pairs = sum(joint_counts.values())
        return joint_counts[(word_x, word_y)] / number_of_pairs if number_of_pairs > 0 else 0.0
    
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

class EntropyEstimator:
    def __init__(self, tokens: list[str], method: str) -> None:
        """
        Methods: naive, grassberger, miller-madow
        """
        self.tokens: list[str] = tokens
        self.method: str = method

    def lagged_tokens(self, lag: int):
        return self.tokens[:-lag], self.tokens[lag:]
    
    def naive_entropy(self, N_i: dict) -> float:
        N = sum(N_i.values())
        if N == 0:
            return 0.0
        
        return -sum(n_i * log2(n_i / N) for n_i in N_i.values()) / N
    
    def miller_madow_entropy(self, N_i: dict) -> float:
        N = sum(N_i.values())
        if N == 0:
            return 0.0
        
        # Number of bins with non-zero count
        m_hat = len(N_i)

        H_naive = self.naive_entropy(N_i)
        correction = (m_hat - 1) / (2 * N * log(2))

        return H_naive + correction

    def grassberger_entropy(self, N_i: dict) -> float:
        """
        Formula D1 (Lin, Tegmark, 2017):
            S_hat = log(N) - 1/N \sum_i N_i * digamma(N_i)
        where 
            - N_i - number of occurrences of ith token
            - N = \sum_i N_i
        """
        N = sum(N_i.values())
        if N == 0:
            return 0.0
        
        term = sum(n_i * digamma(n_i) for n_i in N_i.values())
        return log2(N) - (term / N)
        
    def estimate_entropy(self, tokens: list[str]) -> float:
        counts = Counter(tokens)
        if self.method == "naive":
            return self.naive_entropy(counts)
        elif self.method == "grassberger":
            return self.grassberger_entropy(counts)
        elif self.method == "miller-madow":
            return self.miller_madow_entropy(counts)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def estimate_joint_entropy(self, token_pairs) -> float:
        pass

    def calculate_mutual_information(self, lag: int) -> float:
        """
        It is well known fact from information theory that MI(X,Y) = H(X) + H(Y) - H(X, Y)
        """
        X, Y = self.lagged_tokens(lag)
        H_X = self.estimate_entropy(X)
        H_Y = self.estimate_entropy(Y)
        H_XY = self.estimate_entropy(list(zip(X, Y)))

        return H_X + H_Y - H_XY