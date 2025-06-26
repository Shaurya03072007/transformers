import numpy as np
import os
import math

# Define softmax function
softmax = lambda x: np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)))

# Define layer normalization
def LayerNormalisation(srr, gamma, beta):
    E = 1e-9
    mean = np.mean(srr)
    variance = np.var(srr)
    array_L = np.zeros(len(srr))
    for i in range(len(srr)):
        array_L[i] = gamma[i] * (srr[i] - mean) / (np.sqrt(variance + E)) + beta[i]
    return array_L

# Sample dataset with 20 vocabulary words and input sentences
arr  = [
    "i", "am", "playing", "football", "she", "is", "reading", "book",
    "they", "are", "dancing", "he", "was", "sleeping", "in", "park",
    "a", "dog", "runs", "<PAD>"
]
embedding_dim = 10
vocab = list(dict.fromkeys(arr))
vocab_size = len(vocab)

# Example input (simulate training sample)
a_input = "sleeping is"
a_words = a_input.split()
if len(a_words) < 5:
    a_words.append("<PAD>")

# One-hot encoding mapping (not used directly here)
one_hot = np.eye(vocab_size)
word_to_onehot = {word: one_hot[i] for i, word in enumerate(vocab)}

# Embeddings
file_path_embeddings = f"word_embeddings_{embedding_dim}.npy"
if os.path.exists(file_path_embeddings):
    word_to_vec = np.load(file_path_embeddings, allow_pickle=True).item()
else:
    word_to_vec = {word: np.random.rand(embedding_dim) * 2 - 1 for word in vocab}
    np.save(file_path_embeddings, word_to_vec)
# Prepare input embeddings
a_dense = {word: word_to_vec[word] for word in a_words}
a_dense_emb = np.zeros((len(a_dense), embedding_dim))
a_pos_enc = np.zeros((len(a_dense), embedding_dim))
for i, word in enumerate(a_dense):
    ar = a_dense[word]
    for j in range(len(ar)):
        exp = i / (1000 ** (2 * j / embedding_dim))
        a_pos_enc[i][j] = math.sin(exp) if j % 2 == 0 else math.cos(exp)
        a_dense_emb[i][j] = ar[j]
X = a_dense_emb + a_pos_enc

# Transformer attention weights
paths = {"WV": "WV.npy", "WK": "WK.npy", "WQ": "WQ.npy", "W0": "W0.npy"}
if all(os.path.exists(p) for p in paths.values()):
    WV = np.load(paths["WV"])
    WK = np.load(paths["WK"])
    WQ = np.load(paths["WQ"])
    W0 = np.load(paths["W0"])
else:
    WV = np.random.rand(embedding_dim, len(a_words)) * 2 - 1
    WK = np.random.rand(embedding_dim, len(a_words)) * 2 - 1
    WQ = np.random.rand(embedding_dim, len(a_words)) * 2 - 1
    W0 = np.random.rand(len(a_words), embedding_dim) * 2 - 1
    np.save(paths["WV"], WV)
    np.save(paths["WK"], WK)
    np.save(paths["WQ"], WQ)
    np.save(paths["W0"], W0)

# Attention mechanism
in_size = 3
Q = X @ WQ
K = X @ WK
V = X @ WV
scores = (Q @ K.T) / np.sqrt(in_size)
weights = softmax(scores)
attention = weights @ V
attention_out = attention @ W0

# Add & Norm 1
Layer_add = X + attention_out
gamma_file, beta_file = "gamma.npy", "beta.npy"
if os.path.exists(gamma_file) and os.path.exists(beta_file):
    gamma = np.load(gamma_file)
    beta = np.load(beta_file)
else:
    gamma = np.random.uniform(0.9, 1.1, size=(embedding_dim,))
    beta = np.random.uniform(-0.1, 0.1, size=(embedding_dim,))
    np.save(gamma_file, gamma)
    np.save(beta_file, beta)
Layer_norm = np.array([LayerNormalisation(Layer_add[i], gamma, beta) for i in range(len(Layer_add))])

# Feed Forward Network (FFN)
hidden_dim = 20
ffn_w1_file, ffn_b1_file = "ffn_w1.npy", "ffn_b1.npy"
ffn_w2_file, ffn_b2_file = "ffn_w2.npy", "ffn_b2.npy"
if all(os.path.exists(f) for f in [ffn_w1_file, ffn_b1_file, ffn_w2_file, ffn_b2_file]):
    W1 = np.load(ffn_w1_file)
    b1 = np.load(ffn_b1_file)
    W2 = np.load(ffn_w2_file)
    b2 = np.load(ffn_b2_file)
else:
    W1 = np.random.randn(embedding_dim, hidden_dim) * 0.02
    b1 = np.zeros(hidden_dim)
    W2 = np.random.randn(hidden_dim, embedding_dim) * 0.02
    b2 = np.zeros(embedding_dim)
    np.save(ffn_w1_file, W1)
    np.save(ffn_b1_file, b1)
    np.save(ffn_w2_file, W2)
    np.save(ffn_b2_file, b2)

FFN_output = np.array([np.maximum(0, x @ W1 + b1) @ W2 + b2 for x in Layer_norm])

# Add & Norm 2
Layer_add_fnn = X + FFN_output
Layer_norm_fnn = np.array([LayerNormalisation(Layer_add_fnn[i], gamma, beta) for i in range(len(Layer_add_fnn))])

# Sentence vector & prediction
sentence_vector = np.mean(Layer_norm_fnn, axis=0)
input_words = set(a_words)
best_match, best_score = None, -float("inf")
for word in vocab:
    if word in input_words:
        continue
    vec = word_to_vec[word]
    score = np.dot(sentence_vector, vec)
    if score > best_score:
        best_score = score
        best_match = word

print(best_match)
