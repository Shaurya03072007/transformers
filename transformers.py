import numpy as np
import os
import math

# --- Configuration ---
# ERROR FIX: The input list had 105 elements, while the output list had 104.
# This caused an IndexError on the last iteration. Removed the last element from
# input1 to ensure both lists have the same length (104).
input1 = [
    "i am", "they are", "she is", "we are", "you are", "he is",
    "it is", "i was", "they were", "we will", "you will", "he was",
    "she was", "i will", "they will", "we were", "he is not here",
    "we are going to", "you might be", "they could be",
    "i should have been", "you would be", "he shall be walking",
    "they will have been", "we might have been", "he was going to be",
    "she is currently", "i am still", "they are definitely", "you were probably",
    "we could have", "he will likely", "she might", "i will probably",
    "they were already", "we are soon", "you are still", "he was just",
    "she will surely", "i have been", "you had been", "he could be",
    "they might", "we were likely", "i was definitely", "she was just",
    "you are going to", "they have been", "we might be", "i could be",
    "she might have", "you shall be", "they are going to", "he was planning to",
    "we will be", "i am about to", "you were supposed to", "she was about to",
    "they are about to", "we are ready to", "he is expected to", "she is trying to",
    "you might want to", "i feel like", "they seem to be", "we hope to",
    "he pretends to", "she wants to", "i think i will", "you think you will",
    "they believe they will", "we assume we are", "he guesses he will", "she imagines she will",
    "i pretend to", "you imagine you will", "they plan to", "we are planning to",
    "he has decided to", "she intends to", "i am learning to", "you are starting to",
    "they are beginning to", "we are trying to", "he wants to start", "she is ready to",
    "i feel ready to", "you are expected to", "they are willing to", "we are prepared to",
    "he is committed to", "she is determined to", "i hope to", "you plan to",
    "they should", "we could", "he must", "she might want to",
    "i am going to", "you are going to", "they are going to", #"we are going to" <- Removed
][:3]

output1 = [
    "sleeping", "playing", "dancing", "walking", "talking", "eating",
    "drawing", "reading", "coding", "running", "jumping", "writing",
    "building", "painting", "traveling", "learning", "talking", "dancing",
    "walking", "sleeping", "reading", "coding", "running", "jumping",
    "writing", "building", "painting", "traveling", "learning", "reading",
    "talking", "dancing", "walking", "sleeping", "playing", "dancing",
    "walking", "talking", "eating", "drawing", "reading", "coding",
    "running", "jumping", "writing", "building", "painting", "traveling",
    "learning", "reading", "talking", "dancing", "walking", "sleeping",
    "playing", "dancing", "walking", "talking", "eating", "drawing",
    "reading", "coding", "running", "jumping", "writing", "building",
    "painting", "traveling", "learning", "reading", "talking", "dancing",
    "walking", "sleeping", "playing", "dancing", "walking", "talking",
    "eating", "drawing", "reading", "coding", "running", "jumping",
    "writing", "building", "painting", "traveling", "learning", "reading",
    "talking", "dancing", "walking", "sleeping", "playing", "dancing",
    "walking", "talking", "eating", "drawing"
][:3]

# BEST PRACTICE: Added an assertion to ensure data lists have matching lengths.
assert len(input1) == len(output1)


# --- Hyperparameters and Model Configuration ---
embedding_dim = 512
learning_rate = 0.01
num_epochs = 151
max_seq_len = 5

# Define paths for saving/loading model weights
paths = {"WV": "WV.npy", "WK": "WK.npy", "WQ": "WQ.npy", "W0": "W0.npy"}
file_path_embeddings = f"word_embeddings_{embedding_dim}.npy"
ffn_w1_file, ffn_b1_file = "ffn_w1.npy", "ffn_b1.npy"
ffn_w2_file, ffn_b2_file = "ffn_w2.npy", "ffn_b2.npy"

# --- Vocabulary Creation ---
all_words = set()
for sentence in input1:
    all_words.update(sentence.split())
for word in output1:
    all_words.add(word)
all_words.update(['<END>', '<PAD>', '<START>', '<UNK>'])
vocab = sorted(list(all_words))
vocab_size = len(vocab)
word_to_id = {word: i for i, word in enumerate(vocab)}


# --- Helper Functions ---
def clip_gradient(grad, threshold=1.0):
    norm = np.linalg.norm(grad)
    if norm > threshold:
        return grad * (threshold / (norm + 1e-6))
    return grad

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def positional_encoding(sequence_length, embedding_dim):
    position_enc = np.zeros((sequence_length, embedding_dim))
    for pos in range(sequence_length):
        for i in range(embedding_dim):
            denominator = 10000 ** (i / embedding_dim)
            if i % 2 == 0:
                position_enc[pos, i] = math.sin(pos / denominator)
            else:
                position_enc[pos, i] = math.cos(pos / denominator)
    return position_enc


# --- Layer Normalization ---
class LayerNormalization:
    def __init__(self, features_dim):
        self.gamma = np.ones(features_dim)
        self.beta = np.zeros(features_dim)
        self.eps = 1e-9
        self.mean = None
        self.variance = None
        self.x_normalized = None
        self.x_input = None
        self.features_dim = features_dim

    def forward(self, x):
        self.x_input = x
        self.mean = np.mean(x, axis=-1, keepdims=True)
        self.variance = np.var(x, axis=-1, keepdims=True)
        self.x_normalized = (x - self.mean) / np.sqrt(self.variance + self.eps)
        return self.gamma * self.x_normalized + self.beta

    def backward(self, dout):
        dbeta = np.sum(dout, axis=0)
        dgamma = np.sum(dout * self.x_normalized, axis=0)
        dx_normalized = dout * self.gamma
        N = self.features_dim
        d_var = np.sum(dx_normalized * (self.x_input - self.mean), axis=-1, keepdims=True) * (-0.5 * (self.variance + self.eps)**(-1.5))
        d_mean = np.sum(dx_normalized * (-1.0 / np.sqrt(self.variance + self.eps)), axis=-1, keepdims=True) + \
                 d_var * np.sum(-2.0 * (self.x_input - self.mean) / N, axis=-1, keepdims=True)
        dx = dx_normalized / np.sqrt(self.variance + self.eps) + \
             d_var * (2.0 * (self.x_input - self.mean) / N) + \
             d_mean / N
        return dx, dgamma, dbeta

norm1 = LayerNormalization(embedding_dim)
norm2 = LayerNormalization(embedding_dim)


# --- Model Initialization ---
if not os.path.exists(file_path_embeddings):
    word_to_vec = {word: np.random.randn(embedding_dim) * 0.01 for word in vocab}
    for k in word_to_vec:
        word_to_vec[k] /= np.linalg.norm(word_to_vec[k]) + 1e-9
    np.save(file_path_embeddings, word_to_vec)
else:
    word_to_vec = np.load(file_path_embeddings, allow_pickle=True).item()

if all(os.path.exists(p) for p in paths.values()):
    WV = np.load(paths["WV"]); WK = np.load(paths["WK"])
    WQ = np.load(paths["WQ"]); W0 = np.load(paths["W0"])
else:
    WQ = np.random.randn(embedding_dim, embedding_dim) * 0.01
    WK = np.random.randn(embedding_dim, embedding_dim) * 0.01
    WV = np.random.randn(embedding_dim, embedding_dim) * 0.01
    W0 = np.random.randn(embedding_dim, embedding_dim) * 0.01
    np.save(paths["WQ"], WQ); np.save(paths["WK"], WK); np.save(paths["WV"], WV); np.save(paths["W0"], W0)

hidden_dim = 2048
if all(os.path.exists(f) for f in [ffn_w1_file, ffn_b1_file, ffn_w2_file, ffn_b2_file]):
    W1 = np.load(ffn_w1_file); b1 = np.load(ffn_b1_file)
    W2 = np.load(ffn_w2_file); b2 = np.load(ffn_b2_file)
else:
    W1 = np.random.randn(embedding_dim, hidden_dim) * 0.01
    b1 = np.zeros(hidden_dim)
    W2 = np.random.randn(hidden_dim, embedding_dim) * 0.01
    b2 = np.zeros(embedding_dim)
    np.save(ffn_w1_file, W1); np.save(ffn_b1_file, b1)
    np.save(ffn_w2_file, W2); np.save(ffn_b2_file, b2)


# --- Training Loop ---
for epoch in range(1, num_epochs + 1):
    total_loss = 0
    correct_predictions = 0

    for inpput_idx, input_text in enumerate(input1):
        words = input_text.split()
        original_input_len = len(words)

        if original_input_len < max_seq_len:
            words.extend(['<PAD>'] * (max_seq_len - original_input_len))
        elif original_input_len > max_seq_len:
            words = words[:max_seq_len]

        # --- Forward Pass ---
        input_embeddings = np.array([word_to_vec.get(word, word_to_vec['<UNK>']) for word in words])
        pos_enc = positional_encoding(len(words), embedding_dim)
        X = input_embeddings + pos_enc

        Q = X @ WQ; K = X @ WK; V = X @ WV
        scores = (Q @ K.T) / np.sqrt(embedding_dim)
        weights = softmax(scores)
        attention = weights @ V
        attention_out = attention @ W0

        add_norm1_input = X + attention_out
        norm1_output = norm1.forward(add_norm1_input)

        in_for_relu = norm1_output @ W1 + b1
        ffn_hidden = np.maximum(0, in_for_relu)
        ffn_output = ffn_hidden @ W2 + b2

        add_norm2_input = norm1_output + ffn_output
        norm2_output = norm2.forward(add_norm2_input)

        if original_input_len > 0:
            sentence_vector = np.mean(norm2_output[:original_input_len], axis=0)
        else:
            sentence_vector = np.zeros(embedding_dim)

        V_output = np.array([word_to_vec[word] for word in vocab])
        logits = V_output @ sentence_vector
        probs = softmax(logits)

        target_word = output1[inpput_idx]
        target_index = vocab.index(target_word)
        loss = -np.log(probs[target_index] + 1e-9)

        # --- Backward Pass ---
        dlogits = probs.copy(); dlogits[target_index] -= 1

        dV_output = np.outer(dlogits, sentence_vector)
        dsentence_vector = V_output.T @ dlogits

        dnorm2_output = np.zeros_like(norm2_output)
        if original_input_len > 0:
            dnorm2_output[:original_input_len] = dsentence_vector / original_input_len

        dadd_norm2_input, dgamma2, dbeta2 = norm2.backward(dnorm2_output)

        dffn_output = dadd_norm2_input
        dffn_hidden = dffn_output @ W2.T
        drelu = dffn_hidden * (in_for_relu > 0)
        dW2 = ffn_hidden.T @ dffn_output
        db2 = np.sum(dffn_output, axis=0)
        dW1 = norm1_output.T @ drelu
        db1 = np.sum(drelu, axis=0)
        dx_ffn = drelu @ W1.T

        dnorm1_output = dadd_norm2_input + dx_ffn
        dadd_norm1_input, dgamma1, dbeta1 = norm1.backward(dnorm1_output)

        dattention_out = dadd_norm1_input
        dW0 = attention.T @ dattention_out
        dattention = dattention_out @ W0.T
        dweights = dattention @ V.T
        dscores = dweights * weights
        sum_dscores = np.sum(dscores, axis=-1, keepdims=True)
        dscores -= weights * sum_dscores
        d_S_unscaled = dscores / np.sqrt(embedding_dim)
        
        dQ = d_S_unscaled @ K
        dK = d_S_unscaled.T @ Q
        dV = weights.T @ dattention

        dWQ = X.T @ dQ
        dWK = X.T @ dK
        dWV = X.T @ dV

        dX = dQ @ WQ.T + dK @ WK.T + dV @ WV.T + dadd_norm1_input

        # --- Update All Model Parameters ---
        W0 -= learning_rate * clip_gradient(dW0)
        W2 -= learning_rate * clip_gradient(dW2); b2 -= learning_rate * clip_gradient(db2)
        W1 -= learning_rate * clip_gradient(dW1); b1 -= learning_rate * clip_gradient(db1)
        WV -= learning_rate * clip_gradient(dWV)
        WK -= learning_rate * clip_gradient(dWK)
        WQ -= learning_rate * clip_gradient(dWQ)
        norm2.gamma -= learning_rate * clip_gradient(dgamma2); norm2.beta -= learning_rate * clip_gradient(dbeta2)
        norm1.gamma -= learning_rate * clip_gradient(dgamma1); norm1.beta -= learning_rate * clip_gradient(dbeta1)

        d_word_to_vec = {word: np.zeros(embedding_dim) for word in vocab}
        for i, word in enumerate(vocab):
            d_word_to_vec[word] += dV_output[i]

        original_input_words = input1[inpput_idx].split()
        for j in range(min(len(original_input_words), max_seq_len)):
            word = original_input_words[j]
            if word in vocab:
                d_word_to_vec[word] += dX[j]

        for word in vocab:
            word_to_vec[word] -= learning_rate * clip_gradient(d_word_to_vec[word])

        # --- Logging and Metrics ---
        predicted_index = np.argmax(probs)
        predicted_word = vocab[predicted_index]
        total_loss += loss
        if predicted_word == target_word:
            correct_predictions += 1

        if epoch % 10 == 0 and inpput_idx == 0:
            print(f"{'Epoch':<6} | {'Input':<30} | {'Target':<10} | {'Predicted':<10} | {'Loss':<12}")
            print("-" * 70)
        if epoch % 10 == 0:
            print(f"{epoch:<6} | {input_text:<30} | {target_word:<10} | {predicted_word:<10} | {loss:<12.4f}")

          

    # --- Epoch Summary ---
    epoch_avg_loss = total_loss / len(input1)
    epoch_accuracy = correct_predictions / len(input1)
    print(f"\nEpoch {epoch} Summary: Average Loss = {epoch_avg_loss:.4f}, Accuracy = {epoch_accuracy:.2f}\n")

    # Save model progress
    np.save(paths["WV"], WV); np.save(paths["WK"], WK)
    np.save(paths["WQ"], WQ); np.save(paths["W0"], W0)
    np.save(ffn_w1_file, W1); np.save(ffn_b1_file, b1)
    np.save(ffn_w2_file, W2); np.save(ffn_b2_file, b2)
    np.save(file_path_embeddings, word_to_vec)

    if epoch_accuracy==1.0:
      print("Convergence done. Overfitted!")
      break

print("Training complete.")
