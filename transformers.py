import numpy as np
import os
import math
counting = 0
while True:
    # Define softmax function
    softmax = lambda x: np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)))

    # Define layer normalization
    def LayerNormalisation(srr, gamma, beta ,a=0):
        if a==0:
            E = 1e-9
            mean = np.mean(srr)
            variance = np.var(srr)
            array_L = np.zeros(len(srr))
            for i in range(len(srr)):
                array_L[i] = gamma[i] * (srr[i] - mean) / (np.sqrt(variance + E)) + beta[i]
            return array_L
        else:        
            E = 1e-9
            x = srr
            N = len(x)
            mean = np.mean(x)
            var = np.var(x)
            std = np.sqrt(var + E)

            x_hat = (x - mean) / std

            d_y_dx = np.zeros_like(x)
            for j in range(N):
                d_y_dx[j] = (
                    gamma[j] / std
                    - gamma[j] * (x[j] - mean) * (2 / (N * std**3)) * (x[j] - mean)
                    - gamma[j] * (1 / N) * (1 / std)
                )
            return d_y_dx
    # Sample dataset with 20 vocabulary words and input sentences
    arr  = [
        "i", "am", "playing", "football", "she", "is", "reading", "book",
        "they", "are", "dancing", "he", "was", "sleeping", "in", "park",
        "a", "dog", "runs", "<PAD>"
    ]
    embedding_dim = 10
    vocab = list(dict.fromkeys(arr))
    vocab_size = len(vocab)

    learningrate = 0.001
    # Example input (simulate training sample)
    a_input = "i am"
    input1 = "i am"
    output1 = "sleeping"

    input2 = "they are"
    output2 = "playing"

    input3 = "she is"
    output3 = "dancing"

    input4 = "i am"
    output4 = "reading"

    a_words = input2.split()
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
    print(X)
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
    FFN_hidden = np.array([np.maximum(0, x @ W1 + b1) for x in Layer_norm])
    FFN_output = np.array([x @ W2 + b2 for x in FFN_hidden])

    gamma_file1, beta_file1 = "gamma1.npy", "beta1.npy"
    if os.path.exists(gamma_file1) and os.path.exists(beta_file1):
        gamma1 = np.load(gamma_file1)
        beta1 = np.load(beta_file1)
    else:
        gamma1 = np.random.uniform(0.9, 1.1, size=(embedding_dim,))
        beta1 = np.random.uniform(-0.1, 0.1, size=(embedding_dim,))
        np.save(gamma_file1, gamma1)
        np.save(beta_file1, beta1)
    # Add & Norm 2
    Layer_add_fnn = X + FFN_output
    Layer_norm_fnn = ([LayerNormalisation(Layer_add_fnn[i], gamma1 , beta1) for i in range(len(Layer_add_fnn))])

    # Sentence vector & prediction
    sentence_vector = np.mean(Layer_norm_fnn, axis=0)
    input_words = set(a_words)


    V1 = np.array([word_to_vec[word] for word in vocab])  # Shape: (20, d)
    logits = V1 @ sentence_vector  # Shape: (20,)

    probs = softmax(logits)  # Shape: (20,)
    target_index = vocab.index(output1)
    loss = -np.log(probs[target_index])

    dlogits = probs
    dlogits[target_index] -= 1  # shape: (20,)

    dsentence_vector = V1.T @ dlogits  # shape: (d,)
    dV1 = np.outer(dlogits, sentence_vector)  # shape: (20, d)

    pred_index = np.argmax(probs)
    predicted_word = vocab[pred_index]
    print(predicted_word)
    def layernorm_fnn_training(srr, gamma_1, beta_1,dlbydout):
        E = 1e-9
        mean = np.mean(srr)
        variance = np.var(srr)
        for i in range(len(gamma_1)):
            gamma_1[i] = gamma_1[i] - dlbydout * (learningrate * ((srr[i] - mean) / (np.sqrt(variance + E))))
            beta_1[i] = beta_1[i] - dlbydout * (learningrate * beta_1[i])
        np.save(gamma_file1, gamma_1)
        np.save(beta_file1, beta_1)
        
    def fnn_training(inputs , weights , biases , dlbydout):
        i_len , w_len , b_len = len(inputs) , len(weights) , len(biases)
        for i in range(i_len):
            weight = w_len[i]
            for j in range(len(weight)):
                print()

    def W2_training(dffnout):
        for i in range(len(FFN_output)):
            for j in range(len(W2)):
                for k in range(len(W2[0])):
                     dL_dffnout_i1 = dl_by_dfnnout[i][k]      # ∂L/∂FFN_output[i][1]
                     h_i0 = FFN_hidden[i][j]                  # FFN_hidden[i][0] is the input to W2[0][1]
                     grad = dL_dffnout_i1 * h_i0              # chain rule
                     W2[j][k] -= learningrate * grad
                     b2[k] -= learningrate * dL_dffnout_i1
                      #print(W2[j][k])
        np.save(ffn_w2_file, W2)
        np.save(ffn_b2_file, b2)
        
    dl_by_dout = -(1/(probs[target_index]*np.log(10)))
    dl_by_dlogits = probs - one_hot[target_index]
    dl_by_dsentence = V1.T @ dl_by_dlogits
    dl_by_dffn_layernorm = dl_by_dsentence * 1/len(a_words)
    for i in range(len(Layer_norm_fnn)):
          layernorm_fnn_training(Layer_norm_fnn[i] , gamma1 , beta1 , dl_by_dffn_layernorm[i])
    #print(W2)
    dffn_layernorm_by_dfnn_layeradd = np.array([LayerNormalisation(Layer_add_fnn[i], gamma1 , beta1 ,1) for i in range(len(Layer_add_fnn))])
    #dl_by_dfnn_layeradd = np.array([i @ dl_by_dffn_layernorm for i in dffn_layernorm_by_dfnn_layeradd])
    dl_by_dfnn_layeradd = dffn_layernorm_by_dfnn_layeradd * dl_by_dffn_layernorm
    dl_by_dfnnout = dl_by_dfnn_layeradd * 1
    #print(len(W2))
    #print(dl_by_dfnnout)
    #print()
    #print(FFN_hidden)
    #print()
    #print(W2)
    #print()
    W2_training(dl_by_dfnnout)
    #print(W2)
    print(counting)
    counting=counting+1
    dl_by_w21 = dl_by_dfnnout * np.array([x @ W2 + b2 for x in FFN_hidden])
    #print()
    #print(dl_by_dfnn_layeradd)

    #for i in range(len(Layer_add_fnn)): layernorm_fnn_training(Layer_add_fnn[i],gamma1,beta1,dl_by_dout) 
