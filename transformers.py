import numpy as np
import os
import math
counting = 0
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
    "i am going to", "you are going to", "they are going to", "we are going to"
]

output1 = [
    "sleeping", "playing", "dancing", "walking", "talking", "eating",
    "drawing", "reading", "coding", "running", "jumping", "writing",
    "building", "painting", "traveling", "learning", "talking",
    "dancing", "walking", "sleeping",
    "reading", "coding", "running", "jumping", "writing", "building",
    "painting", "traveling", "learning", "reading",
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
]
#input1 = ["i am", "you are", "they are"]
#output1 = ["sleeping", "eating", "coding"]
embedding_dim = 512
paths = {"WV": "WV.npy", "WK": "WK.npy", "WQ": "WQ.npy", "W0": "W0.npy"}
if all(os.path.exists(p) for p in paths.values()):
        WV = np.load(paths["WV"])
        WK = np.load(paths["WK"])
        WQ = np.load(paths["WQ"])
        W0 = np.load(paths["W0"])
        
file_path_embeddings = f"word_embeddings_{embedding_dim}.npy"
if os.path.exists(file_path_embeddings):
        word_to_vec = np.load(file_path_embeddings, allow_pickle=True).item()

ffn_w1_file, ffn_b1_file = "ffn_w1.npy", "ffn_b1.npy"
ffn_w2_file, ffn_b2_file = "ffn_w2.npy", "ffn_b2.npy"
if all(os.path.exists(f) for f in [ffn_w1_file, ffn_b1_file, ffn_w2_file, ffn_b2_file]):
        W1 = np.load(ffn_w1_file)
        b1 = np.load(ffn_b1_file)
        W2 = np.load(ffn_w2_file)
        b2 = np.load(ffn_b2_file)

gamma_file, beta_file = "gamma.npy", "beta.npy"
if os.path.exists(gamma_file) and os.path.exists(beta_file):
        gamma = np.load(gamma_file)
        beta = np.load(beta_file)

gamma_file1, beta_file1 = "gamma1.npy", "beta1.npy"
if os.path.exists(gamma_file1) and os.path.exists(beta_file1):
        gamma1 = np.load(gamma_file1)
        beta1 = np.load(beta_file1)
        
#while counting==0:
  #counting = 1
while True:
  #inputs =  np.random.permutation(len(input1)-1)
  for inpput in range(len(input1)):       
    # Define softmax function
    def clip_gradient(grad, threshold=1.0):
        norm = np.linalg.norm(grad)
        return grad if norm <= threshold else grad * (threshold / (norm + 1e-6))

    def softmax(x, axis=-1):
        x = x - np.max(x, axis=axis, keepdims=True)
        return np.exp(x) / np.sum(np.exp(x), axis=axis, keepdims=True)

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
    arr = ['<END>', '<PAD>', '<START>', '<UNK>', 'a', 'about', 'after', 'afternoon', 'already', 'am', 'an', 'and', 'angry', 'are', "aren't", 'assume', 'at', 'ball', 'be', 'beach', 'because', 'bed', 'been', 'before',
           'beginning', 'being', 'believe', 'big', 'bike', 'book', 'build', 'building', 'bus', 'but', 'can', "can't", 'car', 'cat', 'city', 'code', 'coding', 'cold', 'college', 'committed', 'cook', 'cooking', 'could',
           "couldn't", 'currently', 'dance', 'dancing', 'decided', 'definitely', 'determined', 'did', "didn't", 'do', 'does', "doesn't", 'dog', "don't", 'draw', 'drawing', 'drive', 'driving', 'during', 'eat', 'eating',
           'evening', 'expected', 'fast', 'feel', 'food', 'for', 'from', 'game', 'garden', 'going', 'guesses', 'had', 'happy', 'has', 'have', 'having', 'he', "he's", 'her', 'here', 'him', 'his', 'home', 'hope', 'hot',
           'house', 'i', "i'm", "i've", 'if', 'imagine', 'imagines', 'in', 'intends', 'is', "isn't", 'it', "it's", 'its', 'jump', 'jumping', 'just', 'laptop', 'learn', 'learning', 'like', 'likely', 'market', 'may', 'me',
           'might', 'mine', 'morning', 'movie', 'music', 'must', 'my', 'new', 'night', 'no', 'not', 'now', 'old', 'on', 'or', 'our', 'ours', 'paint', 'painting', 'paper', 'park', 'pen', 'phone', 'plan', 'planning', 'play',
           'playing', 'prepared', 'pretend', 'pretends', 'probably', 'read', 'reading', 'ready', 'road', 'room', 'run', 'running', 'sad', 'school', 'seem', 'shall', 'she', "she's", 'shop', 'should', 'sing', 'singing', 'sleep',
           'sleeping', 'slow', 'small', 'so', 'soon', 'start', 'starting', 'still', 'student', 'study', 'studying', 'supposed', 'surely', 'swim', 'swimming', 'talk', 'talking', 'teacher', 'the', 'their', 'theirs', 'them',
           'then', 'they', "they're", "they've", 'think', 'tired', 'to', 'today', 'tomorrow', 'train', 'travel', 'traveling', 'trying', 'us', 'village', 'walk', 'walking', 'want', 'wants', 'was', "wasn't", 'water', 'we',
           "we're", "we've", 'were', "weren't", 'while', 'will', 'willing', 'with', "won't", 'work', 'working', 'would', 'write', 'writing', 'yes', 'yesterday', 'you', "you're", "you've", 'your', 'yours']
    vocab = list(dict.fromkeys(arr))
    vocab_size = len(vocab)

    learningrate = 0.01
    # Example input (simulate training sample)
    a_words = input1[inpput].split()
    input_ids = [vocab.index(word) for word in a_words]  # THIS IS `input_token_ids`
    if len(a_words) < 5:
        a_words.append("<PAD>")

    # One-hot encoding mapping (not used directly here)
    one_hot = np.eye(vocab_size)
    word_to_onehot = {word: one_hot[i] for i, word in enumerate(vocab)}

    # Embeddings
    
    if not os.path.exists(file_path_embeddings):
        word_to_vec = {word: np.random.rand(embedding_dim) * 2 - 1 for word in vocab}
        np.save(file_path_embeddings, word_to_vec)
    # Prepare input embeddings
    for k in word_to_vec:
        word_to_vec[k] /= np.linalg.norm(word_to_vec[k]) + 1e-9
    a_dense = {word: word_to_vec[word] for word in a_words}
    a_dense_emb = np.zeros((len(a_dense), embedding_dim))
    a_pos_enc = np.zeros((len(a_dense), embedding_dim))
    for i, word in enumerate(a_dense):
        ar = a_dense[word]
        for j in range(len(ar)):
            exp = i / (10000 ** (2 * j / embedding_dim))
            a_pos_enc[i][j] = math.sin(exp) if j % 2 == 0 else math.cos(exp)
            a_dense_emb[i][j] = ar[j]
    X = a_dense_emb + a_pos_enc

    # Transformer attention weights
    
    if all(not os.path.exists(p) for p in paths.values()):
        WV = np.random.rand(embedding_dim, len(a_words)) * 2 - 1
        WK = np.random.rand(embedding_dim, len(a_words)) * 2 - 1
        WQ = np.random.rand(embedding_dim, len(a_words)) * 2 - 1
        W0 = np.random.rand(len(a_words), embedding_dim) * 2 - 1
        np.save(paths["WV"], WV)
        np.save(paths["WK"], WK)
        np.save(paths["WQ"], WQ)
        np.save(paths["W0"], W0)

    # Attention mechanism
    Q = X @ WQ
    K = X @ WK
    V = X @ WV
    scores = (Q @ K.T) / np.sqrt(len(a_words))
    weights = softmax(scores)
    attention = weights @ V
    attention_out = attention @ W0

    # Add & Norm 1
    Layer_add = X + attention_out
    if not os.path.exists(gamma_file) and not os.path.exists(beta_file):
        gamma = np.random.uniform(0.9, 1.1, size=(embedding_dim,))
        beta = np.random.uniform(-0.1, 0.1, size=(embedding_dim,))
        np.save(gamma_file, gamma)
        np.save(beta_file, beta)
    Layer_norm = np.array([LayerNormalisation(Layer_add[i], gamma, beta) for i in range(len(Layer_add))])

    # Feed Forward Network (FFN)
    hidden_dim = 2048
    
    if all(not os.path.exists(f) for f in [ffn_w1_file, ffn_b1_file, ffn_w2_file, ffn_b2_file]):
        W1 = np.random.randn(embedding_dim, hidden_dim) * 0.02
        b1 = np.zeros(hidden_dim)
        W2 = np.random.randn(hidden_dim, embedding_dim) * 0.02
        b2 = np.zeros(embedding_dim)
        np.save(ffn_w1_file, W1)
        np.save(ffn_b1_file, b1)
        np.save(ffn_w2_file, W2)
        np.save(ffn_b2_file, b2)
    in_for_relu = np.array([ x @ W1 + b1 for x in Layer_norm])
    FFN_hidden = np.array([np.maximum(0, x) for x in in_for_relu])
    FFN_output = np.array([x @ W2 + b2 for x in FFN_hidden])

    if not os.path.exists(gamma_file1) and not os.path.exists(beta_file1):
        gamma1 = np.random.uniform(0.9, 1.1, size=(embedding_dim,))
        beta1 = np.random.uniform(-0.1, 0.1, size=(embedding_dim,))
        np.save(gamma_file1, gamma1)
        np.save(beta_file1, beta1)
    # Add & Norm 2
    Layer_add_fnn = X + FFN_output
    Layer_norm_fnn = ([LayerNormalisation(Layer_add_fnn[i], gamma1 , beta1) for i in range(len(Layer_add_fnn))])

    # Sentence vector & prediction
    sentence_vector = Layer_norm_fnn[-1]
    input_words = set(a_words)

    V1 = np.array([word_to_vec[word] for word in vocab])  # Shape: (20, d)
    logits = V1 @ sentence_vector  # Shape: (20,)
    logits = logits - np.max(logits)
    probs = softmax(logits)  # Shape: (20,)
    target_index = vocab.index(output1[inpput])
    loss = -np.log(probs[target_index])

    dlogits = probs.copy()
    dlogits[target_index] -= 1  # shape: (20,)

    dsentence_vector = V1.T @ dlogits  # shape: (d,)
    dV1 = np.outer(dlogits, sentence_vector)  # shape: (20, d)

    pred_index = np.argmax(probs)
    predicted_word = vocab[pred_index]
    print(loss, end=" ")
    print(predicted_word,end =" : ")
    print(output1[inpput])
    
    def layernorm_fnn_training(srr, gamma_1, beta_1,dlbydout,choice):
        E = 1e-9
        mean = np.mean(srr)
        variance = np.var(srr)
        if choice == 0:
            for i in range(len(gamma_1)):
                gamma_1[i] = gamma_1[i] - clip_gradient(dlbydout) * (learningrate * ((srr[i] - mean) / (np.sqrt(variance + E))))
                beta_1[i] = beta_1[i] - clip_gradient(dlbydout) * (learningrate * beta_1[i])
            np.save(gamma_file1, gamma_1)
            np.save(beta_file1, beta_1)
        elif choice == 1:
            for i in range(len(gamma_1)):
                for j in range(len(dlbydout)):
                    gamma_1[i] = gamma_1[i] - clip_gradient(dlbydout[j]) * (learningrate * ((srr[i] - mean) / (np.sqrt(variance + E))))
                    beta_1[i] = beta_1[i] - clip_gradient(dlbydout[j]) * (learningrate * beta_1[i])
            np.save(gamma_file, gamma_1)
            np.save(beta_file, beta_1)   

    def W2_training(dffnout):
        for i in range(len(FFN_output)):
            for j in range(len(W2)):
                for k in range(len(W2[0])):
                     dL_dffnout_i1 = clip_gradient(dl_by_dfnnout[i][k] )     # ∂L/∂FFN_output[i][1]
                     h_i0 = FFN_hidden[i][j]                  # FFN_hidden[i][0] is the input to W2[0][1]
                     grad = dL_dffnout_i1 * h_i0              # chain rule
                     W2[j][k] -= learningrate * grad
                     b2[k] -= learningrate * dL_dffnout_i1
                     #print(W2[j][k])
        np.save(ffn_w2_file, W2)
        np.save(ffn_b2_file, b2)
        
    def W1_training(dffnhidden):
        for i in range(len(dffnhidden)):
            for j in range(len(W1)):
                for k in range(len(W1[0])):
                     dL_dffnout_i1 = clip_gradient(dffnhidden[i][k])       # ∂L/∂FFN_output[i][1]
                     h_i0 = Layer_norm[i][j]                  # FFN_hidden[i][0] is the input to W2[0][1]
                     grad = dL_dffnout_i1 * h_i0              # chain rule
                     W1[j][k] -= learningrate * grad
                     b1[k] -= learningrate * dL_dffnout_i1
                     #print(W2[j][k])
        np.save(ffn_w1_file, W1)
        np.save(ffn_b1_file, b1)
    dl_by_dout = -(1/(probs[target_index]*np.log(10)))
    dl_by_dlogits = probs - one_hot[target_index]
    dl_by_dsentence = V1.T @ dl_by_dlogits
    dl_by_dffn_layernorm = dl_by_dsentence * 1/len(a_words)
    
    def W0_training(dattentionout,w0):
        dl_by_dw0 = attention.T @ clip_gradient(dattentionout)
        w0 -= learningrate * dl_by_dw0
        np.save(paths["W0"], w0)

    def attention_weights_training(dattention,dscores,wv,wq,wk):
        wv -= learningrate * clip_gradient(dl_by_dWV)
        wq -= learningrate * clip_gradient(dl_by_dWQ)
        wk -= learningrate * clip_gradient(dl_by_dWK)
        np.save(paths["WV"], wv)
        np.save(paths["WK"], wk)
        np.save(paths["WQ"], wq)        
    #print(W2)
    dffn_layernorm_by_dfnn_layeradd = np.array([LayerNormalisation(Layer_add_fnn[i], gamma1 , beta1 ,1) for i in range(len(Layer_add_fnn))])
    #dl_by_dfnn_layeradd = np.array([i @ dl_by_dffn_layernorm for i in dffn_layernorm_by_dfnn_layeradd])
    dl_by_dfnn_layeradd = dffn_layernorm_by_dfnn_layeradd * dl_by_dffn_layernorm
    dl_by_dfnnout = dl_by_dfnn_layeradd * 1
    dl_by_dfnnhidden = dl_by_dfnnout @ W2.T
    dl_by_drelu = dl_by_dfnnhidden * (FFN_hidden > 0).astype(float)
    dl_by_dlayernorm = (dl_by_drelu @ W1.T) * 1/len(a_words)
    dlayernorm_by_dlayeradd = np.array([LayerNormalisation(Layer_add[i], gamma , beta ,1) for i in range(len(Layer_add))])
    dl_by_dlayeradd = dl_by_dlayernorm * dlayernorm_by_dlayeradd
    dl_by_dattentionout = dl_by_dlayeradd * 1
    dl_by_dattention = dl_by_dattentionout @ W0.T
    dl_by_dweights = dl_by_dattention @ V.T
    dot = np.sum(dl_by_dweights * weights, axis=1, keepdims=True)
    dl_by_dscores = weights * (dl_by_dweights - dot)
    dl_by_dV = weights.T @ dl_by_dattention
    dl_by_dQ = (dl_by_dscores @ K)  * (1/np.sqrt(len(a_words)))
    dl_by_dK = (dl_by_dscores.T @ Q) * (1/np.sqrt(len(a_words)))
    dl_by_dWV = X.T @ dl_by_dV
    dl_by_dWQ = X.T @ dl_by_dQ
    dl_by_dWK = X.T @ dl_by_dK
    dL_by_dX = clip_gradient(dl_by_dQ) @ WQ.T + clip_gradient(dl_by_dK) @ WK.T + clip_gradient(dl_by_dV) @ WV.T
    #print(dl_by_drelu)
    #print(len(W2))
    #print(dl_by_drelu)
    #print()
    #print(Layer_norm)
    #print()
    #print(W1)
    #print()
    for i in range(len(Layer_norm_fnn)):
          layernorm_fnn_training(Layer_norm_fnn[i] , gamma1 , beta1 , dl_by_dffn_layernorm[i],0)
    W2_training(dl_by_dfnnout)
    W1_training(dl_by_drelu)
    for i in range(len(Layer_norm)):
          layernorm_fnn_training(Layer_norm[i],gamma,beta,dl_by_drelu[i],1)
    W0_training(dl_by_dattentionout,W0)
    attention_weights_training(dl_by_dattention,dl_by_dscores,WV,WQ,WK)
    for j, token_index in enumerate(input_ids):
        word_to_vec[vocab[token_index]] -= learningrate * dL_by_dX[j]
    np.save(file_path_embeddings, word_to_vec)
    #print(W2)
    #print(counting)
    #counting = counting + 1
    #print()
    #print(dl_by_dfnn_layeradd)
    #for i in range(len(Layer_add_fnn)): layernorm_fnn_training(Layer_add_fnn[i],gamma1,beta1,dl_by_dout)