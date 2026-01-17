# Algorithm 1: Proposed Cross-Modal Attention-Based Emotion Recognition Model

## Input:
- Text input sequence $T = \{t_1, t_2, \ldots, t_n\}$
- Audio input sequence $A = \{a_1, a_2, \ldots, a_m\}$

## Output:
- Predicted emotion class $\hat{y}$ and cross-modal attention weights

## Algorithm Steps:

### 1. Base Feature Extraction:
**Step 2:** Initialize a pre-trained BERT model for text feature extraction

**Step 3:** Initialize a pre-trained Wav2Vec 2.0 model for audio feature extraction

**Step 4:** Extract contextual text embeddings
$$T_{\text{raw}} = \text{BERT}(T)$$

**Step 5:** Extract acoustic embeddings
$$A_{\text{raw}} = \text{Wav2Vec2}(A)$$

### 6. Cross-Modal Attention Setup:
**Step 7:** Initialize projection matrices
$$W_Q^t, W_K^t, W_V^t \in \mathbb{R}^{768 \times d_k}$$
$$W_Q^a, W_K^a, W_V^a \in \mathbb{R}^{768 \times d_k}$$

**Step 8:** Set number of attention heads $(h)$ and head dimension $(d_k)$

### 9. Text-to-Audio Attention Computation:
**Step 10:** for each text token $t_i \in T_{\text{raw}}$ do

**Step 11:** &nbsp;&nbsp;&nbsp;&nbsp;Compute query vector $Q_t = W_Q^t t_i$

**Step 12:** &nbsp;&nbsp;&nbsp;&nbsp;for each audio frame $a_j \in A_{\text{raw}}$ do

**Step 13:** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Compute key $K_a = W_K^a a_j$ and value $V_a = W_V^a a_j$

**Step 14:** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Compute attention score
$$e_{ta}(i,j) = \frac{Q_t^T K_a}{\sqrt{d_k}}$$

**Step 15:** &nbsp;&nbsp;&nbsp;&nbsp;end for

**Step 16:** &nbsp;&nbsp;&nbsp;&nbsp;Apply softmax normalization to obtain attention weights $\alpha_{ta}(i,:)$

**Step 17:** &nbsp;&nbsp;&nbsp;&nbsp;Compute attended audio feature
$$A_{\text{att}}(i) = \sum_j \alpha_{ta}(i,j) V_a$$

**Step 18:** end for

### 19. Audio-to-Text Attention Computation:
**Step 20:** for each audio frame $a_i \in A_{\text{raw}}$ do

**Step 21:** &nbsp;&nbsp;&nbsp;&nbsp;Compute query vector $Q_a = W_Q^a a_i$

**Step 22:** &nbsp;&nbsp;&nbsp;&nbsp;for each text token $t_j \in T_{\text{raw}}$ do

**Step 23:** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Compute key $K_t = W_K^t t_j$ and value $V_t = W_V^t t_j$

**Step 24:** &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Compute attention score
$$e_{at}(i,j) = \frac{Q_a^T K_t}{\sqrt{d_k}}$$

**Step 25:** &nbsp;&nbsp;&nbsp;&nbsp;end for

**Step 26:** &nbsp;&nbsp;&nbsp;&nbsp;Apply softmax normalization to obtain attention weights $\alpha_{at}(i,:)$

**Step 27:** &nbsp;&nbsp;&nbsp;&nbsp;Compute attended text feature
$$T_{\text{att}}(i) = \sum_j \alpha_{at}(i,j) V_t$$

**Step 28:** end for

### 29. Multi-Level Feature Fusion:
**Step 30:** Apply mean pooling to obtain fixed-length representations

**Step 31:** Concatenate attended and raw features
$$H_{\text{fused}} = \text{Concat}(T_{\text{att}}, A_{\text{att}}, T_{\text{raw}}, A_{\text{raw}})$$

### 32. Classification:
**Step 33:** Pass fused representation through fully connected layers

**Step 34:** Apply ReLU activation and dropout regularization

**Step 35:** Compute final emotion logits

**Step 36:** Predict emotion class
$$\hat{y} = \arg\max(\text{Softmax}(H_{\text{fused}}))$$

### 37. Result:
**Step 38:** Return predicted emotion label, fused features, and cross-modal attention maps

---

## Algorithm 2: Adaptive Gated Fusion for Multimodal Emotion Recognition

### Input:
- Multimodal dataset consisting of video $(V)$, audio $(A)$, text $(T)$, and user metadata $(M)$
- Ground truth emotion labels $(Y)$

### Output:
- Predicted emotion label $\hat{y}$ and optimized fusion parameters $\theta^*$

---

### 1: Pretrained Encoder Initialization:
**Step 2:** Initialize BERT for text feature extraction

**Step 3:** Initialize Wav2Vec 2.0 for audio feature extraction

**Step 4:** Initialize Transformer-based video encoder (MIMAMO Net)

**Step 5:** Initialize metadata encoder

**Step 6:** Freeze all pretrained encoders

---

### 7: Feature Extraction:
**Step 8:** Extract text embeddings
$$h_t = \text{BERT}(T)$$

**Step 9:** Extract audio embeddings
$$h_a = \text{Wav2Vec2}(A)$$

**Step 10:** Extract video embeddings
$$h_v = \text{VideoEncoder}(V)$$

**Step 11:** Extract metadata embeddings
$$m = \text{MetadataEncoder}(M)$$

---

### 12: Adaptive Gated Fusion Setup:
**Step 13:** Initialize learnable gating parameters $(W_i, b_i)$ for each modality

**Step 14:** Initialize fusion weights $(W_f)$ and bias $(b_{\text{bias}})$

---

### 15: Adaptive Gate Computation:
**Step 16:** for each modality $i \in \{t, a, v\}$ do

**Step 17:** &nbsp;&nbsp;&nbsp;&nbsp;Compute gate value
$$g_i = \sigma \left( W_i [h_i ; m] + b_i \right)$$

**Step 18:** end for

**Step 19:** Normalize gate values using softmax
$$\alpha_i = \frac{e^{g_i}}{\sum_j e^{g_j}}$$

---

### 20: Feature Fusion:
**Step 21:** Compute fused multimodal representation
$$h_{\text{fusion}} = \sum_i \alpha_i h_i$$

**Step 22:** Apply enhanced fusion with learnable scaling
$$H = \text{softmax}(W_f) \odot [h_v, h_a, h_t] + b_{\text{bias}}$$

---

### 23: Classification:
**Step 24:** Pass fused representation through fully connected layers

**Step 25:** Compute emotion logits

**Step 26:** Predict emotion class
$$\hat{y} = \arg\max(\text{Softmax}(H))$$

---

### 27: Training Strategy:
**Step 28:** Compute classification loss using ground truth labels

**Step 29:** Update fusion weights and gating parameters using AdamW optimizer

---

### 30: Result:
**Step 31:** Return trained Adaptive Gated Fusion model, predicted emotion labels, and learned modality importance weights
