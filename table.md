# Ablation Study

## Definition and Methodology

An ablation study in deep learning refers to a systematic analysis used to evaluate the importance of various components, features, or modules within a model.

**Purpose:**
1. To identify the relative importance of each component of a system.
2. To verify whether certain design choices are critical to the model's performance or if they could be simplified or removed.

**How It Works:**
1. Selectively remove or modify specific parts of the model (e.g., layers, features, regularizers, or optimization strategies).
2. They then retrain or evaluate the model on the same task to see how performance changes.
3. Performance metrics (e.g., accuracy, loss, F1 score) are compared before and after the ablation.

**Common Scenarios:**
1. **Removing Features:** Analyzing the contribution of specific input features to the model's predictions.
2. **Simplifying Architecture:** Removing certain layers or connections to determine their significance.
3. **Testing Modules:** Disabling individual modules (e.g., attention mechanisms, dropout layers) to evaluate their role.
4. **Examining Hyperparameters:** Modifying or removing specific techniques (e.g., weight decay, learning rate schedules) to assess their impact.

## Table VI. ABLATION STUDY OF THE PROPOSED MULTIMODAL EMPATHETIC DETECTION MODEL

| Variant | Architecture | Modalities | Fusion Strategy | F1-Score | Accuracy | Precision |
|---------|-------------|------------|------------------|----------|----------|-----------|
| **Audio-Text Model** | BERT + Wav2Vec2 + Metadata | Audio + Text | Early Fusion | **82.37%** | **83.15%** | **82.76%** |
| Video Model | Video Transformer + Text | Video + Text | Late Fusion | 69.84% | 72.24% | 72.31% |
| Late Fusion | Combined Models | Video + Audio + Text | Late Fusion (89.5% Video, 10.5% Audio-Text) | 55.98% | 60.42% | 60.39% |
| Deep Fusion | MIMAMO Architecture | Video + Audio + Text | Deep Integration | 56.12% | 58.04% | 57.89% |

**Test Samples:** 4,939 across all experiments  
**Note:** Results ordered by F1-Score performance

## Analysis of Results

### Key Findings from Experimental Results

**1. Audio-Text Model Dominance (F1: 82.37%, Accuracy: 83.15%)**
- **Architecture:** BERT + Wav2Vec2 + Metadata fusion
- **Best Performance:** Achieves highest scores across all metrics
- **Insight:** Early fusion of audio and text with metadata provides optimal results
- **Training:** Well-optimized with checkpoint `best_7class_model.pth`

**2. Video Model Performance (F1: 69.84%, Accuracy: 72.24%)**
- **Architecture:** Video Transformer with text encoding
- **Moderate Performance:** Second-best individual modality
- **Gap Analysis:** 12.53% F1-score gap from best model
- **Potential:** Room for improvement in video feature extraction

**3. Fusion Method Challenges**

**Late Fusion Results (F1: 55.98%, Accuracy: 60.42%)**
- **Strategy:** Combined models with 89.5% video weight, 10.5% audio-text weight
- **Issue:** Performance degradation despite multimodal approach
- **Analysis:** Suboptimal weight distribution or conflicting modality signals
- **Training:** Only 14 epochs, potentially undertrained

**Deep Fusion Results (F1: 56.12%, Accuracy: 58.04%)**  
- **Strategy:** MIMAMO architecture with deep integration
- **Challenge:** Lowest performance despite sophisticated architecture
- **Parameters:** Only 9 trainable parameters (2 weights + 7 bias) - severely limited
- **Insight:** Insufficient model capacity for complex multimodal learning

### Critical Insights

**1. Modality Effectiveness Ranking:**
   - Audio-Text (Early Fusion): 82.37% F1
   - Video + Text: 69.84% F1  
   - Multimodal Fusion: 55.98-56.12% F1

**2. Fusion Paradox:**
   - Individual modalities outperform fusion approaches
   - Suggests modality interference or suboptimal fusion strategies
   - Need for better fusion architectures

**3. Training Parameter Impact:**
   - Deep fusion severely limited by only 9 trainable parameters
   - Late fusion may need more training epochs (only 14 completed)
   - Audio-text model benefits from full parameter training

### Performance Analysis by Architecture

| Aspect | Audio-Text | Video Model | Late Fusion | Deep Fusion |
|--------|------------|-------------|-------------|-------------|
| **Complexity** | High | Medium | High | Medium |
| **Parameters** | Full training | Full training | Limited | 9 trainable |
| **Training** | Optimized | Stable | 14 epochs | Constrained |
| **F1 Score** | **82.37%** | 69.84% | 55.98% | 56.12% |
| **Effectiveness** | Excellent | Good | Poor | Poor |

### Key Findings

1. **Audio-Text Modality Dominance:** The BERT + Wav2Vec2 + Metadata combination significantly outperforms other approaches, suggesting that linguistic and acoustic cues are highly informative for empathetic detection.

2. **Fusion Strategy Challenges:** Contrary to typical multimodal expectations, fusion approaches underperformed individual modalities, indicating potential issues with:
   - Modality alignment and synchronization
   - Feature space compatibility
   - Fusion weight optimization
   - Training methodology

3. **Parameter Efficiency Issues:** Deep fusion's limitation to only 9 trainable parameters severely constrains learning capacity, explaining poor performance despite sophisticated architecture.

4. **Video Modality Potential:** While video alone achieves 69.84% F1, there's significant room for improvement in visual feature extraction and temporal modeling.

5. **Training Optimization:** The audio-text model's superior performance suggests better hyperparameter tuning and training convergence compared to fusion approaches.

## Conclusions

1. **Best Practice Identification:** Early fusion with comprehensive feature integration (audio + text + metadata) proves most effective for empathetic detection.

2. **Fusion Complexity:** Simple fusion strategies may not capture complex inter-modal relationships needed for emotion recognition.

3. **Architecture Optimization:** Individual modality optimization should precede multimodal fusion development.

4. **Resource Allocation:** Focus development resources on improving individual modality performance before attempting complex fusion.

## Recommendations

1. **Immediate Actions:**
   - Deploy audio-text model as primary system (83.15% accuracy)
   - Investigate video model improvements to bridge 12.53% performance gap
   - Redesign fusion strategies with increased parameter capacity

2. **Future Research:**
   - Develop advanced fusion architectures with attention mechanisms between modalities
   - Investigate cross-modal pre-training strategies
   - Explore temporal alignment techniques for video-audio synchronization
   - Implement adaptive fusion weights based on input characteristics

3. **Technical Improvements:**
   - Increase deep fusion trainable parameters significantly
   - Extend late fusion training beyond 14 epochs
   - Optimize video transformer architecture for emotion-specific features
   - Implement modality-specific preprocessing enhancements
  
   - | Variant | Description | Components | Preprocessing | Attention | Fusion | Weighted Loss | Accuracy | F1 Score |
|---------|-------------|------------|---------------|-----------|---------|--------------|----------|----------|
| **Full Model (Enhanced)** | Complete model with all enhancements | Video, Audio, Text | ✓ | ✓ | Late Enhanced | ✓ | **0.9862** | **0.9850** |
| **Video Only** | Video modality with basic preprocessing | Video | ✓ | ✗ | ✗ | ✗ | 0.9862 | 0.9850 |
| **Multimodal (No Weighted Loss)** | Full multimodal without weighted loss | Video, Audio, Text | ✓ | ✓ | Late | ✗ | 0.9073 | 0.8865 |
| **Multimodal (No Attention)** | Full multimodal without attention | Video, Audio, Text | ✓ | ✗ | Late | ✓ | 0.8679 | 0.8372 |
| **Audio-Text** | Audio and text fusion | Audio, Text | ✓ | ✓ | Early | ✓ | 0.8315 | 0.8200 |
| **Video + Basic Fusion** | Video with simple multimodal fusion | Video, Text | ✓ | ✗ | Simple | ✗ | 0.8383 | 0.8077 |
| **Baseline CNN** | Simple CNN baseline | Video | ✗ | ✗ | ✗ | ✗ | 0.6500 | 0.6200 |
| **MIMAMO Net (Optimized)** | MIMAMO architecture with optimization | Video, Audio, Text | ✓ | ✓ | MIMAMO | ✓ | 0.6273 | 0.6215 |

