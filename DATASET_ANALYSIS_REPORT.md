# Empathetic Detection Dataset - Comprehensive Analysis Report

## Executive Summary

This report provides a comprehensive analysis of the empathetic detection dataset used for emotion recognition in conversational AI. The dataset includes **56,780 emotion-labeled samples** derived from **24,696 conversations**, with corresponding video and audio files.

---

## 1. Dataset Overview

### Key Metrics

| Metric | Value |
|--------|-------|
| **Total Conversations** | 24,696 |
| **Total Emotion-Labeled Samples** | 56,780 |
| **Unique Emotion Classes** | 32 |
| **Unique Conversation Topics** | 10 |
| **Video Files** | 226,484 |
| **Audio Files** | 226,484 |
| **Combined Dataset Size** | 42.44 GB |
| **Average Turns per Conversation** | 2.30 |

### Data Format

The dataset is stored in JSON format with the following structure:
- **Conversation ID**: Unique identifier for each conversation
- **Speaker/Listener Profile**: Demographics (age, gender, timbre) and participant ID
- **Topic**: Conversation subject matter
- **Turns**: List of dialogue turns, each containing:
  - Context and dialogue history
  - Listener's response
  - Chain of Empathy with speaker emotion annotation

---

## 2. Emotion Class Distribution

### Overall Distribution

The dataset contains **32 distinct emotion classes** with significant class imbalance:

| Rank | Emotion | Count | Percentage | Augmentation Factor |
|------|---------|-------|-----------|-------------------|
| 1 | **anxious** | 18,737 | 33.00% | 1.00x |
| 2 | **sad** | 5,915 | 10.42% | 3.17x |
| 3 | **annoyed** | 2,394 | 4.22% | 7.83x |
| 4 | **surprised** | 1,654 | 2.91% | 11.33x |
| 5 | **lonely** | 1,403 | 2.47% | 13.35x |
| 6-32 | *Other emotions* | 24,682 | ~43.48% | 16-31x |

### Class Imbalance Analysis

- **Maximum class size**: 18,737 (anxious)
- **Minimum class size**: 605 (faithful)
- **Class imbalance ratio**: **30.97x**
- **Standard deviation**: 3,225.81
- **Average class size**: 1,774.4

### Top 10 Emotions

```
1. anxious (18,737)        - 33.00%
2. sad (5,915)             - 10.42%
3. annoyed (2,394)         - 4.22%
4. surprised (1,654)       - 2.91%
5. lonely (1,403)          - 2.47%
6. excited (1,236)         - 2.18%
7. nostalgic (1,185)       - 2.09%
8. proud (1,152)           - 2.03%
9. angry (1,146)           - 2.02%
10. afraid (1,137)         - 2.00%
```

---

## 3. Conversation Topics

### Topic Distribution

The dataset covers **10 conversation topics** with the following distribution:

| Rank | Topic | Count | Percentage |
|------|-------|-------|-----------|
| 1 | **Personal Struggles and Challenges** | 31,387 | 55.28% |
| 2 | **Life Events** | 9,281 | 16.35% |
| 3 | **Emotions and Feelings** | 3,650 | 6.43% |
| 4 | **Interpersonal Relationships** | 3,091 | 5.44% |
| 5 | **Achievements and Self-Realization** | 2,948 | 5.19% |
| 6 | **Health and Well-being** | 2,679 | 4.72% |
| 7 | **Support and Comfort** | 1,904 | 3.35% |
| 8 | **Disappointments and Expectations** | 1,240 | 2.18% |
| 9 | **Social Issues and Moral Dilemmas** | 306 | 0.54% |
| 10 | **Uncertainty About the Future** | 294 | 0.52% |

**Insight**: "Personal Struggles and Challenges" dominates the dataset (55%), indicating that the conversations heavily focus on emotional struggles and personal challenges.

---

## 4. Data Augmentation Impact

### Augmentation Strategy

Data augmentation is applied to balance the dataset:
- **Video Augmentation**: Rotation, flipping, brightness/contrast adjustment
- **Audio Augmentation**: Time stretching, pitch shifting, noise injection
- **Target**: Balance minority classes to match majority class size (18,737 samples)

### Pre vs Post-Augmentation Comparison

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| **Total Samples** | 56,780 | 599,584 | +542,804 |
| **Dataset Size Increase** | - | - | **+956.0%** |
| **Class Balance Ratio** | 30.97x | 1.00x | Perfectly Balanced |
| **Samples per Class** | 605-18,737 | 18,737 (all) | Uniform |

### Augmentation by Class

Minority classes receive the most augmentation:
- **Faithful** (minority): +18,132 new samples (30.97x augmentation)
- **Apprehensive**: +17,973 new samples (24.52x augmentation)
- **Anxious** (majority): 0 new samples (already at target size)

---

## 5. Media Files

### Video Properties

- **Total video files**: 226,484
- **Average file size**: ~0.08-0.28 MB
- **Total size**: 13.19 GB
- **File format**: MP4
- **Naming convention**: `dia[CONVERSATION_ID]utt[TURN_ID]_[EMOTION_CODE].mp4`

### Audio Properties

- **Total audio files**: 226,484
- **Total size**: 29.25 GB
- **File format**: WAV
- **File naming**: Matches video files with .wav extension

---

## 6. Key Insights & Findings

### Dataset Characteristics

1. **Highly Imbalanced**: The dataset shows significant class imbalance with "anxious" emotion being 30.97x more prevalent than "faithful"

2. **Multimodal Rich**: Each sample includes synchronized video and audio, enabling multimodal emotion recognition research

3. **Large Scale**: With 56,780 samples across 226,484 video-audio files, the dataset is substantial for deep learning

4. **Topic-Specific**: The emotion-topic distribution shows that different topics correlate with different emotions (e.g., "Achievements" correlates with excited/proud emotions)

5. **Augmentation-Ready**: Significant class imbalance provides clear justification for data augmentation strategies

---

## 7. Recommended Training Practices

### For Model Development

1. **Use Augmented Dataset**: Leverage augmented data for balanced training
2. **Stratified Splitting**: Implement stratified k-fold cross-validation by emotion class
3. **Weighted Loss**: Consider weighted loss functions accounting for original class distribution
4. **Macro-Averaging**: Use macro-averaged metrics (F1, precision, recall) for balanced evaluation

### For Data Processing

1. **Feature Extraction**: 
   - Extract facial features from video frames (facial landmarks, AUs)
   - Extract acoustic features from audio (MFCCs, spectrograms, prosody)

2. **Temporal Dynamics**: Model temporal patterns in both modalities

3. **Fusion Strategy**: Implement late, early, or intermediate fusion approaches

### For Evaluation

1. **Confusion Analysis**: Analyze commonly confused emotion pairs
2. **Emotion-Specific Analysis**: Evaluate performance by emotion class
3. **Topic-Specific Performance**: Assess model performance across different topics
4. **Baseline Comparison**: Compare with single-modality baselines

---

## 8. Generated Analysis Files

This analysis generated the following files:

### Visualizations
- `emotion_distribution_analysis.png` - Comprehensive emotion class distribution charts (4 subplots)
- `augmentation_comparison.png` - Pre vs post-augmentation comparison (6 subplots)
- `topic_emotion_distribution.png` - Topic distribution and emotion-topic heatmap

### Data Tables (CSV)
- `dataset_summary.csv` - Overall dataset statistics
- `class_distribution_before_augmentation.csv` - Detailed class distribution before augmentation
- `class_distribution_after_augmentation.csv` - Detailed class distribution after augmentation

### Interactive Analysis
- `Dataset_Analysis.ipynb` - Complete Jupyter notebook with all analysis and code

---

## 9. Conclusion

The empathetic detection dataset is a comprehensive, multimodal resource for emotion recognition research with:

- **56,780 emotion-labeled samples** across 32 emotion classes
- **226,484 synchronized video-audio file pairs** (42.44 GB)
- **10 diverse conversation topics** with emotion context
- **Significant class imbalance** necessitating augmentation strategies
- **High potential** for multimodal fusion research

This dataset supports research in:
- Emotion recognition
- Empathetic AI
- Multimodal learning
- Data augmentation techniques
- Imbalanced dataset handling

---

## Document Information

- **Generated**: January 13, 2026
- **Dataset Version**: v5.0
- **Analysis Tool**: Python with Pandas, Matplotlib, Seaborn
- **Location**: `d:\NSU SEMESTER\10th Semester\CSE499B(RTK)\empathetic-detection-main`
