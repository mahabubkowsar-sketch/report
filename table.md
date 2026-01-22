## Table VI. ABLATION STUDY OF THE PROPOSED MULTIMODAL EMPATHETIC DETECTION MODEL

| Variant | Preprocessing | Attention Module | Fusion Strategy | Weighted Loss | F1 | Accuracy |
|---------|---------------|------------------|------------------|---------------|----|---------| 
| Baseline CNN | ✗ | ✗ | ✗ | ✗ | 0.62 | 65.0% |
| + Preprocessing | ✓ | ✗ | ✗ | ✗ | 0.82 | 83.8% |
| + Attention | ✓ | ✓ | ✗ | ✗ | 0.84 | 86.8% |
| + Fusion | ✓ | ✓ | ✓ | ✗ | 0.87 | 90.7% |
| + Weighted Loss | ✓ | ✓ | ✓ | ✓ | 0.89 | 91.2% |
| **Full Model** | **✓** | **✓** | **✓** | **✓** | **0.99** | **98.6%** |
