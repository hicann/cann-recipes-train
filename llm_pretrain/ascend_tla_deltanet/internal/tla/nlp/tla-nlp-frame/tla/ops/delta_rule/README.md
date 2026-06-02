
## 🚀 Kernel Performance Comparison (Native vs. Triton)

**Device:** Ascend 910B (1x)  | 
**Settings:** $B=32, H=8, D=64$

| Seq Len ($L$) | Stage | `torch_npu` (ms)  | `Triton` (ms) |                   Speedup                   |
|:-------------:| :--- |:-----------------:|:-------------:|:-------------------------------------------:|
|    **512**    | Forward |      21.0553      |    20.7116     |   <span style="color: green">1.02x</span>   |
|               | Backward |      61.3314      |    34.3343     |   <span style="color: green">1.79x</span>   |
|               | **Total** |    **83.1223**    |  **55.2134**   | <span style="color: green">**1.51x**</span> |
|   **1024**    | Forward |      32.4127      |    40.8766     |   <span style="color: green">0.79x</span>   |
|               | Backward |     106.4374      |    67.6445   |   <span style="color: green">1.57x</span>   |
|               | **Total** |   **139.6989**    |  **108.6874**  | <span style="color: green">**1.29x**</span> |
|   **2048**    | Forward |      55.1317      |    81.0733     |   <span style="color: green">0.68x</span>   |
|               | Backward |      246.4703      |    134.2998    |   <span style="color: green">1.84x</span>   |
|               | **Total** |    **302.7201**    |  **215.5629**  | <span style="color: green">**1.40x**</span> |
|   **4096**    | Forward |      109.4693      |    161.6783     |   <span style="color: green">0.68x</span>   |
|               | Backward |     744.5641      |    267.6126    |   <span style="color: green">2.78x</span>   |
|               | **Total** |   **855.8185**    |  **429.5763**  | <span style="color: green">**1.99x**</span> |


