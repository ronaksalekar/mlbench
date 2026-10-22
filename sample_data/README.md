# Sample Datasets for MLBench

This folder contains sample benchmark datasets for testing and demonstrating MLBench.

## 1. Parkinson's Disease Voice Dataset (`parkinsons_sample.csv`)

This dataset is based on the landmark **UCI Machine Learning Repository Parkinson's Disease Dataset** created by Max Little et al. (Oxford University) in collaboration with the National Centre for Voice and Speech, Denver, Colorado.

### Dataset Specifications:
- **Task:** Binary Classification
- **Observations:** Voice recordings from multiple subjects (each subject provided 6 phonations of sustained vowel 'a').
- **Target Column:** `status`
  - `1`: Diagnosed with Parkinson's Disease (PD)
  - `0`: Healthy Control (HC)
- **Subject Column:** `name` (e.g., `phon_R01_S01_1`, where `phon_R01_S01` denotes Subject 1, recording 1).

### Feature Dictionary:
- `name`: ASCII subject name and recording number
- `MDVP:Fo(Hz)`: Average vocal fundamental frequency
- `MDVP:Fhi(Hz)`: Maximum vocal fundamental frequency
- `MDVP:Flo(Hz)`: Minimum vocal fundamental frequency
- `MDVP:Jitter(%)`, `MDVP:Jitter(Abs)`, `MDVP:RAP`, `MDVP:PPQ`, `Jitter:DDP`: Measures of frequency variation
- `MDVP:Shimmer`, `MDVP:Shimmer(dB)`, `Shimmer:APQ3`, `Shimmer:APQ5`, `MDVP:APQ`, `Shimmer:DDA`: Measures of amplitude variation
- `NHR`, `HNR`: Ratio of noise to tonal components in the voice
- `status`: Health status (1 = Parkinson's, 0 = Healthy)
- `RPDE`, `D2`: Recurrence period density entropy and correlation dimension (nonlinear complexity measures)
- `DFA`: Detrended fluctuation analysis (signal fractal scaling exponent)
- `spread1`, `spread2`, `PPE`: Pitch period entropy and fundamental frequency variation

### Educational & Laboratory Demonstration Note:
In laboratory experiments, students must exclude the `name` column from feature inputs (as it is an identifier) and select `status` as the target column. Furthermore, students can test both **Standard Stratified Split** and **Subject-Level Grouped Split** to observe how intra-subject recording correlation affects validation accuracy.
