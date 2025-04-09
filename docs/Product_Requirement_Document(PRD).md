PRD: AFNet – Electrogram-based Wavefront Prediction System
🔍 1. Project Overview
AFNet is an AI-based system designed to model atrial electrograms (EGMs) and predict the spatial propagation of electrical activity in the heart. Given 19-channel atrial electrograms, the model outputs:

Activation Map (2D wavefront propagation field)

Activation Times (per-electrode timing)

Propagation Direction Vectors (per-electrode spatiotemporal flow)

The system includes synthetic data generation based on real signal segments, a deep learning model for training, and visualization tools for interpreting the predicted wavefronts.

👤 2. Target Users & Use Case Flow
👩‍🔬 Target Users:
Biomedical engineering researchers

AI + cardiology graduate students

EP (Electrophysiology) labs and research centers

📈 Workflow Overview:
Synthesize electrogram data from real peak windows (with controlled wavefront types: focal, rotor, planar)

Train a deep model (ResNet-based) to jointly predict activation maps and directions

Evaluate performance using physical losses (directional consistency, MAE)

Visualize wavefronts and predicted propagation vectors

[Optional future] Apply to real-patient data for research/clinical hypothesis validation

🧩 3. Core Features
Module	Description
✅ Data Augmentation	Augment real electrograms to simulate various wavefront types
✅ Synthetic Generator	Generate clean activation maps from virtual source points
✅ Deep Model Training	Extract temporal-spatial features and regress multiple outputs
✅ Directional Supervision	Compute per-electrode direction vectors based on activation time gradients
✅ Custom Loss Functions	Multi-objective: activation map MSE + activation time MSE + vector cosine loss
✅ Visualization	Matplotlib-based visualization of wavefronts and direction vectors
✅ Logging	Training logs via TensorBoard and CSV; best model checkpoint saving
🔒 4. Out-of-Scope
The following are not included in this stage of development:

Web-based GUI or front-end for interactive usage

Real-time API or cloud deployment

Cross-patient generalization (personalized modeling in future roadmap)

Clinical integration with EP systems or mapping devices

⚙️ 5. Tech Stack
Component	Tools / Libraries
Language	Python 3.9+
ML Framework	PyTorch
Signal Processing	NumPy, SciPy, Hilbert transform, Gaussian filter
Data Handling	pandas, h5py
Model Backbone	ResNet18 (customized input channels)
Visualization	Matplotlib, TensorBoard
IDE / Dev Env	Spyder, CLI, Git (project versioning)
📁 6. Project Structure
text
Copy
Edit
AFNew/
├── config/           ← Global constants and training config
├── data/             ← Raw and generated EGM data
├── models/           ← Model architecture and weights
├── loss/             ← Custom loss functions
├── scripts/          ← Training, prediction, and analysis scripts
├── results/          ← Training logs, scores, and outputs
├── logs/             ← TensorBoard logs
└── docs/             ← Documentation
✅ 7. Success Criteria
Metric	Target
Activation Time MAE	≤ 0.05
Activation Map MAE	≤ 0.05
Direction Vector Similarity (cosine)	≥ 0.90
Visualization Quality	Interpretability and spatial coherence