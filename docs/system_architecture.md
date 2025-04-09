🧠 System Architecture Documentation

🎯 Project Goal

Build a modular, extensible deep learning system that predicts:

Activation map from electrogram data

Per-electrode propagation direction vectors

The system is trained on synthetic electrogram data derived from real waveform segments and can be evaluated and validated on real electrogram recordings.

📐 System Architecture Overview

🔧 Modules:

1. data/ — Data Generation & Augmentation

generate_electrogram.py: Unified generator for focal, rotor, and planar wavefronts

augment_real_data.py: Augments and normalizes real peak-window signals

dataset_loader.py: PyTorch Dataset for loading .h5 format electrogram data

Output data format: 19-channel electrograms + GT activation map + GT activation times (for direction)

2. models/ — Network Architectures

afnet.py: ResNet-based dual-output model

Outputs:

Activation map [B, 100, 100]

Activation time per electrode [B, 19]

Shared feature encoder, split heads

Future: Extendable to GNN or Transformer architectures

3. loss/ — Custom Losses

directional_loss.py: Cosine-similarity loss for direction vector alignment

Combined total loss = activation_map_loss + activation_time_loss + directional_loss

Future: Support physics-informed Eikonal loss or velocity-consistency penalty

4. utils/ — Utilities

Signal preprocessing: signal_processing.py

Visualization: vis_activation_map.py, vis_direction_vector.py

5. scripts/ — Main Training & Execution

train_on_synthesis.py: Train model on synthetic data with directional loss

predict_and_vis.py: Predict from real data and visualize activation/direction

real_data_aug_generate.py: Synthesize train/val set in .h5 format

6. config/ — Parameter Configurations

constants.py: Paths, grid size, sampling parameters, electrode layout

train_config.py: Training parameters (batch size, LR, epochs, device)

Optional: Future support for default.yaml config files

7. results/, models/, logs/

results/: CSV logs, visual outputs per experiment

models/: Saved model checkpoints (best & final)

logs/: TensorBoard logs per experiment run

🔁 Data Flow

real_data_aug_generate.py

Uses real waveform segments → generates synthetic wavefront data

Saves to .h5 including: electrograms, activation_map, activation_times

train_on_synthesis.py

Loads .h5 via ElectrogramDataset

Trains model with combined losses

Outputs model checkpoints + metrics logs

predict_and_vis.py (future extension)

Loads real .mat waveform windows

Predicts activation + direction

Visualizes wavefront propagation

💡 Extensibility Design

Future Feature

Module Impact

➕ Add Transformer/GNN

models/afnet.py or new module

➕ Add velocity-based loss

loss/directional_loss.py extension

➕ Real-to-synthetic domain adaptation

New data/domain_adapt.py

➕ GUI-based result visualization

utils/visualizer_gui.py

➕ Clinical data loader (CSV/EDF)

data/dataset_loader.py extension

📊 Evaluation Pipeline (Planned)

Save direction prediction as .npy

Use quiver plot to overlay GT vs predicted directions

Quantify angular error per electrode

Visualize activation time RMSE heatmap

Export result images to results/experiment_name/








AFNew deep learning AF mapping system UML

─────────────────────────────────────────────
Class: AFNetResNet (torch.nn.Module)
─────────────────────────────────────────────
+ resnet : Sequential                       ← ResNet18 backbone
+ shared_fc : Linear                        ← shared 512→1024
+ shared_relu : ReLU
+ activation_time_fc : Linear               ← output activation time (19)
+ activation_fc : Linear                    ← output activation map (100×100)

Methods:
+ forward(x) → activation_map, activation_times

Used by:
→ train() and evaluate() in train_on_synthesis.py
─────────────────────────────────────────────


─────────────────────────────────────────────
Class: ElectrogramDataset (torch.utils.data.Dataset)
─────────────────────────────────────────────
Attributes:
+ files : List[str]                         ← paths to .h5 files
+ label_map : Dict[str, int]                ← mapping type → label

Methods:
+ __getitem__(idx) → (electrograms, activation_map, activation_times)

Used by:
→ train_loader, validation_loader in train_on_synthesis.py
─────────────────────────────────────────────


─────────────────────────────────────────────
Function: fast_compute_direction_vectors
─────────────────────────────────────────────
Inputs:
+ activation_times: Tensor[B, 19]
+ coords: Tensor[19, 2]

Returns:
+ direction_vectors: Tensor[B, 19, 2]

Used by:
→ train(), evaluate() for direction supervision
─────────────────────────────────────────────


─────────────────────────────────────────────
Function: directional_vector_loss
─────────────────────────────────────────────
Inputs:
+ pred_vectors: Tensor[B, 19, 2]
+ true_vectors: Tensor[B, 19, 2]

Returns:
+ loss: scalar (1 - cosine similarity)

Used by:
→ train(), evaluate()
─────────────────────────────────────────────


─────────────────────────────────────────────
Script: train_on_synthesis.py
─────────────────────────────────────────────
Calls:
→ ElectrogramDataset (for data loading)
→ AFNetResNet (model creation)
→ directional_vector_loss
→ fast_compute_direction_vectors

Logs:
→ TensorBoard log
→ CSV log in results/
→ model checkpoints in models/

Dependencies:
← config/constants.py (paths, electrode layout)
← config/train_config.py (hyperparams)
─────────────────────────────────────────────














