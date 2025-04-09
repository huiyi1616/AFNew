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



