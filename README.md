# **Joint VAE Model – P(shape, color)**  

This project implements a **Joint Variational Autoencoder (VAE)** that learns the full joint distribution of **shapes and colors** from a synthetic RGB dataset. It is part of a larger exploration of **compositional generative modeling**, inspired by the paper:  
**"Compositional Generative Modeling: A Single Model is Not All You Need"**.  

---

## ✅ **Project Overview**
- **Objective:** Model P(shape, color) directly with a single VAE.
- **Dataset:** Synthetic RGB images of geometric shapes (circle, square, triangle) in {red, green, blue}.
- **Model:** Convolutional VAE with latent dimension = 32.

---

## ✅ **Key Files**
- `scripts/generate_dataset.py` → Generate synthetic RGB dataset.
- `vae_model.py` → Convolutional VAE architecture.
- `vae_dataset.py` → PyTorch dataset loader.
- `scripts/train_joint_vae.py` → Training script for the Joint VAE.
- `scripts/view_results.ipynb` → Visualization notebook for reconstructions.
- `samples/joint/` → Input and reconstructed outputs across epochs.
- `models/joint_vae.pth` → Trained model weights.

---

## ✅ **Results (100 Epochs)**
The model improves significantly with training:

### **Epoch 10**
- Outputs are blurry and mostly grayscale.
- Shape structure is weakly captured.

![Epoch 10 Output](<img width="266" height="134" alt="image" src="https://github.com/user-attachments/assets/dde13d6e-def9-4ab3-99c1-da71ea4c75bd" />
)

---

### **Epoch 50**
- Shapes are more distinct, basic colors appear faintly.

*(Add image if available)*

---

### **Epoch 100**
- Clear shapes, accurate red and blue colors.
- Green improved but still less consistent.
- Minor color bleeding on edges.

![Epoch 100 Output](<img width="266" height="134" alt="image" src="https://github.com/user-attachments/assets/aca7e71a-f975-4eb4-9ad6-ffe3eaef89df" />
)

---

**Observation:**  
The Joint VAE successfully learns the joint distribution P(shape, color), but struggles with green consistency due to possible class imbalance or limited latent capacity.

---


## ✅ **How to Run**
```bash
# 1. Generate dataset
python scripts/generate_dataset.py

# 2. Train the model
python scripts/train_joint_vae.py

# 3. Visualize results
jupyter notebook scripts/view_results.ipynb
