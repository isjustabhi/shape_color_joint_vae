# Joint Generative Model (P(shape, color))

This project implements a **Joint VAE** to model both shape and color jointly as RGB images.

---

##  Features
- Trains a convolutional VAE on colored shape images
- Learns the full joint distribution P(shape, color)
- Outputs side-by-side reconstructions over epochs

---

##  Project Structure
- `scripts/generate_dataset.py`: Creates RGB toy dataset
- `vae_dataset.py`: Dataset loader
- `vae_model.py`: ConvVAE model
- `scripts/train_joint_vae.py`: Training loop
- `scripts/view_results.ipynb`: Visualize reconstructions
- `models/`: Stores trained weights
- `samples/joint/`: Stores reconstruction samples

---

##  How to Run
```bash
# 1. Generate dataset
python scripts/generate_dataset.py

# 2. Train VAE
python scripts/train_joint_vae.py

# 3. Visualize in Jupyter
jupyter notebook scripts/view_results.ipynb
