# 🧬 Graph Neural Networks for Protein-Ligand Interaction Interpretation

## 📌 Overview
This project focuses on predicting protein–ligand binding interactions using Graph Neural Networks (GNNs). It aims to model molecular structures as graphs and learn interaction patterns for drug discovery applications.

Protein–ligand interactions are fundamental in drug design, as they determine how a drug molecule binds to its target protein.

---

## 🎯 Objectives
- Predict binding affinity between protein and ligand
- Represent molecules as graphs (nodes = atoms, edges = bonds/interactions)
- Capture spatial and chemical relationships using GNN
- Interpret important features influencing binding

---

## 🧠 Methodology

### 🔹 Graph Representation
- Protein and ligand converted into graph structures
- Nodes: atoms with features (type, charge, hybridization)
- Edges: bonds or distance-based interactions

### 🔹 Model
- Graph Neural Network (GNN)
- Message Passing mechanism
- Fully connected layers for prediction

GNNs work by passing information between nodes to learn structural relationships.

---

## 🛠️ Tech Stack
- Python
- PyTorch
- PyTorch Geometric
- RDKit
- NumPy, Pandas

---

## 📊 Dataset
- PDBBind dataset (or custom dataset)
- Contains protein-ligand complexes with binding affinity values

---

## 📈 Results
(*Update after training*)

- RMSE:  1.424
- MAE:  1.126
- R² Score:  0.458
- PR–AUC:  0.765
- Regression Accuracy (%):  45.782681979346826
- pKd Classification Accuracy:  0.7470238095238095
- Precision:  0.7553816046966731
- Recall:  0.5298558682223747
- F1-score:  0.6228317870108915

---

## 🔍 Interpretation
- Model interpretability using GNNExplainer / attention
- Identifies important atoms and interactions

---

## 📁 Project Structure
├── data/
├── models/
├── scripts/
├── notebooks/
├── results/

---

## 🚀 How to Run

``bash
git clone https://github.com/S-Khilar/Graph-Neural-Networks-GNNs-for-Protein-Ligand-Interaction-Inrerpretation.git

pip install -r requirements.txt

python train.py

## Future Work
Improve model accuracy
Add attention-based GNN
Deploy as web application
Integrate with drug discovery pipeline

## Author
Subhasankar Khilar
Interested in AI for Drug Discovery

<img width="280" height="227" align="left" alt="Binding Afinity" src="https://github.com/user-attachments/assets/5ad28a8d-c4e0-48ee-b8ea-46745bb579c5" />

<img width="280" height="227" align="right" alt="ROC Curve" src="https://github.com/user-attachments/assets/13db8b61-d5d6-4478-be1f-f2bd0201f4ab" />

