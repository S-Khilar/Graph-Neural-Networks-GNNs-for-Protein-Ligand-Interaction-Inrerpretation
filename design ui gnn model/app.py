import streamlit as st
import torch
import tempfile
import requests
import plotly.graph_objects as go

from model import ProteinLigandGNN
from graph_utils import protein_to_graph, mol_to_graph


# =========================================================
# Page Configuration — MUST COME FIRST
# =========================================================

st.set_page_config(
    page_title="GNN-PLIP | Protein-Ligand Interaction Predictor",
    page_icon="🔬",
    layout="centered"
)


# =========================================================
# GNN-PLIP — Animated Scientific Background
# =========================================================

st.markdown("""
<style>

.stApp {
    background:
        radial-gradient(
            circle at 10% 20%,
            rgba(0, 210, 255, 0.18),
            transparent 32%
        ),
        radial-gradient(
            circle at 85% 15%,
            rgba(140, 80, 255, 0.18),
            transparent 32%
        ),
        radial-gradient(
            circle at 75% 80%,
            rgba(0, 255, 200, 0.12),
            transparent 30%
        ),
        radial-gradient(
            circle at 40% 55%,
            rgba(255, 0, 170, 0.07),
            transparent 35%
        ),
        linear-gradient(
            135deg,
            #020617 0%,
            #07152b 45%,
            #030712 100%
        );
}

/* Animated scientific grid */

.stApp::before {

    content: "";

    position: fixed;

    inset: 0;

    pointer-events: none;

    z-index: 0;

    background-image:
        linear-gradient(
            rgba(80,180,255,0.035) 1px,
            transparent 1px
        ),
        linear-gradient(
            90deg,
            rgba(80,180,255,0.035) 1px,
            transparent 1px
        );

    background-size: 45px 45px;

    animation: gridMove 20s linear infinite;
}


@keyframes gridMove {

    from {
        transform: translate(0, 0);
    }

    to {
        transform: translate(45px, 45px);
    }

}


/* Floating molecular nodes */

.gnn-node {

    position: fixed;

    width: 7px;

    height: 7px;

    border-radius: 50%;

    background: #5ee7ff;

    box-shadow:
        0 0 10px #5ee7ff,
        0 0 25px rgba(94,231,255,0.6);

    opacity: 0.7;

    pointer-events: none;

    z-index: 0;

    animation:
        floatNode 5s ease-in-out infinite;
}


.node1 {
    top: 18%;
    left: 8%;
}

.node2 {
    top: 35%;
    right: 10%;
    animation-delay: 2s;
}

.node3 {
    top: 70%;
    left: 12%;
    animation-delay: 4s;
}

.node4 {
    top: 80%;
    right: 15%;
    animation-delay: 0.5s;
}

.node5 {
    top: 70%;
    left: 12%;
    animation-delay: 0.1s;
}


@keyframes floatNode {

    0%, 100% {
        transform: translate(0, 0);
        opacity: 0.35;
    }

    50% {
        transform: translate(20px, -25px);
        opacity: 0.9;
    }

}


/* Keep Streamlit content above background */

[data-testid="stAppViewContainer"] {
    position: relative;
    z-index: 1;
}

[data-testid="stHeader"] {
    background: transparent;
}

/* Glassmorphism cards */

.gnn-card {
    background: rgba(15, 25, 45, 0.72);

    border: 1px solid rgba(100, 200, 255, 0.18);

    border-radius: 18px;

    padding: 25px;

    margin: 20px 0;

    backdrop-filter: blur(14px);

    -webkit-backdrop-filter: blur(14px);

    box-shadow:
        0 10px 35px rgba(0,0,0,0.35),
        inset 0 1px 0 rgba(255,255,255,0.05);
}

</style>


<div class="gnn-node node1"></div>
<div class="gnn-node node2"></div>
<div class="gnn-node node3"></div>
<div class="gnn-node node4"></div>

""", unsafe_allow_html=True)




# ------------------------
# Page Configuration
# ------------------------


st.markdown("""
<div style="
    text-align:center;
    padding:35px 10px 25px 10px;
">

<h1 style="
    font-size:52px;
    font-weight:800;
    margin-bottom:5px;
    letter-spacing:2px;
">
🔬 GNN-PLIP
</h1>

<h3 style="
    font-weight:400;
    opacity:0.85;
">
Graph Neural Network–Based Protein-Ligand Interaction Predictor
</h3>

<p style="
    font-size:16px;
    opacity:0.65;
">
AI-powered prediction of pKd, pKi and binding affinity
</p>

<p style="
    font-size:13px;
    opacity:0.55;
">
Research application based on Graph Neural Networks
</p>

<p style="
    font-size:10px;
    opacity:0.45;
">
Developed by Subhasankar Khilar
</p>
</div>
""", unsafe_allow_html=True)


st.markdown("<br>", unsafe_allow_html=True)

device = torch.device("cpu")


# ------------------------
# Load Model
# ------------------------
@st.cache_resource
def load_model():
    model = ProteinLigandGNN(hidden_dim=128)
    model.load_state_dict(torch.load("model_weights..pth", map_location=device))
    model.eval()
    return model

model = load_model()


# ------------------------
# Inputs
# ------------------------
st.subheader("🧪 Ligand Input")
smiles = st.text_input("Enter Ligand SMILES")

st.subheader("🧬 Protein Input")

input_method = st.radio(
    "Choose Protein Input Method:",
    ["Upload PDB File", "Enter PDB ID"]
)

pdb_file = None
pdb_id = None

if input_method == "Upload PDB File":
    pdb_file = st.file_uploader("Upload Protein PDB File", type=["pdb"])
else:
    pdb_id = st.text_input("Enter PDB ID (e.g., 1HSG)")


# ------------------------
# Prediction
# ------------------------
if st.button("🚀 Predict"):

    if smiles == "":
        st.warning("Please enter SMILES string")
        st.stop()

    try:

        # Handle Protein Input
        if input_method == "Upload PDB File":

            if pdb_file is None:
                st.warning("Please upload PDB file")
                st.stop()

            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(pdb_file.read())
                pdb_path = tmp.name

        else:
            if pdb_id == "":
                st.warning("Please enter PDB ID")
                st.stop()

            pdb_id = pdb_id.upper()
            url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
            response = requests.get(url)

            if response.status_code != 200:
                st.error("Invalid PDB ID ❌")
                st.stop()

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdb") as tmp:
                tmp.write(response.content)
                pdb_path = tmp.name

            st.success(f"PDB {pdb_id} downloaded successfully ✅")

        # Convert to Graph
        protein_graph = protein_to_graph(pdb_path)
        ligand_graph = mol_to_graph(smiles)

        protein_graph.batch = torch.zeros(
            protein_graph.num_nodes, dtype=torch.long
        )
        ligand_graph.batch = torch.zeros(
            ligand_graph.num_nodes, dtype=torch.long
        )

        # Run Model
        with st.spinner("Running GNN model... ⏳"):
            with torch.no_grad():
                pkd, pki, ba = model(protein_graph, ligand_graph)

        st.success("Prediction completed 🎉")

        pkd_val = pkd.item()
        pki_val = pki.item()
        ba_val = ba.item()

        col1, col2, col3 = st.columns(3)
        col1.metric("pKd", f"{pkd_val:.3f}")
        col2.metric("pKi", f"{pki_val:.3f}")
        col3.metric("Binding Affinity", f"{ba_val:.3f}")
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("<br><br>", unsafe_allow_html=True)

       

        # ------------------------
        # Fixed 3D-Style Bar Plot (Non-Rotatable)
        # ------------------------
        st.subheader("📊 Binding Affinity Profile")
        import plotly.graph_objects as go

        labels = ["pKd", "pKi", "Binding Affinity"]
        values = [pkd_val, pki_val, ba_val]
        colors = ["#E63946", "#2A9D8F", "#457B9D"]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=labels,
            y=values,
            marker=dict(
                color=colors,
                line=dict(color="black", width=1.5)
            ),
            hovertemplate="<b>%{x}</b><br>Value: %{y:.3f}<extra></extra>"
        ))
        fig.update_layout(
            template="plotly_dark",
            title="Protein–Ligand Binding Prediction",
            xaxis=dict(title=""),
            yaxis=dict(title="Predicted Value"),
            margin=dict(l=40, r=40, t=60, b=40),
            bargap=0.4
        )
        
        # Add slight 3D effect illusion
        fig.update_traces(
            marker=dict(
                color=colors,
                line=dict(color="rgba(0,0,0,0.6)", width=2)
            )
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("<br><br>", unsafe_allow_html=True)



        # ------------------------
        # Professional 3D Plot
        # ------------------------
        st.subheader("📊 3D Prediction Visualization")

        labels = ["pKd", "pKi", "Binding Affinity"]
        values = [pkd_val, pki_val, ba_val]

        fig = go.Figure()

        fig.add_trace(go.Scatter3d(
            x=[0, 1, 2],
            y=[0, 0, 0],
            z=values,
            mode='markers+text',
            marker=dict(
                size=12,
                color=values,
                colorscale='Viridis',
                opacity=0.9
            ),
            text=labels,
            textposition="top center"
        ))

        fig.update_layout(
            template="plotly_dark",
            scene=dict(
                xaxis=dict(
                    tickvals=[0, 1, 2],
                    ticktext=labels,
                    title=""
                ),
                yaxis=dict(title=""),
                zaxis=dict(title="Predicted Value")
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )

        st.plotly_chart(fig, use_container_width=True)



    except Exception as e:
     st.error(f"Error occurred: {str(e)}")
 #this is end
