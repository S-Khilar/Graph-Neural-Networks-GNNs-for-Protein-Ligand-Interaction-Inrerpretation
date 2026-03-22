import streamlit as st
import torch
import tempfile
import requests
import plotly.graph_objects as go

from model import ProteinLigandGNN
from graph_utils import protein_to_graph, mol_to_graph


# ------------------------
# Page Configuration
# ------------------------
st.set_page_config(
    page_title="Protein–Ligand GNN Predictor",
    page_icon="🔬",
    layout="centered"
)

st.title("🔬 Protein–Ligand Binding Affinity Predictor")
st.markdown("**AI-powered prediction of pKd, pKi and Binding Affinity**")
st.markdown(
    "<p style='font-size:9px; color:green;'>By Subhasankar Khilar</p>",
    unsafe_allow_html=True
)
st.markdown("<br><br>", unsafe_allow_html=True)


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