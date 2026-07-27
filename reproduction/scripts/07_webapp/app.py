"""
app.py — CytoSafe web server

Accepts a SMILES string, runs the 3T3 and HEK293 LGBM models, and returns:
  1. Prediction (Toxic / Non-toxic) + confidence for each cell line
  2. Similarity map heatmap (Riniker & Landrum atom-contribution method)
     red = promotes toxicity, green = reduces toxicity
  3. Top-10 ECFP4 bit contributions with substructure drawings

Start: ./start_web_server.sh
"""

import base64
import io
import os
import warnings

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify
from joblib import load
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Draw
from rdkit.Chem.Draw import SimilarityMaps
from rdkit.Chem import rdMolDescriptors

warnings.filterwarnings('ignore')

app = Flask(__name__)

# ── Model paths ───────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, '..', '..', 'model')
MODELS     = {}

def load_models():
    for cell_line in ('3T3', 'HEK293'):
        path = os.path.join(MODEL_DIR, f'{cell_line}_lgbm_ecfp4.joblib')
        if os.path.exists(path):
            MODELS[cell_line] = load(path)
            print(f'Loaded model: {cell_line}')
        else:
            print(f'WARNING: model not found at {path}')


# ── Fingerprint helpers ───────────────────────────────────────────────────────
NBITS  = 1024
RADIUS = 2

def mol_to_fp_array(mol):
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, RADIUS, NBITS)
    arr = np.zeros((NBITS,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


# ── Atom contributions (Riniker & Landrum 2013) ───────────────────────────────

def atom_contributions(model, mol):
    """
    For each atom: zero out all ECFP4 bits that atom participates in,
    measure ΔP(cytotoxic). Positive weight → atom promotes toxicity prediction.
    Returns (weights list, base_prob).
    """
    bit_info = {}
    base_fp = AllChem.GetMorganFingerprintAsBitVect(mol, RADIUS, NBITS, bitInfo=bit_info)
    base_arr = np.zeros((NBITS,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(base_fp, base_arr)
    base_prob = float(model.predict_proba(base_arr.reshape(1, -1))[0, 1])

    # atom → set of bits it participates in
    atom_bits = {a.GetIdx(): set() for a in mol.GetAtoms()}
    for bit, entries in bit_info.items():
        for atom_idx, _ in entries:
            atom_bits[atom_idx].add(bit)

    weights = []
    for atom_idx in range(mol.GetNumAtoms()):
        bits = atom_bits.get(atom_idx, set())
        if not bits:
            weights.append(0.0)
            continue
        perturbed = base_arr.copy()
        for b in bits:
            perturbed[b] = 0
        perturbed_prob = float(model.predict_proba(perturbed.reshape(1, -1))[0, 1])
        weights.append(base_prob - perturbed_prob)

    return weights, base_prob


# ── Bit contributions split into positive / negative ─────────────────────────

def top_bit_contributions(model, mol, top_n=5):
    """
    For each ECFP4 bit set in the molecule, flip it to 0 and measure ΔP(cytotoxic).
    Returns:
      pos_bits: top_n bits with positive delta (sorted high→low) — support toxicity
      neg_bits: top_n bits with negative delta (sorted by abs value high→low) — support non-toxic
    """
    bit_info = {}
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, RADIUS, NBITS, bitInfo=bit_info)
    base_arr = np.zeros((NBITS,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, base_arr)
    base_prob = float(model.predict_proba(base_arr.reshape(1, -1))[0, 1])

    contributions = []
    for bit in bit_info.keys():
        perturbed = base_arr.copy()
        perturbed[bit] = 0
        delta = base_prob - float(model.predict_proba(perturbed.reshape(1, -1))[0, 1])
        contributions.append((bit, delta, bit_info[bit]))

    pos_bits = sorted([c for c in contributions if c[1] > 0], key=lambda x: -x[1])[:top_n]
    neg_bits = sorted([c for c in contributions if c[1] < 0], key=lambda x: x[1])[:top_n]
    return pos_bits, neg_bits, base_prob


def draw_bit_substructure(mol, bit, bit_info_entry, direction='toxic'):
    """
    Draw the full molecule with the atoms of the ECFP4 bit's circular
    environment highlighted. Red highlight = toxic-supporting bit,
    green = non-toxic-supporting bit.
    """
    highlight_color = (0.99, 0.60, 0.60) if direction == 'toxic' else (0.60, 0.90, 0.66)

    try:
        atom_idx, radius = bit_info_entry[0]

        # collect all atoms in the circular environment
        if radius == 0:
            highlight_atoms = [atom_idx]
            highlight_bonds = []
        else:
            env_bonds = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_idx)
            highlight_atoms = set()
            highlight_bonds = list(env_bonds)
            for bond_idx in env_bonds:
                bond = mol.GetBondWithIdx(bond_idx)
                highlight_atoms.add(bond.GetBeginAtomIdx())
                highlight_atoms.add(bond.GetEndAtomIdx())
            highlight_atoms = list(highlight_atoms)

        atom_colors = {a: highlight_color for a in highlight_atoms}
        bond_colors = {b: highlight_color for b in highlight_bonds}

        img = Draw.MolToImage(
            mol,
            size=(220, 160),
            highlightAtoms=highlight_atoms,
            highlightBonds=highlight_bonds,
            highlightAtomColors=atom_colors,
            highlightBondColors=bond_colors,
        )
    except Exception:
        img = Draw.MolToImage(mol, size=(220, 160))

    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()



# ── SHAP-style bar chart → base64 PNG ────────────────────────────────────────

def shap_bar_chart_b64(pos_bits, neg_bits):
    """
    Horizontal bar chart of ΔP for the 5 positive + 5 negative bits.
    Sorted by abs(ΔP) high → low (matching bit card order below).
    Positive bars → red (right), negative bars → green (left).
    Returns base64-encoded PNG.
    """
    all_bits = pos_bits + neg_bits
    # sort by abs(delta) descending — top bar = highest |ΔP|
    all_bits = sorted(all_bits, key=lambda x: -abs(x[1]))

    labels = [f'Bit {b[0]}' for b in all_bits]
    deltas = [b[1] for b in all_bits]
    colors = ['#fc8181' if d > 0 else '#68d391' for d in deltas]

    n = len(all_bits)
    fig, ax = plt.subplots(figsize=(5, max(2.5, n * 0.42)))
    bars = ax.barh(range(n - 1, -1, -1), deltas, color=colors,
                   edgecolor='white', height=0.6)
    ax.set_yticks(range(n))
    ax.set_yticklabels(list(reversed(labels)), fontsize=8)
    ax.axvline(0, color='#4a5568', linewidth=0.8, linestyle='--')
    ax.set_xlabel('ΔP (cytotoxic)', fontsize=8)
    ax.set_title('Bit contribution (ΔP) — sorted by |ΔP|', fontsize=9, fontweight='bold')

    for bar, val in zip(bars, deltas):
        x = bar.get_width()
        offset = 0.0003 if x >= 0 else -0.0003
        ax.text(x + offset, bar.get_y() + bar.get_height() / 2,
                f'{val:+.4f}', va='center',
                ha='left' if x >= 0 else 'right',
                fontsize=7, color='#2d3748')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=130, bbox_inches='tight')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


# ── Similarity map → base64 PNG ───────────────────────────────────────────────

def similarity_map_b64(model, mol, cell_line):
    weights, base_prob = atom_contributions(model, mol)
    max_w = max(abs(w) for w in weights) if any(w != 0 for w in weights) else 1.0
    norm_weights = [w / max_w if max_w > 0 else 0.0 for w in weights]

    pred_label = 'Cytotoxic' if base_prob >= 0.5 else 'Non-cytotoxic'

    # RDKit >= 2022 requires a MolDraw2D object instead of a matplotlib ax
    from rdkit.Chem.Draw import rdMolDraw2D
    width, height = 500, 400
    drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    SimilarityMaps.GetSimilarityMapFromWeights(
        mol, norm_weights, drawer,
        colorMap='RdYlGn_r',
        alpha=0.5,
    )
    drawer.FinishDrawing()
    png_bytes = drawer.GetDrawingText()

    # Overlay title using matplotlib (composite onto RDKit PNG)
    img = plt.imread(io.BytesIO(png_bytes))
    fig, ax = plt.subplots(figsize=(width / 100, (height + 40) / 100), dpi=100)
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(f'{cell_line}  |  P(cytotoxic) = {base_prob:.3f}  →  {pred_label}',
                 fontsize=10, fontweight='bold', pad=4)
    plt.tight_layout(pad=0.2)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode(), base_prob, pred_label


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    smiles = (request.form.get('smiles') or '').strip()
    if not smiles:
        return render_template('index.html', error='Please enter a SMILES string.')

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return render_template('index.html',
                               error=f'Could not parse SMILES: {smiles}',
                               smiles=smiles)

    results = {}

    for cell_line, model in MODELS.items():
        # --- similarity map ---
        sim_map_b64, prob, pred_label = similarity_map_b64(model, mol, cell_line)

        # --- bit contributions split into pos / neg ---
        pos_bits, neg_bits, _ = top_bit_contributions(model, mol, top_n=5)

        def make_cards(bit_list):
            cards = []
            for bit, delta, bit_info_entry in bit_list:
                direction = 'toxic' if delta > 0 else 'nontoxic'
                subst_img = draw_bit_substructure(mol, bit, bit_info_entry, direction)
                cards.append({
                    'bit':       bit,
                    'delta':     round(delta, 4),
                    'direction': direction,
                    'img_b64':   subst_img,
                })
            return cards

        cards_pos = make_cards(pos_bits)
        cards_neg = make_cards(neg_bits)
        shap_chart = shap_bar_chart_b64(pos_bits, neg_bits)

        results[cell_line] = {
            'prob':       round(prob, 4),
            'pred_label': pred_label,
            'sim_map':    sim_map_b64,
            'shap_chart': shap_chart,
            'pos_bits':   cards_pos,
            'neg_bits':   cards_neg,
        }

    return render_template('index.html',
                           smiles=smiles,
                           results=results)


if __name__ == '__main__':
    load_models()
    app.run(host='0.0.0.0', port=5050, debug=False)
