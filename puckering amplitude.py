import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os

def best_fit_plane(coords):
    centroid = np.mean(coords, axis=0)
    centered = coords - centroid
    u, s, vh = np.linalg.svd(centered)
    normal = vh[2]
    return normal, centroid

def compute_z_deviations(coords, normal, center):
    z = [np.dot(coord - center, normal) for coord in coords]
    return np.array(z)

def describe_mode(k, qk, Q):
    ratio = qk / Q if Q != 0 else 0
    if ratio > 0.4:
        strength = "Dominant"
        if k == 1:
            shape = "Global bending"
        elif k == 2:
            shape = "Half-wave distortion (e.g., boat or twist)"
        elif k == 3:
            shape = "Tri-lobed twist or saddle"
        else:
            shape = f"{k}-fold alternating distortion"
    elif ratio > 0.15:
        strength = "Moderate"
        shape = f"{k}-fold puckering feature"
    else:
        strength = "Minor"
        shape = "Weak contribution"
    return strength, shape

def fourier_decomposition(z_devs):
    N = len(z_devs)
    Q = np.linalg.norm(z_devs)
    q_ks = []
    for k in range(1, N // 2 + 1):
        q_cos = (2 / N) * sum(z_devs[i] * np.cos(2 * np.pi * k * i / N) for i in range(N))
        q_sin = (2 / N) * sum(z_devs[i] * np.sin(2 * np.pi * k * i / N) for i in range(N))
        q_k = np.sqrt(q_cos**2 + q_sin**2)
        phi_k = np.degrees(np.arctan2(q_sin, q_cos)) % 360
        q_ks.append((k, q_k, phi_k))
    return Q, q_ks

def analyze_ring(coords):
    normal, center = best_fit_plane(coords)
    z = compute_z_deviations(coords, normal, center)
    Q, q_ks = fourier_decomposition(z)
    return Q, q_ks, z

def plot_ring(z_devs, ring_index, atom_labels, save_prefix):
    N = len(z_devs)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)
    z = z_devs

    x = np.append(x, x[0])
    y = np.append(y, y[0])
    z = np.append(z, z[0])
    atom_labels = atom_labels + [atom_labels[0]]

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, marker='o', linestyle='-', color='black', label=f'Ring {ring_index+1}')
    
    for i in range(N):
        ax.text(x[i], y[i], z[i] + 0.15, f"{z[i]:+.2f} Å", color='darkred', fontsize=10, ha='center')
        ax.text(x[i], y[i], z[i] - 0.1, atom_labels[i], color='navy', fontsize=9, ha='center')

    ax.set_title(f'3D Puckering: Ring {ring_index+1}')
    ax.set_xlabel("X (ring plane)")
    ax.set_ylabel("Y (ring plane)")
    ax.set_zlabel("Z (out-of-plane)")
    ax.view_init(elev=20, azim=120)
    plt.tight_layout()

    filename = f"{save_prefix}_ring{ring_index+1}.png"
    plt.savefig(filename, dpi=300)
    print(f"Plot saved as: {filename}")
    plt.close()

def save_z_deviation_dat(z_devs, atom_labels, ring_index, prefix):
    filename = f"{prefix}_ring{ring_index+1}.dat"
    with open(filename, "w") as f:
        f.write("# AtomIndex\tAtomLabel\tZ_Deviation(Å)\n")
        for i, (label, zval) in enumerate(zip(atom_labels, z_devs)):
            f.write(f"{i}\t{label}\t{zval:+.4f}\n")
    print(f"Z-deviation data saved as: {filename}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python puckering_analysis_named_output.py input.mol")
        sys.exit(1)

    molfile = sys.argv[1]
    if not os.path.isfile(molfile):
        print(f"File not found: {molfile}")
        sys.exit(1)

    base_name = os.path.splitext(os.path.basename(molfile))[0]

    mol = Chem.MolFromMolFile(molfile, removeHs=False)
    if mol is None:
        print("Could not read molecule.")
        sys.exit(1)

    if mol.GetNumConformers() == 0:
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        AllChem.UFFOptimizeMolecule(mol)

    conf = mol.GetConformer()
    rings = mol.GetRingInfo().AtomRings()
    if not rings:
        print("No rings found.")
        return

    print(f"Total rings found: {len(rings)}\n")

    for i, ring in enumerate(rings):
        coords = np.array([conf.GetAtomPosition(idx) for idx in ring])
        atom_labels = [f"{mol.GetAtomWithIdx(idx).GetSymbol()}{idx}" for idx in ring]
        Q, q_ks, z = analyze_ring(coords)

        print(f"Ring {i+1} (Size {len(ring)}):")
        print(f"  Q (total puckering amplitude): {Q:.4f} Å")
        print(f"  z-deviations: {[f'{val:.3f}' for val in z]}")
        print(f"  Fourier mode amplitudes (q_k) and phase angles (φ_k):")
        for k, qk, phik in q_ks:
            strength, shape = describe_mode(k, qk, Q)
            print(f"    k={k}: q_k = {qk:.4f} Å, φ_k = {phik:.2f}° — {strength}: {shape}")
        print("-" * 40)

        # Save image and data with base name
        plot_ring(z, i, atom_labels, save_prefix=base_name)
        save_z_deviation_dat(z, atom_labels, i, prefix=base_name)

if __name__ == "__main__":
    main()
