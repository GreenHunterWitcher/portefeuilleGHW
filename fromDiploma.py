import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.interpolate import make_interp_spline
import MDAnalysis as mda
from MDAnalysis.analysis import rms, align, pca
#import prody as pd
import os

# Настройка стиля графиков
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")

# Вектор временной шкалы для 100 нс
md_step_ns = np.arange(0, 100, 0.1)

# Загрузка RMSD файлов
wt_rmsd = pd.read_csv("wt-trajrmsd.dat", sep="\s+", header=0)
k137e_rmsd = pd.read_csv("k137e-trajrmsd.dat", sep="\s+", header=0)
k141e_rmsd = pd.read_csv("k141e-trajrmsd.dat", sep="\s+", header=0)

# Построение графиков RMSD
plt.figure(figsize=(10, 6))
plt.plot(md_step_ns, wt_rmsd.iloc[:, 1], color='blue', label='WT')
plt.plot(md_step_ns, k137e_rmsd.iloc[:, 1], color='red', label='K137E')
plt.plot(md_step_ns, k141e_rmsd.iloc[:, 1], color='darkgreen', label='K141E')

# Сглаживание LOWESS
from statsmodels.nonparametric.smoothers_lowess import lowess

wt_smooth = lowess(wt_rmsd.iloc[:, 1], md_step_ns, frac=0.1)
k137e_smooth = lowess(k137e_rmsd.iloc[:, 1], md_step_ns, frac=0.1)
k141e_smooth = lowess(k141e_rmsd.iloc[:, 1], md_step_ns, frac=0.1)

plt.plot(wt_smooth[:, 0], wt_smooth[:, 1], '--', color='blue', linewidth=3, alpha=0.7)
plt.plot(k137e_smooth[:, 0], k137e_smooth[:, 1], '--', color='red', linewidth=3, alpha=0.7)
plt.plot(k141e_smooth[:, 0], k141e_smooth[:, 1], '--', color='darkgreen', linewidth=3, alpha=0.7)

plt.title("HSPB8")
plt.xlabel("Time, 100ns")
plt.ylabel("RMSD (Å)")
plt.legend()
plt.show()

# Загрузка структур PDB с использованием MDAnalysis
wt_universe = mda.Universe("WT/protein/ionized_Hspb8.pdb", "WT/tr/tr_Hspb8_GMD.dcd")
k137e_universe = mda.Universe("K137E/protein/ionized_K137E.pdb", "K137E/tr/tr_K137E.dcd")
k141e_universe = mda.Universe("K141E/protein/ionized_K141E.pdb", "K141E/tr/tr_K141E.dcd")

# Выбор CA атомов
wt_ca = wt_universe.select_atoms("protein and name CA")
k137e_ca = k137e_universe.select_atoms("protein and name CA")
k141e_ca = k141e_universe.select_atoms("protein and name CA")

# Выравнивание траекторий
align.AlignTraj(wt_universe, wt_universe, select="protein and name CA", in_memory=True).run()
align.AlignTraj(k137e_universe, k137e_universe, select="protein and name CA", in_memory=True).run()
align.AlignTraj(k141e_universe, k141e_universe, select="protein and name CA", in_memory=True).run()

# Расчет RMSD
wt_rmsd_calc = rms.RMSD(wt_universe, wt_universe, select="protein and name CA", ref_frame=0).run()
k137e_rmsd_calc = rms.RMSD(k137e_universe, k137e_universe, select="protein and name CA", ref_frame=0).run()
k141e_rmsd_calc = rms.RMSD(k141e_universe, k141e_universe, select="protein and name CA", ref_frame=0).run()

wt_rd = wt_rmsd_calc.rmsd[:, 2]
k137e_rd = k137e_rmsd_calc.rmsd[:, 2]
k141e_rd = k141e_rmsd_calc.rmsd[:, 2]

# Основной график RMSD
plt.figure(figsize=(12, 8))
for i, (data, color, label) in enumerate(zip(
        [wt_rd, k137e_rd, k141e_rd],
        ['blue', 'red', 'darkgreen'],
        ['WT', 'K137E', 'K141E']
), 1):
    plt.subplot(2, 2, i)
    plt.plot(md_step_ns, data, color=color, linewidth=1)
    smooth = lowess(data, md_step_ns, frac=0.1)
    plt.plot(smooth[:, 0], smooth[:, 1], '--', color=color, linewidth=2)
    plt.title(f"HSPB8-{label}")
    plt.xlabel("Time, ns")
    plt.ylabel("RMSD, Å")

plt.subplot(2, 2, 4)
for data, color, label in zip([wt_rd, k137e_rd, k141e_rd],
                              ['blue', 'red', 'darkgreen'],
                              ['WT', 'K137E', 'K141E']):
    plt.plot(md_step_ns, data, color=color, label=label, linewidth=1)
    smooth = lowess(data, md_step_ns, frac=0.1)
    plt.plot(smooth[:, 0], smooth[:, 1], '--', color=color, linewidth=2)

plt.title("HSPB8")
plt.xlabel("Time, ns")
plt.ylabel("RMSD, Å")
plt.legend()
plt.tight_layout()
plt.show()

# Гистограммы RMSD
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, data, color, label in zip(axes, [wt_rd, k137e_rd, k141e_rd],
                                  ['blue', 'red', 'darkgreen'],
                                  ['WT', 'K137E', 'K141E']):
    ax.hist(data, bins=40, density=True, color=color, alpha=0.6)
    kde = stats.gaussian_kde(data)
    x_range = np.linspace(data.min(), data.max(), 100)
    ax.plot(x_range, kde(x_range), color=color, linewidth=2)
    ax.set_title(f"HSPB8-{label} RMSD Histogram")
    ax.set_xlabel("RMSD")
    ax.set_ylabel("Density")

plt.tight_layout()
plt.show()

# Статистика RMSD
print("WT RMSD Summary:")
print(pd.Series(wt_rd).describe())
print("\nK137E RMSD Summary:")
print(pd.Series(k137e_rd).describe())
print("\nK141E RMSD Summary:")
print(pd.Series(k141e_rd).describe())



# Анализ радиуса инерции
wt_rg = pd.read_csv("wt_rg_pc3.dat", sep="\s+", header=0)
k137e_rg = pd.read_csv("k137e_rg_pc3.dat", sep="\s+", header=0)
k141e_rg = pd.read_csv("k141e_rg_pc3.dat", sep="\s+", header=0)

# Графики радиуса инерции
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(md_step_ns, wt_rg.iloc[:, 1], 'b-', alpha=0.7, label='WT')
plt.plot(md_step_ns, k137e_rg.iloc[:, 1], 'r-', alpha=0.7, label='K137E')
plt.plot(md_step_ns, k141e_rg.iloc[:, 1], 'g-', alpha=0.7, label='K141E')

# Сглаживание
for data, color in zip([wt_rg.iloc[:, 1], k137e_rg.iloc[:, 1], k141e_rg.iloc[:, 1]],
                       ['blue', 'red', 'darkgreen']):
    smooth = lowess(data, md_step_ns, frac=0.1)
    plt.plot(smooth[:, 0], smooth[:, 1], '--', color=color, linewidth=2)

plt.title("Radius of Gyration")
plt.xlabel("Time, ns")
plt.ylabel("Rg, nm")
plt.legend()

plt.subplot(1, 2, 2)
# Прозрачные гистограммы
plt.hist(wt_rg.iloc[:, 1], bins=40, density=True, alpha=0.3, color='blue', label='WT')
plt.hist(k137e_rg.iloc[:, 1], bins=40, density=True, alpha=0.3, color='red', label='K137E')
plt.hist(k141e_rg.iloc[:, 1], bins=40, density=True, alpha=0.3, color='green', label='K141E')

# KDE
for data, color in zip([wt_rg.iloc[:, 1], k137e_rg.iloc[:, 1], k141e_rg.iloc[:, 1]],
                       ['blue', 'red', 'darkgreen']):
    kde = stats.gaussian_kde(data)
    x_range = np.linspace(data.min(), data.max(), 100)
    plt.plot(x_range, kde(x_range), color=color, linewidth=2)

plt.title("Radius of Gyration Distribution")
plt.xlabel("Rg, nm")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.show()

# Анализ SASA
wt_sasa = pd.read_csv("wt_SASA.dat", sep="\s+", header=None)
k137e_sasa = pd.read_csv("k137e_SASA.dat", sep="\s+", header=None)
k141e_sasa = pd.read_csv("k141e_SASA.dat", sep="\s+", header=None)

# Графики SASA
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
frames = np.arange(1, 1001)
plt.plot(frames, wt_sasa.iloc[:, 0] / 10, 'b-', label='WT')
plt.plot(frames, k137e_sasa.iloc[:, 0] / 10, 'r-', label='K137E')
plt.plot(frames, k141e_sasa.iloc[:, 0] / 10, 'g-', label='K141E')
plt.title("SASA Over Time")
plt.xlabel("Frame No.")
plt.ylabel("SASA")
plt.legend()

plt.subplot(1, 3, 2)
# KDE SASA
for data, color, label in zip([wt_sasa.iloc[:, 0] / 10, k137e_sasa.iloc[:, 0] / 10, k141e_sasa.iloc[:, 0] / 10],
                              ['blue', 'red', 'darkgreen'], ['WT', 'K137E', 'K141E']):
    kde = stats.gaussian_kde(data)
    x_range = np.linspace(data.min(), data.max(), 100)
    plt.plot(x_range, kde(x_range), color=color, label=label, linewidth=2)

plt.title("SASA Distribution")
plt.xlabel("SASA")
plt.ylabel("Density")
plt.legend()

plt.subplot(1, 3, 3)
# Гистограммы SASA
plt.hist(wt_sasa.iloc[:, 0] / 10, bins=40, density=True, alpha=0.3, color='blue', label='WT')
plt.hist(k137e_sasa.iloc[:, 0] / 10, bins=40, density=True, alpha=0.3, color='red', label='K137E')
plt.hist(k141e_sasa.iloc[:, 0] / 10, bins=40, density=True, alpha=0.3, color='green', label='K141E')

plt.title("SASA Histograms")
plt.xlabel("SASA")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.show()


# PCA анализ
def perform_pca(universe, ca_selection):
    """Выполняет PCA анализ для траектории"""
    ca_atoms = universe.select_atoms(ca_selection)
    coordinates = []
    for ts in universe.trajectory:
        coordinates.append(ca_atoms.positions.flatten())
    coordinates = np.array(coordinates)

    # Центрирование данных
    coordinates_centered = coordinates - coordinates.mean(axis=0)

    # PCA
    cov_matrix = np.cov(coordinates_centered.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Сортировка по убыванию собственных значений
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Проекция на главные компоненты
    projected = coordinates_centered.dot(eigenvectors)

    return {
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors,
        'projected': projected,
        'mean': coordinates.mean(axis=0)
    }


# Выполнение PCA для каждой системы
wt_pca = perform_pca(wt_universe, "protein and name CA")
k137e_pca = perform_pca(k137e_universe, "protein and name CA")
k141e_pca = perform_pca(k141e_universe, "protein and name CA")

# Графики PCA
plt.figure(figsize=(15, 5))

# PC1 vs PC2
plt.subplot(1, 3, 1)
plt.scatter(wt_pca['projected'][:, 0], wt_pca['projected'][:, 1],
            c=np.arange(len(wt_pca['projected'])), cmap='bwr', alpha=0.6)
plt.colorbar(label='Time')
plt.title("WT PCA")
plt.xlabel("PC1")
plt.ylabel("PC2")

plt.subplot(1, 3, 2)
plt.scatter(k137e_pca['projected'][:, 0], k137e_pca['projected'][:, 1],
            c=np.arange(len(k137e_pca['projected'])), cmap='bwr', alpha=0.6)
plt.colorbar(label='Time')
plt.title("K137E PCA")
plt.xlabel("PC1")

plt.subplot(1, 3, 3)
plt.scatter(k141e_pca['projected'][:, 0], k141e_pca['projected'][:, 1],
            c=np.arange(len(k141e_pca['projected'])), cmap='bwr', alpha=0.6)
plt.colorbar(label='Time')
plt.title("K141E PCA")
plt.xlabel("PC1")
plt.tight_layout()
plt.show()


# Анализ вклада остатков в главные компоненты
def calculate_residue_contribution(pca_result, n_residues):
    """Рассчитывает вклад остатков в главные компоненты"""
    pc1_contribution = pca_result['eigenvectors'][:, 0].reshape(n_residues, 3)
    pc2_contribution = pca_result['eigenvectors'][:, 1].reshape(n_residues, 3)
    pc3_contribution = pca_result['eigenvectors'][:, 2].reshape(n_residues, 3)

    return {
        'pc1': np.linalg.norm(pc1_contribution, axis=1),
        'pc2': np.linalg.norm(pc2_contribution, axis=1),
        'pc3': np.linalg.norm(pc3_contribution, axis=1)
    }


n_residues = len(wt_ca)
wt_contrib = calculate_residue_contribution(wt_pca, n_residues)
k137e_contrib = calculate_residue_contribution(k137e_pca, n_residues)
k141e_contrib = calculate_residue_contribution(k141e_pca, n_residues)

# Графики вклада в главные компоненты
residues = np.arange(1, n_residues + 1)

fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# PC1
axes[0].plot(residues, wt_contrib['pc1'], 'b-', linewidth=2, label='WT')
axes[0].plot(residues, k137e_contrib['pc1'], 'r-', linewidth=2, label='K137E')
axes[0].plot(residues, k141e_contrib['pc1'], 'g-', linewidth=2, label='K141E')
axes[0].set_title("PC1 Contribution")
axes[0].set_ylabel("PC1 Contribution")
axes[0].legend()

# PC2
axes[1].plot(residues, wt_contrib['pc2'], 'b-', linewidth=2, label='WT')
axes[1].plot(residues, k137e_contrib['pc2'], 'r-', linewidth=2, label='K137E')
axes[1].plot(residues, k141e_contrib['pc2'], 'g-', linewidth=2, label='K141E')
axes[1].set_title("PC2 Contribution")
axes[1].set_ylabel("PC2 Contribution")

# PC3
axes[2].plot(residues, wt_contrib['pc3'], 'b-', linewidth=2, label='WT')
axes[2].plot(residues, k137e_contrib['pc3'], 'r-', linewidth=2, label='K137E')
axes[2].plot(residues, k141e_contrib['pc3'], 'g-', linewidth=2, label='K141E')
axes[2].set_title("PC3 Contribution")
axes[2].set_xlabel("Residue Position")
axes[2].set_ylabel("PC3 Contribution")

plt.tight_layout()
plt.show()

print("Анализ завершен!")