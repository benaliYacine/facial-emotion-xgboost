import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the facial features data
df = pd.read_csv("facial_features22.csv")

# Remove the last two columns (emotion and image_path) as they're not features for analysis
X_with_names = df.drop(["emotion", "image_path"], axis=1)

# We'll use the original features for our PCA (no normalized features in the data)
feature_cols = X_with_names.columns.tolist()

# Get the data matrix
X = X_with_names.values

# Define individuals (faces) and characteristics (features)
individus = df.index.tolist()  # Using row indices as individual identifiers
caracteristiques = feature_cols

# Create DataFrame for better visualization
df_data = pd.DataFrame(X, index=individus, columns=caracteristiques)
print("Matrice de données :")
print(df_data.head())

# Calculate mean individual (individu type)
individu_type = np.mean(X, axis=0)
print("\nIndividu type :")
print(pd.Series(individu_type, index=caracteristiques))

# Center the data
X_centered = X - individu_type
print("\nMatrice centrée :")
print(pd.DataFrame(X_centered, index=individus, columns=caracteristiques).head())

# Calculate standard deviation
s = np.std(X, axis=0, ddof=0)
print("\nÉcart type :")
print(pd.Series(s, index=caracteristiques))

# Check for zeros in standard deviation and handle them
print("\nChecking for zeros in standard deviation:")
zero_std_indices = np.where(s == 0)[0]
if len(zero_std_indices) > 0:
    print(
        f"Warning: Zero standard deviation for features: {[caracteristiques[i] for i in zero_std_indices]}"
    )
    # Replace zeros with a small value to avoid division by zero
    s[zero_std_indices] = 1e-10

# Calculate centered and reduced matrix (Z)
Z = X_centered / s
print("\nMatrice centrée réduite :")
print(pd.DataFrame(Z, index=individus, columns=caracteristiques).head())

# Calculate correlation matrix using the np.corrcoef function (more numerically stable)
R = np.corrcoef(Z.T)
R_df = pd.DataFrame(R, columns=caracteristiques, index=caracteristiques)
print("\nMatrice de corrélation :")
print(R_df)

# Save the correlation matrix to a CSV file
R_df.to_csv("correlation_matrix.csv")
print("Correlation matrix saved to 'correlation_matrix.csv'")

# Check for NaN or inf values in R
if np.isnan(R).any() or np.isinf(R).any():
    print(
        "Warning: NaN or inf values found in correlation matrix. Replacing with zeros."
    )
    R = np.nan_to_num(R)

# Calculate eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(R)
# Convert to real if complex (they should be real theoretically, but sometimes have tiny imaginary parts)
eigenvalues = np.real(eigenvalues)
eigenvectors = np.real(eigenvectors)

# Sort eigenvalues and eigenvectors in descending order
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print("\nValeurs propres triées :")
print(eigenvalues)

# Calculate inertia
inertia = eigenvalues / np.sum(eigenvalues)
cumulative_inertia = np.cumsum(inertia)

# Calculate how many components are needed to reach 80% of inertia
components_80_percent = np.where(cumulative_inertia >= 0.8)[0][0] + 1
print(
    f"\nNombre de composantes nécessaires pour atteindre 80% d'inertie: {components_80_percent}"
)

# Create results table
results_table = pd.DataFrame(
    {
        "Valeurs propres": np.round(eigenvalues, 4),
        "Inertie (%)": np.round(inertia * 100, 4),
        "Inertie cumulée (%)": np.round(cumulative_inertia * 100, 4),
    }
)
print("-------------------------------------------------------------------------")
print("\nTableau des valeurs propres et inerties :")
print(results_table)

# Plot scree plot with 80% threshold line
plt.figure(figsize=(10, 6))
plt.plot(
    range(1, len(eigenvalues) + 1),
    eigenvalues,
    marker="o",
    linestyle="-",
    label="Valeurs propres",
)
plt.axhline(y=1, color="r", linestyle="--", label="Valeur propre = 1")
plt.axhline(y=0, color="k", linestyle="-", alpha=0.3)
plt.plot(
    range(1, len(eigenvalues) + 1),
    cumulative_inertia * max(eigenvalues),
    marker="s",
    linestyle=":",
    color="green",
    label="Inertie cumulée (%)",
)
plt.axhline(
    y=0.8 * max(eigenvalues),
    color="green",
    linestyle="--",
    label="Seuil 80% d'inertie",
)
plt.xlabel("Numéro de la composante principale")
plt.ylabel("Valeurs propres")
plt.title("Scree Plot - Valeurs propres et inertie cumulée")
plt.grid(True)
plt.xticks(range(1, len(eigenvalues) + 1))
plt.legend()
plt.show()

# Decide how many components to keep based on both criteria
n_components_kaiser = sum(eigenvalues > 1)
print(
    f"\nNombre de composantes à retenir (critère de Kaiser (eigenvalues >1)): {n_components_kaiser}"
)
print(
    f"Nombre de composantes à retenir (critère d'inertie 80%): {components_80_percent}"
)

# Update the number of components to keep (using the 80% inertia criterion)
n_components = components_80_percent

# Calculate projections for 3D visualization if we have at least 3 components
if n_components >= 3:
    # Calculate projections on first three axes
    eigenvectors_1_2_3 = eigenvectors[:, :3]
    projections_1_2_3 = np.dot(Z, eigenvectors_1_2_3)

    # Create DataFrame for 3D projections
    df_projections_3d = pd.DataFrame(
        projections_1_2_3, columns=["Axe 1", "Axe 2", "Axe 3"], index=individus
    )
    print("\nProjections des individus sur les axes 1, 2 et 3 :")
    print(df_projections_3d.head())

    # Calculate variable coordinates for 3D correlation sphere
    variable_coordinates_3d = eigenvectors[:, :3] * np.sqrt(eigenvalues[:3])
    df_variable_coords_3d = pd.DataFrame(
        variable_coordinates_3d,
        columns=["Axe 1", "Axe 2", "Axe 3"],
        index=caracteristiques,
    )
    print("\nCoordonnées des variables dans la sphère de corrélation (3D) :")
    print(df_variable_coords_3d)

    # Plot 3D sphere of correlation
    from mpl_toolkits.mplot3d import Axes3D

    # Create 3D sphere wireframe
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection="3d")

    # Create wireframe sphere
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    # Plot the sphere as wireframe
    ax.plot_wireframe(x, y, z, color="gray", alpha=0.2)

    # Plot the vectors
    for i, var in enumerate(caracteristiques):
        ax.quiver(
            0,
            0,
            0,
            variable_coordinates_3d[i, 0],
            variable_coordinates_3d[i, 1],
            variable_coordinates_3d[i, 2],
            color="red",
            arrow_length_ratio=0.1,
        )
        # Shorten variable names for clarity
        var_label = var.replace("_norm", "")
        ax.text(
            variable_coordinates_3d[i, 0] * 1.1,
            variable_coordinates_3d[i, 1] * 1.1,
            variable_coordinates_3d[i, 2] * 1.1,
            var_label,
            color="black",
            fontsize=8,
        )

    # Set labels and title
    ax.set_xlabel(f"Axe 1 ({inertia[0]*100:.2f}%)")
    ax.set_ylabel(f"Axe 2 ({inertia[1]*100:.2f}%)")
    ax.set_zlabel(f"Axe 3 ({inertia[2]*100:.2f}%)")
    ax.set_title("Sphère de corrélation 3D des caractéristiques faciales")

    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])

    # Set limits to make the sphere look proportional
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    # Add gridlines and axes
    ax.grid(True, linestyle="--", alpha=0.3)

    # Draw axis lines through origin
    ax.plot([-1, 1], [0, 0], [0, 0], "k-", alpha=0.2)  # x-axis
    ax.plot([0, 0], [-1, 1], [0, 0], "k-", alpha=0.2)  # y-axis
    ax.plot([0, 0], [0, 0], [-1, 1], "k-", alpha=0.2)  # z-axis

    plt.tight_layout()
    plt.show()

    # Plot 3D scatter of individuals
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection="3d")

    # Plot the individuals
    ax.scatter(
        projections_1_2_3[:, 0],
        projections_1_2_3[:, 1],
        projections_1_2_3[:, 2],
        color="blue",
        alpha=0.7,
    )

    # Label a few points to avoid overcrowding
    for i in range(min(10, len(individus))):
        ax.text(
            projections_1_2_3[i, 0],
            projections_1_2_3[i, 1],
            projections_1_2_3[i, 2],
            individus[i],
            fontsize=9,
        )

    # Set labels and title
    ax.set_xlabel(f"Axe 1 ({inertia[0]*100:.2f}%)")
    ax.set_ylabel(f"Axe 2 ({inertia[1]*100:.2f}%)")
    ax.set_zlabel(f"Axe 3 ({inertia[2]*100:.2f}%)")
    ax.set_title("Projection 3D des visages dans l'espace factoriel")

    # Add gridlines
    ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Create pair plots to visualize combinations of the first three components
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Axes 1-2
    axs[0].scatter(
        projections_1_2_3[:, 0], projections_1_2_3[:, 1], color="blue", alpha=0.7
    )
    axs[0].set_xlabel(f"Axe 1 ({inertia[0]*100:.2f}%)")
    axs[0].set_ylabel(f"Axe 2 ({inertia[1]*100:.2f}%)")
    axs[0].set_title("Projection sur les axes 1 et 2")
    axs[0].grid(True, linestyle="--", alpha=0.7)
    axs[0].axhline(y=0, color="k", linestyle="-", linewidth=0.5)
    axs[0].axvline(x=0, color="k", linestyle="-", linewidth=0.5)

    # Axes 1-3
    axs[1].scatter(
        projections_1_2_3[:, 0], projections_1_2_3[:, 2], color="green", alpha=0.7
    )
    axs[1].set_xlabel(f"Axe 1 ({inertia[0]*100:.2f}%)")
    axs[1].set_ylabel(f"Axe 3 ({inertia[2]*100:.2f}%)")
    axs[1].set_title("Projection sur les axes 1 et 3")
    axs[1].grid(True, linestyle="--", alpha=0.7)
    axs[1].axhline(y=0, color="k", linestyle="-", linewidth=0.5)
    axs[1].axvline(x=0, color="k", linestyle="-", linewidth=0.5)

    # Axes 2-3
    axs[2].scatter(
        projections_1_2_3[:, 1], projections_1_2_3[:, 2], color="purple", alpha=0.7
    )
    axs[2].set_xlabel(f"Axe 2 ({inertia[1]*100:.2f}%)")
    axs[2].set_ylabel(f"Axe 3 ({inertia[2]*100:.2f}%)")
    axs[2].set_title("Projection sur les axes 2 et 3")
    axs[2].grid(True, linestyle="--", alpha=0.7)
    axs[2].axhline(y=0, color="k", linestyle="-", linewidth=0.5)
    axs[2].axvline(x=0, color="k", linestyle="-", linewidth=0.5)

    plt.tight_layout()
    plt.show()

    # Calculate pairwise correlation circles
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Add correlation circle for axes 1-2
    circle1 = plt.Circle((0, 0), 1, fill=False, color="gray", linestyle="--")
    axs[0].add_artist(circle1)
    for i, var in enumerate(caracteristiques):
        axs[0].arrow(
            0,
            0,
            variable_coordinates_3d[i, 0],
            variable_coordinates_3d[i, 1],
            head_width=0.05,
            head_length=0.1,
            fc="red",
            ec="red",
        )
        var_label = var.replace("_norm", "")
        axs[0].text(
            variable_coordinates_3d[i, 0] * 1.1,
            variable_coordinates_3d[i, 1] * 1.1,
            var_label,
            color="black",
            ha="center",
            va="center",
            fontsize=7,
        )
    axs[0].set_xlim(-1, 1)
    axs[0].set_ylim(-1, 1)
    axs[0].grid(True, linestyle="--", alpha=0.7)
    axs[0].set_title("Cercle de corrélation - Axes 1 et 2")
    axs[0].set_xlabel(f"Axe 1 ({inertia[0]*100:.2f}%)")
    axs[0].set_ylabel(f"Axe 2 ({inertia[1]*100:.2f}%)")

    # Add correlation circle for axes 1-3
    circle2 = plt.Circle((0, 0), 1, fill=False, color="gray", linestyle="--")
    axs[1].add_artist(circle2)
    for i, var in enumerate(caracteristiques):
        axs[1].arrow(
            0,
            0,
            variable_coordinates_3d[i, 0],
            variable_coordinates_3d[i, 2],
            head_width=0.05,
            head_length=0.1,
            fc="green",
            ec="green",
        )
        var_label = var.replace("_norm", "")
        axs[1].text(
            variable_coordinates_3d[i, 0] * 1.1,
            variable_coordinates_3d[i, 2] * 1.1,
            var_label,
            color="black",
            ha="center",
            va="center",
            fontsize=7,
        )
    axs[1].set_xlim(-1, 1)
    axs[1].set_ylim(-1, 1)
    axs[1].grid(True, linestyle="--", alpha=0.7)
    axs[1].set_title("Cercle de corrélation - Axes 1 et 3")
    axs[1].set_xlabel(f"Axe 1 ({inertia[0]*100:.2f}%)")
    axs[1].set_ylabel(f"Axe 3 ({inertia[2]*100:.2f}%)")

    # Add correlation circle for axes 2-3
    circle3 = plt.Circle((0, 0), 1, fill=False, color="gray", linestyle="--")
    axs[2].add_artist(circle3)
    for i, var in enumerate(caracteristiques):
        axs[2].arrow(
            0,
            0,
            variable_coordinates_3d[i, 1],
            variable_coordinates_3d[i, 2],
            head_width=0.05,
            head_length=0.1,
            fc="purple",
            ec="purple",
        )
        var_label = var.replace("_norm", "")
        axs[2].text(
            variable_coordinates_3d[i, 1] * 1.1,
            variable_coordinates_3d[i, 2] * 1.1,
            var_label,
            color="black",
            ha="center",
            va="center",
            fontsize=7,
        )
    axs[2].set_xlim(-1, 1)
    axs[2].set_ylim(-1, 1)
    axs[2].grid(True, linestyle="--", alpha=0.7)
    axs[2].set_title("Cercle de corrélation - Axes 2 et 3")
    axs[2].set_xlabel(f"Axe 2 ({inertia[1]*100:.2f}%)")
    axs[2].set_ylabel(f"Axe 3 ({inertia[2]*100:.2f}%)")

    plt.tight_layout()
    plt.show()

    # Update interpretations to include third axis
    print("\n3. Interprétation du troisième axe principal:")
    axis3_correlations = abs(variable_coordinates_3d[:, 2])
    axis3_vars = [
        caracteristiques[i].replace("_norm", "")
        for i in np.argsort(axis3_correlations)[::-1]
    ]
    print(
        f"Axe 3 ({inertia[2]*100:.2f}%) - Principalement caractérisé par : {', '.join(axis3_vars[:3])}"
    )

    # Calculate contributions for 3 axes
    contributions_ind_3d = (
        (projections_1_2_3**2) / (len(individus) * eigenvalues[:3]) * 100
    )
    df_contributions_ind_3d = pd.DataFrame(
        contributions_ind_3d, columns=["Axe 1", "Axe 2", "Axe 3"], index=individus
    )
    print("\nContributions des individus sur les 3 premiers axes (%) :")
    print(df_contributions_ind_3d.head().round(2))

    # Calculate variable contributions for 3 axes
    contributions_var_3d = variable_coordinates_3d**2 / eigenvalues[:3]
    df_contributions_var_3d = pd.DataFrame(
        contributions_var_3d,
        columns=["Axe 1", "Axe 2", "Axe 3"],
        index=caracteristiques,
    )
    print("\nContributions des variables sur les 3 premiers axes (%) :")
    print(df_contributions_var_3d.round(2))

# Calculate projections on first two axes (this is kept for backward compatibility)
eigenvectors_1_2 = eigenvectors[:, :2]
projections_1_2 = np.dot(Z, eigenvectors_1_2)

# Create DataFrame for projections
df_projections = pd.DataFrame(
    projections_1_2, columns=["Axe 1", "Axe 2"], index=individus
)
print("\nProjections des individus sur les axes 1 et 2 :")
print(df_projections.head())

# Plot individuals in factorial plane
plt.figure(figsize=(12, 10))
plt.scatter(projections_1_2[:, 0], projections_1_2[:, 1], color="blue", alpha=0.7)
# Only label a few points to avoid overcrowding
for i in range(min(10, len(individus))):
    plt.text(projections_1_2[i, 0], projections_1_2[i, 1], individus[i], fontsize=9)
plt.axhline(y=0, color="k", linestyle="-", linewidth=0.5)
plt.axvline(x=0, color="k", linestyle="-", linewidth=0.5)
plt.grid(True, linestyle="--", alpha=0.7)
plt.xlabel(f"Axe 1 ({inertia[0]*100:.2f}%)")
plt.ylabel(f"Axe 2 ({inertia[1]*100:.2f}%)")
plt.title("Projection des visages dans le plan factoriel")
plt.show()

# Calculate variable coordinates for correlation circle
variable_coordinates = eigenvectors[:, :2] * np.sqrt(eigenvalues[:2])
df_variable_coords = pd.DataFrame(
    variable_coordinates, columns=["Axe 1", "Axe 2"], index=caracteristiques
)
print("\nCoordonnées des variables dans le cercle de corrélation :")
print(df_variable_coords)

# Plot correlation circle
plt.figure(figsize=(12, 10))
circle = plt.Circle((0, 0), 1, fill=False, color="gray", linestyle="--")
plt.gca().add_artist(circle)
plt.grid(True, linestyle="--", alpha=0.7)
for i, var in enumerate(caracteristiques):
    plt.arrow(
        0,
        0,
        variable_coordinates[i, 0],
        variable_coordinates[i, 1],
        head_width=0.05,
        head_length=0.1,
        fc="red",
        ec="red",
    )
    # Shorten variable names for clarity
    var_label = var.replace("_norm", "")
    plt.text(
        variable_coordinates[i, 0] * 1.1,
        variable_coordinates[i, 1] * 1.1,
        var_label,
        color="black",
        ha="center",
        va="center",
        fontsize=8,
    )
plt.axhline(y=0, color="k", linestyle="-", linewidth=0.5)
plt.axvline(x=0, color="k", linestyle="-", linewidth=0.5)
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.xlabel(f"Axe 1 ({inertia[0]*100:.2f}%)")
plt.ylabel(f"Axe 2 ({inertia[1]*100:.2f}%)")
plt.title("Cercle de corrélation des caractéristiques faciales")
plt.show()

# Calculate contributions of individuals
contributions_ind = (projections_1_2**2) / (len(individus) * eigenvalues[:2]) * 100
df_contributions_ind = pd.DataFrame(
    contributions_ind, columns=["Axe 1", "Axe 2"], index=individus
)
print("\nContributions des individus (%) :")
print(df_contributions_ind.head().round(2))
print("\nMoyenne des contributions des individus :")
print(np.mean(df_contributions_ind, axis=0).round(2))

# Calculate quality of representation (cos²)
cos2 = (projections_1_2**2) / np.sum(Z**2, axis=1, keepdims=True)
df_cos2 = pd.DataFrame(cos2, columns=["Axe 1", "Axe 2"], index=individus)
print("\nQualité de représentation (cos²) :")
print(df_cos2.head().round(4))

# Calculate variable contributions
contributions_var = variable_coordinates**2 / eigenvalues[:2]
df_contributions_var = pd.DataFrame(
    contributions_var, columns=["Axe 1", "Axe 2"], index=caracteristiques
)
print("\nContributions des variables (%) :")
print(df_contributions_var.round(2))
print("\nMoyenne des contributions des variables :")
print(np.mean(df_contributions_var, axis=0).round(2))

# Interpretations
print("\n" + "=" * 80)
print("INTERPRÉTATIONS DE L'ACP")
print("=" * 80)

print("\n1. Analyse de l'inertie :")
print(f"- Le premier axe explique {inertia[0]*100:.2f}% de l'inertie totale")
print(f"- Le deuxième axe explique {inertia[1]*100:.2f}% de l'inertie totale")
print(
    f"- Les deux premiers axes cumulent {(inertia[0] + inertia[1])*100:.2f}% de l'information totale"
)

print("\n2. Analyse du cercle des corrélations :")
print("- Les variables proches du cercle sont bien représentées")
print("- Les variables proches les unes des autres sont positivement corrélées")
print("- Les variables opposées sont négativement corrélées")
print("- Les variables orthogonales sont indépendantes")

print("\n3. Interprétation des axes principaux :")
# Sort variables by their absolute correlation with Axis 1
axis1_correlations = abs(variable_coordinates[:, 0])
axis1_vars = [
    caracteristiques[i].replace("_norm", "")
    for i in np.argsort(axis1_correlations)[::-1]
]
print(
    f"Axe 1 ({inertia[0]*100:.2f}%) - Principalement caractérisé par : {', '.join(axis1_vars[:3])}"
)

# Sort variables by their absolute correlation with Axis 2
axis2_correlations = abs(variable_coordinates[:, 1])
axis2_vars = [
    caracteristiques[i].replace("_norm", "")
    for i in np.argsort(axis2_correlations)[::-1]
]
print(
    f"Axe 2 ({inertia[1]*100:.2f}%) - Principalement caractérisé par : {', '.join(axis2_vars[:3])}"
)

print("\n4. Analyse des individus :")
print("Individus bien représentés (cos² élevé) :")
well_represented = df_cos2[df_cos2.sum(axis=1) > 0.5].index.tolist()[
    :5
]  # Limiting to first 5 for readability
if well_represented:
    # Convert any non-string elements in the list to strings before joining
    well_represented_str = [str(item) for item in well_represented]
    print(f"- {', '.join(well_represented_str)}")
else:
    print("- Aucun individu n'a un cos² cumulé > 0.5")

print("\nIndividus avec fortes contributions :")
threshold = 100 / len(individus)  # Seuil théorique pour une contribution significative
high_contrib_axis1 = df_contributions_ind[
    df_contributions_ind["Axe 1"] > threshold
].index.tolist()[:5]
high_contrib_axis1_str = [str(item) for item in high_contrib_axis1]
high_contrib_axis2 = df_contributions_ind[
    df_contributions_ind["Axe 2"] > threshold
].index.tolist()[:5]
high_contrib_axis2_str = [str(item) for item in high_contrib_axis2]
print(f"- Sur l'axe 1 : {', '.join(high_contrib_axis1_str)}")
print(f"- Sur l'axe 2 : {', '.join(high_contrib_axis2_str)}")

print("\n5. Relations entre variables :")
# Identifier les corrélations fortes (positives et négatives) à partir de la matrice de corrélation
strong_correlations = []
for i in range(len(caracteristiques)):
    for j in range(i + 1, len(caracteristiques)):
        if abs(R[i, j]) > 0.7:  # Seuil arbitraire pour les corrélations fortes
            strong_correlations.append(
                f"{caracteristiques[i].replace('_norm', '')} et {caracteristiques[j].replace('_norm', '')} (r={R[i,j]:.2f})"
            )

if strong_correlations:
    print("Corrélations fortes détectées entre :")
    for corr in strong_correlations[:5]:  # Limiting to first 5 for readability
        print(f"- {corr}")
    if len(strong_correlations) > 5:
        print(f"- Et {len(strong_correlations) - 5} autres corrélations fortes...")
else:
    print("Pas de corrélations particulièrement fortes entre les variables")

print("\n6. Conclusion générale :")
print(
    f"L'analyse en composantes principales permet d'expliquer {(inertia[0] + inertia[1])*100:.2f}% "
)
print(
    "de la variance totale sur les deux premiers axes, ce qui représente une synthèse des caractéristiques faciales."
)

# New analysis section
print("\n" + "=" * 80)
print("ANALYSE DÉTAILLÉE DES AXES")
print("=" * 80)

# Analysis for Axis 1
print("\nAXE 1:")
print("-" * 40)

# Calculate averages for Axis 1
moy_contrib_ind_axe1 = df_contributions_ind["Axe 1"].mean()
moy_contrib_var_axe1 = df_contributions_var["Axe 1"].mean()

print(f"\nMoyenne des contributions des individus (Axe 1): {moy_contrib_ind_axe1:.2f}%")
print(f"Moyenne des contributions des variables (Axe 1): {moy_contrib_var_axe1:.2f}%")

# Create tables for Axis 1
ind_contrib_sup_axe1 = df_contributions_ind[
    df_contributions_ind["Axe 1"] > moy_contrib_ind_axe1
].head(5)
var_contrib_sup_axe1 = df_contributions_var[
    df_contributions_var["Axe 1"] > moy_contrib_var_axe1
]

print("\nIndividus avec contribution > moyenne sur Axe 1 (top 5):")
contrib_sign_ind_axe1 = pd.DataFrame(
    {
        "Contribution (%)": ind_contrib_sup_axe1["Axe 1"].round(2),
        "Coordonnée": df_projections.loc[ind_contrib_sup_axe1.index, "Axe 1"].round(3),
        "Position": [
            "Positive" if x > 0 else "Negative"
            for x in df_projections.loc[ind_contrib_sup_axe1.index, "Axe 1"]
        ],
    }
)
print(contrib_sign_ind_axe1)

print("\nVariables avec contribution > moyenne sur Axe 1:")
contrib_sign_var_axe1 = pd.DataFrame(
    {
        "Contribution (%)": var_contrib_sup_axe1["Axe 1"].round(2),
        "Corrélation": df_variable_coords.loc[
            var_contrib_sup_axe1.index, "Axe 1"
        ].round(3),
        "Position": [
            "Positive" if x > 0 else "Negative"
            for x in df_variable_coords.loc[var_contrib_sup_axe1.index, "Axe 1"]
        ],
    }
)
print(contrib_sign_var_axe1)

# Analysis for Axis 2
print("\nAXE 2:")
print("-" * 40)

# Calculate averages for Axis 2
moy_contrib_ind_axe2 = df_contributions_ind["Axe 2"].mean()
moy_contrib_var_axe2 = df_contributions_var["Axe 2"].mean()

print(f"\nMoyenne des contributions des individus (Axe 2): {moy_contrib_ind_axe2:.2f}%")
print(f"Moyenne des contributions des variables (Axe 2): {moy_contrib_var_axe2:.2f}%")

# Create tables for Axis 2
ind_contrib_sup_axe2 = df_contributions_ind[
    df_contributions_ind["Axe 2"] > moy_contrib_ind_axe2
].head(5)
var_contrib_sup_axe2 = df_contributions_var[
    df_contributions_var["Axe 2"] > moy_contrib_var_axe2
]

print("\nIndividus avec contribution > moyenne sur Axe 2 (top 5):")
contrib_sign_ind_axe2 = pd.DataFrame(
    {
        "Contribution (%)": ind_contrib_sup_axe2["Axe 2"].round(2),
        "Coordonnée": df_projections.loc[ind_contrib_sup_axe2.index, "Axe 2"].round(3),
        "Position": [
            "Positive" if x > 0 else "Negative"
            for x in df_projections.loc[ind_contrib_sup_axe2.index, "Axe 2"]
        ],
    }
)
print(contrib_sign_ind_axe2)

print("\nVariables avec contribution > moyenne sur Axe 2:")
contrib_sign_var_axe2 = pd.DataFrame(
    {
        "Contribution (%)": var_contrib_sup_axe2["Axe 2"].round(2),
        "Corrélation": df_variable_coords.loc[
            var_contrib_sup_axe2.index, "Axe 2"
        ].round(3),
        "Position": [
            "Positive" if x > 0 else "Negative"
            for x in df_variable_coords.loc[var_contrib_sup_axe2.index, "Axe 2"]
        ],
    }
)
print(contrib_sign_var_axe2)

# Add this new section after calculating components_80_percent
print("\n" + "=" * 80)
print(
    "FEATURES ORDERED BY CONTRIBUTION TO THE FIRST", components_80_percent, "COMPONENTS"
)
print("=" * 80)

# Calculate the contribution of each feature to the first n components
feature_contributions = np.zeros(len(caracteristiques))
for i in range(components_80_percent):
    # For each component, calculate each feature's contribution weighted by eigenvalue
    component_contrib = (eigenvectors[:, i] ** 2) * eigenvalues[i]
    feature_contributions += component_contrib

# Create a DataFrame with the contributions
feature_importance = pd.DataFrame(
    {"Feature": caracteristiques, "Contribution": feature_contributions}
)

# Sort by contribution in descending order
feature_importance = feature_importance.sort_values("Contribution", ascending=False)

# Display the features by importance
print(
    "\nFeatures ordered by contribution to the first",
    components_80_percent,
    "components:",
)
print(feature_importance)

# Plot the feature contributions
plt.figure(figsize=(12, 8))
plt.barh(feature_importance["Feature"], feature_importance["Contribution"])
plt.xlabel("Contribution")
plt.ylabel("Features")
plt.title(
    f"Feature Contributions to the First {components_80_percent} Principal Components"
)
plt.tight_layout()
plt.show()
