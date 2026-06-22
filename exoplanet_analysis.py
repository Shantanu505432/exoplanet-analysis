# Exoplanet Data Analysis
# ---------------------
# I got this dataset from the NASA Exoplanet Archive
# Download link: https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=PSCompPars
# Save it as "exoplanets.csv" in the same folder as this file
#
# What I'm trying to do here:
# - Look at how big exoplanets are compared to Earth/Jupiter
# - See how long their years are (orbital period)
# - Check which ones might be in the habitable zone
# - Basically just explore the data and see what's interesting

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# setting a style so the plots dont look ugly
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")


# ── 1. LOAD THE DATA ───────────

df = pd.read_csv("exoplanets.csv", comment="#")  # NASA files have # comment lines at top

# let's see what columns we have
print("Columns in dataset:")
print(df.columns.tolist())
print(f"\nTotal planets in dataset: {len(df)}")


# ── 2. CLEAN THE DATA ───────────

# honestly a lot of rows have missing values, so I'll just drop rows
# where the important columns are empty - not ideal but works for now

columns_i_need = [
    "pl_name",        # planet name
    "pl_rade",        # planet radius in Earth radii
    "pl_bmasse",      # planet mass in Earth masses (bmasse = best mass estimate)
    "pl_orbper",      # orbital period in days
    "pl_eqt",         # equilibrium temperature in Kelvin
    "st_teff",        # star temperature
    "sy_dist",        # distance from us in parsecs
]

# keep only columns I need and drop rows with missing values
df_clean = df[columns_i_need].dropna()
print(f"\nAfter cleaning (dropped rows with missing values): {len(df_clean)} planets")


# ── 3. BASIC STATS ────────────────────────────

print("\n── Basic Statistics ──")
print(f"Smallest planet:  {df_clean['pl_rade'].min():.2f} Earth radii")
print(f"Largest planet:   {df_clean['pl_rade'].max():.2f} Earth radii")
print(f"Average radius:   {df_clean['pl_rade'].mean():.2f} Earth radii")
print(f"Shortest year:    {df_clean['pl_orbper'].min():.2f} days")
print(f"Longest year:     {df_clean['pl_orbper'].max():.2f} days")
print(f"Farthest planet:  {df_clean['sy_dist'].max():.0f} parsecs away")


#  4. HABITABLE ZONE ───

# The habitable zone is roughly where liquid water can exist
# I looked this up - equilibrium temperature between 200K and 320K is a rough estimate
# Earth is about 255K equilibrium temperature

df_clean["habitable"] = (df_clean["pl_eqt"] >= 200) & (df_clean["pl_eqt"] <= 320)
habitable_count = df_clean["habitable"].sum()
print(f"\nPlanets in rough habitable zone: {habitable_count}")


#  5. PLOTS ───────────────────────

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("NASA Exoplanet Archive — Exploratory Analysis", fontsize=16, fontweight="bold")


# Plot 1 — Planet radius distribution
# I expected most planets to be Jupiter sized but turns out there are lots of small ones
ax1 = axes[0, 0]
ax1.hist(df_clean["pl_rade"], bins=60, color="steelblue", edgecolor="white", alpha=0.8)
ax1.axvline(1,    color="green",  linestyle="--", linewidth=1.5, label="Earth (1 R⊕)")
ax1.axvline(11.2, color="orange", linestyle="--", linewidth=1.5, label="Jupiter (11.2 R⊕)")
ax1.set_xlabel("Planet Radius (Earth Radii)")
ax1.set_ylabel("Number of Planets")
ax1.set_title("Distribution of Planet Sizes")
ax1.legend()
ax1.set_xlim(0, 30)  # cutting off outliers for readability


# Plot 2 — Orbital period distribution (log scale because range is huge)
# some planets orbit in hours, others take thousands of days - log scale helps
ax2 = axes[0, 1]
ax2.hist(np.log10(df_clean["pl_orbper"]), bins=50, color="coral", edgecolor="white", alpha=0.8)
ax2.axvline(np.log10(365), color="green", linestyle="--", linewidth=1.5, label="Earth (365 days)")
ax2.set_xlabel("Orbital Period (log₁₀ days)")
ax2.set_ylabel("Number of Planets")
ax2.set_title("Distribution of Orbital Periods (log scale)")
ax2.legend()

# adding actual day labels on x axis so it's readable
tick_vals = [0, 1, 2, 3, 4]
ax2.set_xticks(tick_vals)
ax2.set_xticklabels([f"{10**v:.0f}d" for v in tick_vals])


# Plot 3 — Mass vs Radius (should follow a rough power law)
# this is basically the most classic exoplanet plot
ax3 = axes[1, 0]

# separating habitable and non-habitable for color coding
non_hab = df_clean[~df_clean["habitable"]]
hab     = df_clean[df_clean["habitable"]]

ax3.scatter(non_hab["pl_bmasse"], non_hab["pl_rade"],
            alpha=0.3, s=8, color="steelblue", label="Not habitable zone")
ax3.scatter(hab["pl_bmasse"], hab["pl_rade"],
            alpha=0.8, s=25, color="red", label="Habitable zone", zorder=5)

# marking Earth and Jupiter for reference
ax3.scatter([1],    [1],    color="green",  s=100, marker="*", zorder=6, label="Earth")
ax3.scatter([317.8],[11.2], color="orange", s=100, marker="*", zorder=6, label="Jupiter")

ax3.set_xscale("log")
ax3.set_yscale("log")
ax3.set_xlabel("Planet Mass (Earth Masses)")
ax3.set_ylabel("Planet Radius (Earth Radii)")
ax3.set_title("Mass vs Radius (red = habitable zone)")
ax3.legend(fontsize=8)


# Plot 4 — Equilibrium temperature distribution
# curious to see how many are Earth-like in temperature
ax4 = axes[1, 1]
ax4.hist(df_clean["pl_eqt"], bins=60, color="mediumpurple", edgecolor="white", alpha=0.8)
ax4.axvline(255, color="green",  linestyle="--", linewidth=1.5, label="Earth (~255K)")
ax4.axvspan(200, 320, alpha=0.15, color="green", label="Rough habitable range")
ax4.set_xlabel("Equilibrium Temperature (K)")
ax4.set_ylabel("Number of Planets")
ax4.set_title("Planet Temperature Distribution")
ax4.legend()
ax4.set_xlim(0, 3000)


plt.tight_layout()
plt.savefig("exoplanet_analysis.png", dpi=150, bbox_inches="tight")
print("\nPlot saved as exoplanet_analysis.png")
plt.show()


# 6. CLOSEST HABITABLE ZONE PLANETS 

print("\n── Top 10 Closest Planets in Habitable Zone ──")
hab_sorted = df_clean[df_clean["habitable"]].sort_values("sy_dist")
print(hab_sorted[["pl_name", "pl_rade", "pl_bmasse", "pl_eqt", "sy_dist"]].head(10).to_string(index=False))


# ── 7. INTERESTING FINDINGS 

print("\n── A few things I noticed ──")
print(f"Most planets are larger than Earth — median radius is {df_clean['pl_rade'].median():.1f} Earth radii")
print(f"Most planets have very short years — median period is {df_clean['pl_orbper'].median():.1f} days")
print(f"Only {habitable_count} out of {len(df_clean)} planets ({100*habitable_count/len(df_clean):.1f}%) are in the rough habitable zone")
print("(This is partly a selection bias — it's easier to detect planets close to their stars)")
