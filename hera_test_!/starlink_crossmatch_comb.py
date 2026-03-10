import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyuvdata import UVData
from skyfield.api import load, wgs84
from sklearn.cluster import DBSCAN

# -----------------------------
# 1️⃣ TLE 로드
# -----------------------------
satellites = load.tle_file("starlink.tle")
print("Total satellites:", len(satellites))

# 속도 위해 subset 사용 (필요하면 조절)
satellites = satellites[:1500]

# -----------------------------
# 2️⃣ HERA observation time
# -----------------------------
uv = UVData()
uv.read_uvh5("C:\\starstar\\hera_uvh5\\zen.LST.baseline.0_5.sum.uvh5", read_data=False)

jd_times = uv.time_array

# 시간 다운샘플링 (매우 중요)
jd_times = jd_times[::50]

ts = load.timescale()
t_obs = ts.tt_jd(jd_times)

print("Time samples used:", len(t_obs))

# -----------------------------
# 3️⃣ HERA 위치
# -----------------------------
hera = wgs84.latlon(-30.7215, 21.4283)

# -----------------------------
# 4️⃣ satellite elevation 계산
# -----------------------------
sat_elevation_time = []

for ti, t in enumerate(t_obs):

    if ti % 10 == 0:
        print("processing time index:", ti)

    visible_count = 0
    max_elev = -90

    for sat in satellites:

        difference = sat - hera
        topocentric = difference.at(t)

        alt, az, dist = topocentric.altaz()
        elev = alt.degrees

        if elev > 0:
            visible_count += 1

        if elev > max_elev:
            max_elev = elev

    sat_elevation_time.append({
        "time_index": ti,
        "visible_satellites": visible_count,
        "max_elevation": max_elev
    })

sat_df = pd.DataFrame(sat_elevation_time)

print("Satellite stats:")
print(sat_df.describe())

# -----------------------------
# 5️⃣ comb detection load
# -----------------------------
comb_df = pd.read_json("comb_results.json")

# comb 데이터도 동일 길이 맞추기
comb_df = comb_df.iloc[:len(sat_df)]

comb_df["time_index"] = np.arange(len(comb_df))

merged = comb_df.merge(sat_df, on="time_index", how="left")

# -----------------------------
# 6️⃣ Plot: spacing vs elevation
# -----------------------------
plt.figure(figsize=(8,4))

plt.scatter(
    merged["max_elevation"],
    merged["delta_f_hat_mhz"],
    alpha=0.7
)

plt.xlabel("Max Starlink Elevation [deg]")
plt.ylabel("Comb Spacing [MHz]")
plt.title("Comb Spacing vs Satellite Elevation")

plt.tight_layout()
plt.savefig("comb_spacing_vs_sat_elev.png")
plt.show()

# -----------------------------
# 7️⃣ linewidth vs elevation
# -----------------------------
plt.figure(figsize=(8,4))

plt.scatter(
    merged["max_elevation"],
    merged["linewidth_mhz"],
    alpha=0.7
)

plt.xlabel("Max Starlink Elevation [deg]")
plt.ylabel("Comb Linewidth [MHz]")
plt.title("Comb Linewidth vs Satellite Elevation")

plt.tight_layout()
plt.savefig("comb_linewidth_vs_sat_elev.png")
plt.show()

# -----------------------------
# 8️⃣ spacing vs satellite count
# -----------------------------
plt.figure(figsize=(8,4))

plt.scatter(
    merged["visible_satellites"],
    merged["delta_f_hat_mhz"],
    alpha=0.7
)

plt.xlabel("Visible Starlink Count")
plt.ylabel("Comb Spacing [MHz]")
plt.title("Comb Spacing vs Satellite Count")

plt.tight_layout()
plt.savefig("comb_spacing_vs_sat_count.png")
plt.show()

# -----------------------------
# 9️⃣ summary
# -----------------------------
print("Visible satellites (median):", merged["visible_satellites"].median())
print("Max elevation (median):", merged["max_elevation"].median())
print("Mean comb spacing (delta_f_hat_mhz):", merged["delta_f_hat_mhz"].mean())
print("Std comb spacing (delta_f_hat_mhz):", merged["delta_f_hat_mhz"].std())
print(merged["delta_f_hat_mhz"].describe())
print(merged["delta_f_hat_mhz"].quantile([0.1,0.25,0.5,0.75,0.9]))

print("Comb spacing value counts:")
print(merged["delta_f_hat_mhz"].value_counts())

# Comb peak stacking: all detected comb peak frequencies histogram
if 'peak_freq' in merged.columns:
    all_peaks = []
    for pf in merged['peak_freq']:
        if isinstance(pf, list):
            all_peaks.extend(pf)
        elif isinstance(pf, np.ndarray):
            all_peaks.extend(pf.tolist())
    all_peaks = np.array(all_peaks)
    plt.figure(figsize=(8,4))
    plt.hist(all_peaks, bins=100, color='purple', edgecolor='k')
    plt.xlabel('Frequency [MHz]')
    plt.ylabel('Stacked Peak Count')
    plt.title('Comb Peak Stacking')
    plt.tight_layout()
    plt.savefig('comb_peak_stacking.png')
    plt.show()

# Family clustering: spacing vs linewidth DBSCAN cluster plot
features = ['delta_f_hat_mhz', 'linewidth_mhz']
X = merged[features].values
X = np.nan_to_num(X, nan=0.0)
clustering = DBSCAN(eps=0.5, min_samples=3).fit(X)
merged['family_id'] = clustering.labels_
plt.figure(figsize=(8,6))
for fam in np.unique(merged['family_id']):
    sel = (merged['family_id']==fam)
    plt.scatter(merged.loc[sel, 'delta_f_hat_mhz'], merged.loc[sel, 'linewidth_mhz'], label=f'Family {fam}', alpha=0.7)
plt.xlabel('delta_f_hat [MHz]')
plt.ylabel('linewidth [MHz]')
plt.title('Comb Family Clustering (Spacing vs Linewidth)')
plt.legend()
plt.tight_layout()
plt.savefig('comb_family_clustering.png')
plt.show()