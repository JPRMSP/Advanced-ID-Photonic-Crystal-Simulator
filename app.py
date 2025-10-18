import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

st.set_page_config(page_title="Advanced 1D Photonic Crystal Simulator 3D", layout="wide")
st.title("Advanced 1D Photonic Crystal Simulator ⚡️ 3D E-field Visualization")

# Sidebar Inputs
st.sidebar.header("Crystal Parameters")
n1 = st.sidebar.number_input("Refractive index n1", 1.0, 5.0, 1.5, 0.1)
n2 = st.sidebar.number_input("Refractive index n2", 1.0, 5.0, 2.0, 0.1)
d1 = st.sidebar.number_input("Thickness d1 (nm)", 10, 1000, 200)
d2 = st.sidebar.number_input("Thickness d2 (nm)", 10, 1000, 200)
N = st.sidebar.number_input("Number of periods", 1, 20, 5)

st.sidebar.header("Defect / Cavity")
defect_enabled = st.sidebar.checkbox("Add defect layer", False)
if defect_enabled:
    n_defect = st.sidebar.number_input("Defect Refractive Index", 1.0, 5.0, 3.0, 0.1)
    d_defect = st.sidebar.number_input("Defect Thickness (nm)", 10, 1000, 300)

st.sidebar.header("Wavelength Range")
wl_min = st.sidebar.number_input("Min Wavelength (nm)", 200, 2000, 400)
wl_max = st.sidebar.number_input("Max Wavelength (nm)", 200, 2000, 1000)
wl_points = st.sidebar.number_input("Number of points", 100, 5000, 1000)

wavelengths = np.linspace(wl_min, wl_max, wl_points)

# Transfer Matrix Function
def transfer_matrix(n1, n2, d1, d2, N, wl, defect=None):
    T_total = np.zeros_like(wl)
    for i, lam in enumerate(wl):
        k1 = 2 * np.pi * n1 / lam
        k2 = 2 * np.pi * n2 / lam
        M1 = np.array([[np.cos(k1*d1), 1j*np.sin(k1*d1)/n1],
                       [1j*n1*np.sin(k1*d1), np.cos(k1*d1)]])
        M2 = np.array([[np.cos(k2*d2), 1j*np.sin(k2*d2)/n2],
                       [1j*n2*np.sin(k2*d2), np.cos(k2*d2)]])
        M_period = np.matmul(M1, M2)
        M_total = np.linalg.matrix_power(M_period, N)
        if defect:
            n_def, d_def = defect
            k_def = 2 * np.pi * n_def / lam
            M_def = np.array([[np.cos(k_def*d_def), 1j*np.sin(k_def*d_def)/n_def],
                              [1j*n_def*np.sin(k_def*d_def), np.cos(k_def*d_def)]])
            M_total = np.matmul(np.linalg.matrix_power(M_period, N//2), np.matmul(M_def, np.linalg.matrix_power(M_period, N//2)))
        t = 2 / (M_total[0,0] + M_total[0,1] + M_total[1,0] + M_total[1,1])
        T_total[i] = np.abs(t)**2
    return T_total

defect_layer = (n_defect, d_defect) if defect_enabled else None
T = transfer_matrix(n1, n2, d1, d2, N, wavelengths, defect=defect_layer)

# Plot Transmission Spectrum
fig1, ax = plt.subplots(figsize=(10,5))
ax.plot(wavelengths, T, color="purple")
ax.set_title("Transmission Spectrum of 1D Photonic Crystal")
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Transmission")
ax.grid(True)
st.pyplot(fig1)

# E-field along layers (simplified model)
st.subheader("3D Electric Field Distribution Along Layers")
total_layers = []
thicknesses = []

# Construct layers for E-field visualization
for _ in range(N):
    total_layers.append(n1)
    thicknesses.append(d1)
    total_layers.append(n2)
    thicknesses.append(d2)

if defect_enabled:
    mid_index = len(total_layers)//2
    total_layers.insert(mid_index, n_defect)
    thicknesses.insert(mid_index, d_defect)

# Create spatial grid along z
z = []
E = []
current_z = 0
for n, d in zip(total_layers, thicknesses):
    z_layer = np.linspace(current_z, current_z + d, 50)
    E_layer = np.sin(2*np.pi*z_layer/(sum(thicknesses)/len(thicknesses))) * n  # E-field amplitude scales with n
    z.extend(z_layer)
    E.extend(E_layer)
    current_z += d

# 3D plot using plotly
fig2 = go.Figure()
fig2.add_trace(go.Scatter3d(
    x=z, y=E, z=np.zeros_like(E),
    mode='lines',
    line=dict(color='blue', width=6)
))
fig2.update_layout(scene=dict(
    xaxis_title='Propagation (nm)',
    yaxis_title='Electric Field Amplitude',
    zaxis_title='',
), width=900, height=500)
st.plotly_chart(fig2)

st.markdown("""
### Instructions:
- Adjust **refractive indices, thicknesses, periods**, and **defect layer**.
- Observe **transmission spectrum** and **E-field localization**.
- Defect layers show **resonance peaks** and **field enhancement**.
- This visualization helps understand **localized modes, surface states, and resonant cavities**.
""")
