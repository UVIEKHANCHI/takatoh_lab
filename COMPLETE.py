import pandas as pd
import numpy as np
import plotly.graph_objects as go
import webbrowser
from pathlib import Path

# ============================================================
# PATHS — edit these if your files live elsewhere
# ============================================================
cells_csv  = "/Users/urvishkhanchi/Downloads/cells_physical.csv"
obj_path   = "/Users/urvishkhanchi/Downloads/Chat-Cre_Ai14_NoDownSample_counting_Original_clean_InsideMeshRepaired_20250715.obj"
output_file = "/Users/urvishkhanchi/PycharmProjects/PythonProject/tongue_cells_interactive.html"

# ============================================================
# OFFICIAL TOP-VIEW ANGLES (DO NOT CHANGE)
# These are the project-standard rotations we must preserve.
# - Yaw: rotation about the Z-axis (horizontal “left/right” turn)
# - Pitch: rotation about the X-axis (vertical “up/down” tilt)
# ============================================================
YAW_OFFICIAL   = 4.56   # degrees, about Z
PITCH_OFFICIAL = 5.33   # degrees, about X

# ============================================================
# SAGITTAL EXTRA TWEAKS (ONLY these may be edited if needed)
# These are *additional* small adjustments on top of the official
# top-view angles when switching to a sagittal (side) view.
# Keep them 0.0 unless you need minor fine-tuning for the side view.
# ============================================================
SAGITTAL_YAW_DELTA   = 0.0    # degrees, about Z (usually small)
SAGITTAL_PITCH_DELTA = 0.0    # degrees, about X (fine leveling)

# ============================================================
# LOAD DATA
# - cells_csv must contain columns: x, y, z
# - obj file is the tongue mesh (vertices + faces)
# ============================================================
cells_df = pd.read_csv(cells_csv)

def load_obj(filename):
    """
    Minimal OBJ loader:
    - Reads 'v' lines as vertex positions (x,y,z)
    - Reads 'f' lines as faces (supports n-gons; triangulated into fans)
    Returns:
        vertices: (N,3) float array
        faces:    list of [i,j,k] integer triplets (0-based)
    """
    vertices, faces = [], []
    with open(filename, "r") as f:
        for line in f:
            if line.startswith("v "):
                # Vertex line: "v x y z"
                _, x, y, z, *rest = line.strip().split()
                vertices.append([float(x), float(y), float(z)])
            elif line.startswith("f "):
                # Face line: "f a b c ..." (1-based indices; may include slashes)
                parts = [p.split("/")[0] for p in line.strip().split()[1:]]
                idxs = [int(p) - 1 for p in parts]  # convert to 0-based
                # Triangulate any polygon (fan triangulation)
                for k in range(1, len(idxs) - 1):
                    faces.append([idxs[0], idxs[k], idxs[k + 1]])
    return np.asarray(vertices, float), faces

# Original (unrotated) data arrays
verts0, faces = load_obj(obj_path)
cells0 = cells_df[["x", "y", "z"]].to_numpy(float)

# ============================================================
# ROTATION UTILITIES (no PCA)
# - R_x, R_y, R_z build rotation matrices for given angles (deg)
# - rotate_about_point applies a rotation about a chosen origin
# ============================================================
def R_x(deg):
    th = np.deg2rad(deg); c, s = np.cos(th), np.sin(th)
    return np.array([[1, 0, 0],
                     [0, c,-s],
                     [0, s, c]], float)

def R_y(deg):
    th = np.deg2rad(deg); c, s = np.cos(th), np.sin(th)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]], float)

def R_z(deg):
    th = np.deg2rad(deg); c, s = np.cos(th), np.sin(th)
    return np.array([[ c,-s, 0],
                     [ s, c, 0],
                     [ 0, 0, 1]], float)

def rotate_about_point(P, R, origin):
    """
    Apply rotation R (3x3) to points P (N,3) about a given origin (3,).
    Returns rotated points with same shape as P.
    """
    P = np.asarray(P, float); origin = np.asarray(origin, float)
    return (P - origin) @ R.T + origin

# ============================================================
# MAIN-AXIS LINE FOR VISUAL INSPECTION (no PCA)
# We compute a simple least-squares direction in the XY plane
# to draw a line through the centroid. This is just a visual guide.
# ============================================================
# Common origin (centroid) of all points — helps keep mesh & cells aligned
center = np.vstack([verts0, cells0]).mean(0)

# SVD on XY (2D) for main direction; purely for a guide line
XY = verts0[:, :2] - center[:2]
U, S, Vt = np.linalg.svd(XY, full_matrices=False)
axis2d = Vt[0]  # principal direction in XY
axis3d = np.array([axis2d[0], axis2d[1], 0.0], float)
axis3d /= np.linalg.norm(axis3d) + 1e-12  # normalize safely

# Make a line segment around the centroid sized to the mesh footprint
bbox = verts0.max(0) - verts0.min(0)
L = 0.6 * float(np.linalg.norm(bbox[:2]))  # length scale from XY size
p1 = center - L * axis3d
p2 = center + L * axis3d

# ============================================================
# ROTATION PIPELINES
# - apply_yaw_pitch: applies yaw (Z) then pitch (X) about the centroid
# - official_top_view: applies the locked official angles
# - sagittal_official_view: official + sagittal deltas (for side view)
# ============================================================
def apply_yaw_pitch(verts, cells, yaw_deg, pitch_deg, origin):
    """
    Apply yaw then pitch using the fixed order:
        R_total = R_x(pitch_deg) @ R_z(yaw_deg)
    The same transform is applied to:
      - mesh vertices
      - cell points
      - the axis line endpoints (for consistent visualization)
    """
    R_total = R_x(pitch_deg) @ R_z(yaw_deg)  # order matters
    v = rotate_about_point(verts, R_total, origin)
    c = rotate_about_point(cells, R_total, origin)
    a = rotate_about_point(np.vstack([p1, p2]), R_total, origin)
    return v, c, a

def official_top_view():
    """Return arrays rotated to the OFFICIAL top-view angles."""
    return apply_yaw_pitch(verts0, cells0, YAW_OFFICIAL, PITCH_OFFICIAL, center)

def sagittal_official_view():
    """
    Return arrays rotated to OFFICIAL top-view + sagittal deltas.
    This preserves the official angles and adds only the extra tweaks
    needed to make the side profile look perfect.
    """
    yaw   = YAW_OFFICIAL   + SAGITTAL_YAW_DELTA
    pitch = PITCH_OFFICIAL + SAGITTAL_PITCH_DELTA
    return apply_yaw_pitch(verts0, cells0, yaw, pitch, center)

# ============================================================
# BUILD INITIAL FIGURE/TRACES
# We start from the OFFICIAL top view to ensure consistency.
# ============================================================
v_init, c_init, a_init = official_top_view()

mesh = go.Mesh3d(
    x=v_init[:,0], y=v_init[:,1], z=v_init[:,2],
    i=[f[0] for f in faces], j=[f[1] for f in faces], k=[f[2] for f in faces],
    color="lightgray", opacity=0.18, flatshading=True, name="Tongue Mesh",
    lighting=dict(ambient=0.9, diffuse=0.9, specular=0.05, roughness=1.0)
)
cells_sc = go.Scatter3d(
    x=c_init[:,0], y=c_init[:,1], z=c_init[:,2],
    mode="markers", marker=dict(size=2.6, opacity=0.95, color="magenta"),
    name="Cells"
)
axis_ln = go.Scatter3d(
    x=[a_init[0,0], a_init[1,0]],
    y=[a_init[0,1], a_init[1,1]],
    z=[a_init[0,2], a_init[1,2]],
    mode="lines+markers", line=dict(width=6), marker=dict(size=3),
    name="Main axis (top view fit)"
)

fig = go.Figure(data=[mesh, cells_sc, axis_ln])

# ============================================================
# UPDATE HELPERS
# These return the dicts used by Plotly "update" buttons.
# They rebuild only the data arrays; layout (camera) set separately.
# ============================================================
def make_update_from_arrays(v, c, a):
    """Package updated coordinates for Mesh3d, Scatter3d, and axis line."""
    return {
        "x": [v[:,0], c[:,0], a[:,0]],
        "y": [v[:,1], c[:,1], a[:,1]],
        "z": [v[:,2], c[:,2], a[:,2]],
    }

def update_official_top():
    """Updater for OFFICIAL top-view orientation."""
    v, c, a = official_top_view()
    return make_update_from_arrays(v, c, a)

def update_sagittal_official():
    """Updater for OFFICIAL top-view + sagittal delta orientation."""
    v, c, a = sagittal_official_view()
    return make_update_from_arrays(v, c, a)

# ============================================================
# BUTTONS (CAMERA + ORIENTATION PRESETS)
# - Top View: applies OFFICIAL angles and a top camera
# - Sagittal Official View: applies OFFICIAL+delta and a side camera
# - Reset buttons: quick return to each official orientation
# NOTE: Buttons only change what you see; your raw data stays intact.
# ============================================================
top_cam     = dict(eye=dict(x=0,   y=0,   z=3.0), up=dict(x=0, y=1, z=0))
side_cam    = dict(eye=dict(x=3.0, y=0,   z=0),   up=dict(x=0, y=0, z=1))
default_cam = dict(eye=dict(x=1.6, y=1.6, z=1.2), up=dict(x=0, y=0, z=1))

fig.update_layout(
    scene=dict(
        xaxis=dict(visible=False),  # hide axes for a clean presentation
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
        aspectmode="data"          # equal scale on x/y/z based on data
    ),
    margin=dict(l=0, r=0, t=0, b=0),
    updatemenus=[
        # View buttons (camera + orientation)
        dict(
            type="buttons", direction="right",
            x=0.02, xanchor="left", y=1.10, yanchor="top", showactive=True,
            buttons=[
                dict(label="Default View", method="relayout",
                     args=[{"scene.camera": default_cam}]),
                dict(label="Official Top View", method="update",
                     args=[update_official_top(), {"scene.camera": top_cam}]),
                dict(label="Sagittal Official View", method="update",
                     args=[update_sagittal_official(), {"scene.camera": side_cam}]),
            ]
        ),
        # Quick resets (guarantee exact official numbers before exporting)
        dict(
            type="buttons", direction="right",
            x=0.02, xanchor="left", y=1.03, yanchor="top", showactive=True,
            buttons=[
                dict(label=f"Reset to OFFICIAL Top ({YAW_OFFICIAL}°, {PITCH_OFFICIAL}°)",
                     method="update",
                     args=[update_official_top(), {"scene.camera": top_cam}]),
                dict(label=f"Reset to Sagittal ({YAW_OFFICIAL+SAGITTAL_YAW_DELTA:.2f}°, "
                            f"{PITCH_OFFICIAL+SAGITTAL_PITCH_DELTA:.2f}°)",
                     method="update",
                     args=[update_sagittal_official(), {"scene.camera": side_cam}]),
            ]
        ),
    ]
)

# Start in the OFFICIAL top view so screenshots are consistent
fig.update_layout(scene_camera=top_cam)

# ============================================================
# SAVE HTML + OPEN IN BROWSER
# ============================================================
Path(output_file).parent.mkdir(parents=True, exist_ok=True)
fig.write_html(output_file)
try:
    webbrowser.get("safari").open(f"file://{output_file}")
except webbrowser.Error:
    # Fallback: default browser if Safari controller isn't available
    webbrowser.open(f"file://{output_file}")

print(f"✅ Saved & opened: {output_file}")
