import numpy as np
import sympy as sp
import streamlit as st

st.title("Axial Flux Motor Designer")
st.write(
    "Size an axial flux motor for a target torque/power with quick geometry estimates, pole counts, "
    "and magnet sizing. These back-of-the-envelope calculators reference the coreless designs in "
    "Batzel et al. and the sizing walkthrough by Caden Kraft."
)

st.markdown(
    "References: "
    "[Batzel et al. (axial-flux design)](https://cd14.ijme.us/papers/088__Todd%20D.%20Batzel,%20Andrew%20M.%20Skraba,%20Ray%20D.%20Massi.pdf) | "
    "[Caden Kraft coreless motor guide](https://cadenkraft.com/designing-a-coreless-axial-flux-motor-part-1/)"
)

st.markdown(
    """
    This page uses a shear-stress-based torque estimate for an axial disk pair and a target electrical
    frequency heuristic to suggest pole counts. It is intended for preliminary sizing; slotting,
    saturation, and thermal limits must still be validated with detailed FEA/loss models.
    """
)

with st.expander("Assumptions and formulas"):
    st.latex(r"\\tau = N_{d}\\,\\sigma_{shear}\\,A_{rotor}\\,r_{mean}")
    st.latex(r"A_{rotor} = \\pi (r_o^2 - r_i^2),\\quad r_{mean}=\\frac{2}{3}\\frac{r_o^3-r_i^3}{r_o^2-r_i^2}")
    st.latex(r"f_e = \\frac{p}{2}\\frac{n_{rpm}}{60}")
    st.markdown(
        "- $N_d$ is the number of active discs (1 for single-sided, 2 for double-sided).\n"
        "- $\\sigma_{shear}$ is the air-gap shear stress (Pa).\n"
        "- $p$ is the pole count (even). The heuristic below chooses $p$ so the electrical frequency"
        " stays near the target you set."
    )

# Inputs
col1, col2, col3 = st.columns(3)
with col1:
    target_torque = st.number_input("Target torque (N·m)", min_value=0.1, value=20.0, step=1.0)
    mech_power = st.number_input("Target mechanical power (W)", min_value=10.0, value=2000.0, step=50.0)
with col2:
    speed_rpm = st.number_input("Operating speed (rpm)", min_value=50.0, value=1200.0, step=50.0)
    shear_stress = st.number_input("Assumed air-gap shear stress (Pa)", min_value=1000.0, value=30000.0, step=1000.0)
with col3:
    disc_count = st.number_input("Active disc count (1=single, 2=double)", min_value=1, max_value=2, value=2)
    inner_ratio = st.slider("Inner-to-outer radius ratio", 0.2, 0.8, 0.4, 0.05)

col4, col5 = st.columns(2)
with col4:
    target_fe = st.number_input("Target electrical frequency (Hz)", min_value=50.0, value=250.0, step=10.0)
with col5:
    axial_stack = st.number_input("Effective axial stack (m) for shear stress", min_value=0.005, value=0.02, step=0.005)

# Solve outer radius using Sympy
r_o = sp.symbols("r_o", positive=True, real=True)
r_i_expr = inner_ratio * r_o
r_mean_expr = (sp.Rational(2, 3) * (r_o ** 3 - r_i_expr ** 3) / (r_o ** 2 - r_i_expr ** 2))
area_expr = sp.pi * (r_o ** 2 - r_i_expr ** 2)
# Include axial stack in the shear stress term for a linear scaling of available area
nominal_torque_expr = disc_count * shear_stress * area_expr * r_mean_expr * axial_stack
torque_solution = sp.solve(sp.Eq(nominal_torque_expr, target_torque), r_o)

outer_radius = None
for sol in torque_solution:
    if sol.is_real and sol > 0:
        outer_radius = float(sol)
        break

if outer_radius is None:
    st.error("Could not solve for an outer radius with the provided parameters. Try reducing torque or shear stress.")
    outer_radius = float(sp.sqrt(target_torque / (shear_stress * sp.pi)))

inner_radius = inner_ratio * outer_radius
r_mean = float(r_mean_expr.subs(r_o, outer_radius))

# Pole suggestion
speed_hz = speed_rpm / 60.0
suggested_poles = int(max(2, 2 * round(target_fe / max(speed_hz, 1e-6))))
if suggested_poles % 2 != 0:
    suggested_poles += 1

# Magnet sizing estimate
magnet_arc = 2 * np.pi * r_mean / suggested_poles
magnet_radial = (outer_radius - inner_radius) / 2

# Derived outputs
mechanical_freq = speed_rpm / 60.0
estimated_power = 2 * np.pi * mechanical_freq * target_torque

st.subheader("Sizing results")
res_cols = st.columns(3)
res_cols[0].metric("Outer radius (m)", f"{outer_radius:.3f}")
res_cols[1].metric("Inner radius (m)", f"{inner_radius:.3f}")
res_cols[2].metric("Mean radius (m)", f"{r_mean:.3f}")

res_cols2 = st.columns(3)
res_cols2[0].metric("Suggested poles", f"{suggested_poles}")
res_cols2[1].metric("Magnet arc length (m)", f"{magnet_arc:.3f}")
res_cols2[2].metric("Magnet radial width (m)", f"{magnet_radial:.3f}")

st.caption(
    "Pole count is chosen so electrical frequency $f_e$ stays near the target. Magnet dimensions assume equally spaced "
    "surface magnets over the mean radius band."
)

summary_data = {
    "Parameter": [
        "Target torque (N·m)",
        "Target power (W)",
        "Operating speed (rpm)",
        "Electrical frequency target (Hz)",
        "Assumed shear stress (Pa)",
        "Disc count",
        "Outer radius (m)",
        "Inner radius (m)",
        "Mean radius (m)",
        "Suggested poles",
        "Magnet arc length (m)",
        "Magnet radial width (m)",
    ],
    "Value": [
        target_torque,
        mech_power,
        speed_rpm,
        target_fe,
        shear_stress,
        disc_count,
        outer_radius,
        inner_radius,
        r_mean,
        suggested_poles,
        magnet_arc,
        magnet_radial,
    ],
}

st.dataframe(summary_data, hide_index=True, use_container_width=True)

with st.expander("Symbolic torque expression"):
    st.write("Outer radius solution (first positive root):")
    st.latex(sp.latex(sp.simplify(nominal_torque_expr)))
    st.write(f"Solved r_o ≈ {outer_radius:.4f} m with r_i = {inner_ratio:.2f} r_o")

st.success(
    "Use these dimensions as starting points for detailed electromagnetic and thermal analysis. "
    "Validate slot-pole combinations, cooling, and mechanical stresses with your full design workflow."
)
