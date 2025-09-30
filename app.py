# app.py
# Streamlit app to simulate models of choice under uncertainty
# Models: Expected Value (EV), Expected Utility (EU), Prospect Theory (PT), Normalization models

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Choice Under Uncertainty Models", layout="wide")
st.title("Choice Under Uncertainty: EV · EU · Prospect Theory · Normalization")

# -----------------------------
# Helper math functions
# -----------------------------
def crra_utility(x, r):
    x = np.array(x)
    x_safe = np.maximum(x, 1e-9)
    if np.isclose(r, 1.0):
        return np.log(x_safe)
    return (np.power(x_safe, 1.0 - r) - 1.0) / (1.0 - r)

def pt_value(x, alpha=0.88, beta=0.88, lamb=2.25):
    x = np.array(x)
    v = np.zeros_like(x)
    gains = x >= 0
    v[gains] = np.power(x[gains], alpha)
    v[~gains] = -lamb * np.power(-x[~gains], beta)
    return v

def prelec_weight(p, delta=1.0, gamma=0.7):
    p = np.clip(np.array(p), 1e-9, 1 - 1e-9)
    return np.exp(-np.power(delta * (-np.log(p)), gamma))

def identity_weight(p):
    return np.array(p)

def divisive_normalization(u, k=1.0, context_mean=None):
    if context_mean is None:
        context_mean = np.mean(u)
    return u / (k + context_mean), context_mean

def range_normalization(u, eps=1e-6):
    umin, umax = np.min(u), np.max(u)
    return (u - umin) / (umax - umin + eps), (umin, umax)

def luce_choice_prob(v_a, v_b):
    v_a = np.maximum(v_a, 0.0)
    v_b = np.maximum(v_b, 0.0)
    denom = v_a + v_b + 1e-12
    return v_a / denom

# -----------------------------
# Sidebar controls
# -----------------------------
model = st.sidebar.selectbox(
    "Select model",
    ["Expected Value (EV)", "Expected Utility (EU)", "Prospect Theory (PT)", "Normalization"],
)

st.sidebar.markdown("---")
value_range = st.sidebar.slider("Outcome range (for value plots)", -100.0, 100.0, (-50.0, 100.0))
num_points = st.sidebar.slider("Resolution (points)", 100, 1000, 400, step=50)

x = np.linspace(value_range[0], value_range[1], num_points)
p = np.linspace(1e-4, 1 - 1e-4, num_points)

# -----------------------------
# Models
# -----------------------------
if model == "Expected Value (EV)":
    st.header("Expected Value (EV)")
    st.markdown(
        r"""
        **Value function**: $v(x) = x$  
        **Probability function**: $w(p) = p$  
        **Expected Value**: $EV = v(x) \times w(p) = x \times p$
        """
    )

    # Single gamble calculator
    st.subheader("Single gamble calculator")
    c1, c2 = st.columns(2)
    with c1:
        x0 = st.number_input("Outcome (x)", value=50.0)
    with c2:
        p0 = st.slider("Probability (p)", 0.0, 1.0, 0.5, step=0.01)
    ev0 = x0 * p0
    st.markdown(rf"**EV = {x0:.4g} × {p0:.4g} = {ev0:.4g}**")

    # Surface (EV)
    v = x
    w = identity_weight(p)
    X, P = np.meshgrid(x, p)
    EV_grid = X * P

    st.subheader("Expected Value Surface: EV = x * p")
    fig3, ax3 = plt.subplots()
    cs = ax3.contourf(X, P, EV_grid, levels=30)
    fig3.colorbar(cs, ax=ax3, label="EV")
    ax3.set_xlabel("Outcome x")
    ax3.set_ylabel("Probability p")
    ax3.set_title("Expected Value Surface")
    st.pyplot(fig3)

elif model == "Expected Utility (EU)":
    st.header("Expected Utility (EU) – CRRA")
    r = st.sidebar.slider("Risk aversion r (CRRA)", 0.0, 2.0, 0.5, 0.01)

    st.markdown(
        r"""
        **Utility function**:  
        $$
        u(x) = \begin{cases}
        \dfrac{x^{1-r} - 1}{1 - r}, & r \neq 1,\\\\[6pt]
        \ln x, & r = 1
        \end{cases} \quad (x>0)
        $$
        **Probability function**: $w(p) = p$  
        **Expected Utility**: $EU = u(x) \times w(p)$
        """
    )

    # Single gamble calculator
    st.subheader("Single gamble calculator")
    c1, c2 = st.columns(2)
    with c1:
        x0 = st.number_input("Outcome (x>0)", value=50.0, min_value=0.0)
    with c2:
        p0 = st.slider("Probability (p)", 0.0, 1.0, 0.5, step=0.01, key="eu_p0")
    if x0 <= 0:
        st.warning("EU with CRRA is defined for x>0. Using a tiny positive value for computation.")
    u0 = crra_utility(np.array([max(x0, 1e-9)]), r)[0]
    eu0 = u0 * p0
    st.markdown(rf"**u(x) = {u0:.4g} · EU = {u0:.4g} × {p0:.4g} = {eu0:.4g}**")

elif model == "Prospect Theory (PT)":
    st.header("Prospect Theory (Tversky & Kahneman 1992 + Prelec weighting)")
    alpha = st.sidebar.slider("α (gains curvature)", 0.1, 1.0, 0.88, 0.01)
    beta = st.sidebar.slider("β (losses curvature)", 0.1, 1.0, 0.88, 0.01)
    lamb = st.sidebar.slider("λ (loss aversion)", 1.0, 5.0, 2.25, 0.05)
    delta = st.sidebar.slider("Prelec δ (scale)", 0.1, 2.0, 1.0, 0.05)
    gamma = st.sidebar.slider("Prelec γ (curvature)", 0.1, 2.0, 0.7, 0.05)

    st.markdown(
        r"""
        **Value function**:  
        $$
        v(x) = \begin{cases}
        x^{\alpha}, & x \ge 0, \\\\
        -\lambda\, (-x)^{\beta}, & x < 0
        \end{cases}
        $$
        **Probability weighting (Prelec)**:  
        $$
        w(p) = \exp\!\left(-\big(\delta\,(-\ln p)\big)^{\gamma}\right)
        $$
        **Prospect Value**: $PV = v(x) \times w(p)$
        """
    )

    # Single gamble calculator
    st.subheader("Single gamble calculator")
    c1, c2 = st.columns(2)
    with c1:
        x0 = st.number_input("Outcome (x)", value=50.0, key="pt_x0")
    with c2:
        p0 = st.slider("Probability (p)", 0.0, 1.0, 0.5, step=0.01, key="pt_p0")
    v0 = pt_value(np.array([x0]), alpha, beta, lamb)[0]
    w0 = prelec_weight(np.array([p0]), delta, gamma)[0]
    pv0 = v0 * w0
    st.markdown(rf"**v(x) = {v0:.4g}, w(p) = {w0:.4g}, PV = {pv0:.4g}**")

elif model == "Normalization":
    st.header("Normalization Models")
    st.markdown(
        r"""
        We compute **normalized values** for two options \(A\) and \(B\) and the **Luce choice probability**:
        $$
        P(A) = \frac{V_A}{V_A + V_B}.
        $$
        Choose a normalization rule and base value function below.
        """
    )

    # Sidebar params for normalization
    norm_kind = st.sidebar.selectbox("Normalization type", ["Divisive", "Range"])
    base_value_kind = st.sidebar.selectbox("Base value function", ["Linear v(x)=x", "CRRA u(x)"])
    if base_value_kind == "CRRA u(x)":
        r = st.sidebar.slider("Risk aversion r (CRRA)", 0.0, 2.0, 0.5, 0.01)
    k = st.sidebar.slider("Divisive constant k", 0.0, 10.0, 1.0, 0.1)
    eps = st.sidebar.slider("Range ε (stability)", 1e-6, 0.1, 1e-3)

    # Calculator for A vs B
    st.subheader("Two-option calculator (A vs B)")
    c1, c2 = st.columns(2)
    with c1:
        x_A = st.number_input("Option A outcome (x_A)", value=20.0)
    with c2:
        x_B = st.number_input("Option B outcome (x_B)", value=0.0)

    # Base utility for the two outcomes
    if base_value_kind == "Linear v(x)=x":
        uA, uB = float(x_A), float(x_B)
        base_eq = r"$v(x)=x$"
    else:
        uA = crra_utility(np.array([max(x_A, 1e-9)]), r)[0]
        uB = crra_utility(np.array([max(x_B, 1e-9)]), r)[0]
        base_eq = r"$u(x)=\begin{cases}(x^{1-r}-1)/(1-r), & r\ne 1 \\ \ln x, & r=1\end{cases}$"

    # Normalize using the *pair* as the context
    if norm_kind == "Divisive":
        context_mean = np.mean([uA, uB])
        VA = uA / (k + context_mean)
        VB = uB / (k + context_mean)
        norm_eq = r"$V_{\text{norm}}(x)=\dfrac{u(x)}{k + \mathrm{mean}[u_A,u_B]}$"
    else:  # Range
        umin, umax = min(uA, uB), max(uA, uB)
        VA = (uA - umin) / (umax - umin + eps)
        VB = (uB - umin) / (umax - umin + eps)
        norm_eq = r"$V_{\text{norm}}(x)=\dfrac{u(x)-u_{\min}}{u_{\max}-u_{\min}+\varepsilon}$,\; u_{\min/\max}\ \text{from } \{u_A,u_B\}$"

    PA = luce_choice_prob(VA, VB)
    PB = 1.0 - PA

    st.markdown(
        rf"""
        **Base value:** {base_eq}  
        **Normalization:** {norm_eq}  
        **Luce probability:** $P(A)=\dfrac{{V_A}}{{V_A+V_B}}$
        """
    )
    st.markdown(
        rf"""
        **Computed values:**  
        - $u_A = {uA:.4g}$, $u_B = {uB:.4g}$  
        - $V_A = {VA:.4g}$, $V_B = {VB:.4g}$  
        - **P(A) = {PA:.3f}**, P(B) = {PB:.3f}
        """
    )

    # ---- New: Plot P(A) vs x_A (holding x_B fixed) ----
    st.subheader("P(A) as a function of x_A (x_B fixed)")
    # Grid for x_A based on the global value_range
    xA_grid = np.linspace(value_range[0], value_range[1], 300)

    if base_value_kind == "Linear v(x)=x":
        uA_grid = xA_grid
        # VB_const already computed above for current x_B:
        VB_const = VB
        if norm_kind == "Divisive":
            C = np.mean([uA, uB])  # keep context consistent with calculator
            VA_grid = uA_grid / (k + C)
        else:  # Range
            umin, umax = min(uA, uB), max(uA, uB)
            VA_grid = (uA_grid - umin) / (umax - umin + eps)
    else:
        # ensure positive for CRRA
        xA_grid_pos = np.maximum(xA_grid, 1e-9)
        uA_grid = crra_utility(xA_grid_pos, r)
        VB_const = VB
        if norm_kind == "Divisive":
            C = np.mean([uA, uB])
            VA_grid = uA_grid / (k + C)
        else:
            umin, umax = min(uA, uB), max(uA, uB)
            VA_grid = (uA_grid - umin) / (umax - umin + eps)

    P_grid = luce_choice_prob(VA_grid, VB_const)

    figP, axP = plt.subplots()
    axP.plot(xA_grid, P_grid)
    axP.set_xlabel("Option A outcome x_A")
    axP.set_ylabel("P(A)")
    axP.set_title("Luce P(A) vs x_A (x_B fixed)")
    st.pyplot(figP)

# -----------------------------
# Footer
# -----------------------------
with st.expander("How to run this app"):
    st.markdown(
        """
        1. Save this file as `app.py`.
        2. Install dependencies: `pip install streamlit matplotlib numpy`
        3. Run: `streamlit run app.py`
        4. Use the sidebar to switch models and adjust parameters.
        """
    )
