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
    return u / (k + context_mean)

def range_normalization(u, eps=1e-6):
    umin, umax = np.min(u), np.max(u)
    return (u - umin) / (umax - umin + eps)

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

    # Functions & surfaces
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
    gamma = st.sidebar.slider("Prelec γ (curvature)", 0.1, 2.0, 0.7, 0.0