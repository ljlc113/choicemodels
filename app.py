import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Choice Models & Normalization", layout="wide")

# ---------------------------------------
# Sidebar Navigation
# ---------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    (
        "Overview",
        "Expected Value (EV)",
        "Expected Utility (EU)",
        "Prospect Theory (PT)",
        "Normalization Techniques",
        "Normalization Comparisons",
    ),
)

# ---------------------------------------
# Helper utilities
# ---------------------------------------
def _two_cols():
    return st.columns(2)


def _show_eq(title: str, latex: str):
    st.markdown(f"### {title}")
    st.latex(latex)


def _plot_simple(x, y, xlabel, ylabel, title):
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    st.pyplot(fig, clear_figure=True)


def _plot_multi(x, ys, labels, xlabel, ylabel, title):
    fig, ax = plt.subplots()
    for y, lab in zip(ys, labels):
        ax.plot(x, y, label=lab)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    st.pyplot(fig, clear_figure=True)


# ---------------------------------------
# Overview
# ---------------------------------------
if page == "Overview":
    st.title("Choice Models under Uncertainty & Normalization Techniques")
    st.markdown(
        """
        This interactive app covers three classic decision models under risk/uncertainty and four 
        normalization schemes used in value coding.

        **Decision Models**
        - **Expected Value (EV):** linear utility, linear probability.
        - **Expected Utility (EU):** nonlinear utility over outcomes.
        - **Prospect Theory (PT):** reference-dependent value and nonlinear probability weighting.

        **Normalization techniques** (applied in a choosing restaurants example):
        - **Range normalization → linear scaling, sensitive to min and max.**
        - **Divisive normalization → relative to the mean, not bounded.**
        - **Recurrent divisive normalization → bounded, compresses large values.**
        - **Adaptive gain / logistic value → nonlinear, highlights contrasts around the mean.**
        """
    )
    st.info("Use the sidebar to visit each page. Sliders let you change parameters and immediately see the equations and curves update.")

# ---------------------------------------
# Expected Value (EV)
# ---------------------------------------

if page == "Expected Value (EV)":
    st.title("Expected Value (EV)")
    st.markdown("EV assumes **linear utility** and **linear probability weighting**. It is computed by multiplying the value of an outcome by its probability.")

    _show_eq("EV of two-outcome lottery, where v1 = probability p and v2 = probability 1-p", r"EV = p \times v_1 + (1 - p) \times v_2")

    # Inputs for two-outcome lottery (user-controlled)
    st.subheader("Two-outcome Lottery (interactive)")
    v1 = st.slider("Value v1", -100.0, 100.0, 50.0, 1.0)
    v2 = st.slider("Value v2", -100.0, 100.0, 0.0, 1.0)
    p = st.slider("Probability p for v1", 0.0, 1.0, 0.5, 0.01)
    ev_value = p * v1 + (1 - p) * v2
    st.metric("Expected Value", f"{ev_value:.2f}")

    st.divider()
    st.subheader("Worked examples")
    # Example 1: Lottery ticket .01% chance to win 100,000
    p1 = 0.0001
    v1_1, v2_1 = 100_000.0, 0.0
    ev1 = p1 * v1_1 + (1 - p1) * v2_1

    # Example 2: 50% chance +55, 50% chance -50
    p2 = 0.5
    v1_2, v2_2 = 55.0, -50.0
    ev2 = p2 * v1_2 + (1 - p2) * v2_2

    colA, colB = st.columns(2)
    with colA:
        st.markdown("**Lottery ticket:** 0.01% chance to win $100,000; otherwise$0")
        st.latex(r"\\mathrm{EV} = 0.0001 \times 100{,}000 + 0.9999 \times 0 = 10")
        st.metric("EV", f"{ev1:.2f}")
        st.markdown("**What does this mean?** You should pursue this gamble, if the ticket costs less than 10 dollars.")
    with colB:
        st.markdown("**50–50 gamble:** +$55 with 50%, −$50 with 50%")
        st.latex(r"\\mathrm{EV} = 0.5 \times 55 + 0.5 \times (-50) = 2.5")
        st.metric("EV", f"{ev2:.2f}")
        st.markdown("**What does this mean?** You should pursue this gamble, because expected value is positive.")

    st.divider()
    st.subheader("Graphics of EV utility and probability weighting functions:")
    # Utility and probability equations with their graphs side by side
    col1, col2 = _two_cols()
    with col1:
        st.latex(r"u(x) = x")
        xr = np.linspace(-100, 100, 400)
        _plot_simple(xr, xr, "Outcome x", "Utility u(x)", "Linear utility: u(x)=x")

    with col2:
        st.latex(r"w(p) = p")
        pr = np.linspace(0, 1, 200)
        _plot_simple(pr, pr, "Probability p", "Weight w(p)", "Identity weighting: w(p)=p")

# ---------------------------------------
# Expected Utility (EU)
# ---------------------------------------
if page == "Expected Utility (EU)":
    st.set_page_config(page_title="Normalization Methods Comparison", layout="wide")

    st.title("Normalization Models of Value – Comparison")
    st.caption("Draft app that mirrors the behavior of your Colab snippet and lets you tweak inputs interactively.")

    # -----------------------------
    # Helper: parse arrays from text
    # -----------------------------
    def parse_array(s: str) -> np.ndarray:
        toks = [t for t in s.replace(",", " ").split() if t]
        try:
            return np.array([float(t) for t in toks], dtype=float)
        except Exception:
            return np.array([], dtype=float)

    # -----------------------------
    # On-page inputs
    # -----------------------------
    st.header("Inputs")

    def_v1 = "1 2 5 10"
    def_v2 = "1 5 9 10"

    col_in1, col_in2 = st.columns(2)
    with col_in1:
        v1_str = st.text_input("v1 (comma/space separated)", value=def_v1)
    with col_in2:
        v2_str = st.text_input("v2 (comma/space separated)", value=def_v2)

    col_in3, col_in4 = st.columns([1,1])
    with col_in3:
        slope = st.slider("Adaptive gain slope k", 0.05, 2.0, 0.7, 0.05)
    with col_in4:
        show_table = st.checkbox("Show numeric table", value=True)

    v1 = parse_array(v1_str)
    v2 = parse_array(v2_str)

    # Guardrail
    if v1.size == 0 or v2.size == 0:
        st.error("Please provide valid numeric arrays for v1 and v2.")
        st.stop()

    # Inline summary right under inputs
    col_sum = st.columns(4)
    col_sum[0].metric("Mean v1", f"{np.mean(v1):.2f}")
    col_sum[1].metric("Range v1", f"{(np.max(v1) - np.min(v1)):.2f}")
    col_sum[2].metric("Mean v2", f"{np.mean(v2):.2f}")
    col_sum[3].metric("Range v2", f"{(np.max(v2) - np.min(v2)):.2f}")

    # -----------------------------
    # Summary box: means and ranges for v1 and v2 (compact layout)
    # -----------------------------
    st.markdown("### Summary: Mean and Range")
    col_sum = st.columns(4)
    col_sum[0].metric("Mean v1", f"{np.mean(v1):.2f}")
    col_sum[1].metric("Range v1", f"{(np.max(v1) - np.min(v1)):.2f}")
    col_sum[2].metric("Mean v2", f"{np.mean(v2):.2f}")
    col_sum[3].metric("Range v2", f"{(np.max(v2) - np.min(v2)):.2f}")

    # -----------------------------
    # Normalization functions
    # -----------------------------
    def range_normalization(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=float)
        if v.size == 0:
            return v
        denom = v.max() - v.min()
        if denom == 0:
            return np.ones_like(v)
        return v / denom


    def divisive_normalization(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=float)
        if v.size == 0:
            return v
        mu = v.mean()
        if mu == 0:
            return np.zeros_like(v)
        return v / mu


    def recurrent_normalization(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=float)
        if v.size == 0:
            return v
        mu = v.mean()
        return v / (v + mu)


    def adaptive_gain(v: np.ndarray, k: float = 0.7) -> np.ndarray:
        v = np.asarray(v, dtype=float)
        if v.size == 0:
            return v
        mu = v.mean()
        return 1.0 / (1.0 + np.exp(-(v - mu) * k))

    # -----------------------------
    # Compute
    # -----------------------------
    v1_rn  = range_normalization(v1)
    v1_dn  = divisive_normalization(v1)
    v1_rdn = recurrent_normalization(v1)
    v1_ag  = adaptive_gain(v1, slope)

    v2_rn  = range_normalization(v2)
    v2_dn  = divisive_normalization(v2)
    v2_rdn = recurrent_normalization(v2)
    v2_ag  = adaptive_gain(v2, slope)

    # -----------------------------
    # Equations
    # -----------------------------
    st.markdown("### Equations")
    colE1, colE2 = st.columns(2)
    with colE1:
        st.latex(r"\text{Range: } f(v)=\frac{v}{\max(v)-\min(v)}")
        st.latex(r"\text{Divisive: } f(v)=\frac{v}{\operatorname{mean}(v)}")
    with colE2:
        st.latex(r"\text{Recurrent divisive: } f(v)=\frac{v}{v+\operatorname{mean}(v)}")
        st.latex(r"\text{Adaptive gain: } f(v)=\frac{1}{1+e^{-(v-\operatorname{mean}(v))\,k}}")

    # -----------------------------
    # Plots (two panels like your Colab)
    # -----------------------------
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    # Colors matching your Colab example
    c_rn = "#F8766D"
    c_dn = "#7CAE00"
    c_rdn = "#00BFC4"
    c_ag = "#C77CFF"

    ax[0].plot(v1, v1_rn,  color=c_rn,  marker='o')
    ax[0].plot(v1, v1_dn,  color=c_dn,  marker='o')
    ax[0].plot(v1, v1_rdn, color=c_rdn, marker='o')
    ax[0].plot(v1, v1_ag,  color=c_ag,  marker='o')
    ax[0].legend(['range normalization','divisive normalization','recurrent divisive norm','adaptive gain/logistic'])
    ax[0].set_xlabel('Value')
    ax[0].set_ylabel('Normalization model output')
    ax[0].set_title('Normalization models (v1)')

    ax[1].plot(v2, v2_rn,  color=c_rn,  marker='o')
    ax[1].plot(v2, v2_dn,  color=c_dn,  marker='o')
    ax[1].plot(v2, v2_rdn, color=c_rdn, marker='o')
    ax[1].plot(v2, v2_ag,  color=c_ag,  marker='o')
    ax[1].tick_params(labelleft=True)
    ax[1].legend(['range normalization','divisive normalization','recurrent divisive norm','adaptive gain/logistic'])
    ax[1].set_xlabel('Value')
    ax[1].set_ylabel('Normalization model output')
    ax[1].set_title('Normalization models (v2)')

    st.pyplot(fig, clear_figure=True)

    # -----------------------------
    # Optional: table
    # -----------------------------
    if show_table:
        st.markdown("### Numeric comparison table")
        import pandas as pd
        df1 = pd.DataFrame({
            'v1': v1,
            'range': v1_rn,
            'divisive': v1_dn,
            'recurrent': v1_rdn,
            'adaptive_gain': v1_ag,
        })
        df2 = pd.DataFrame({
            'v2': v2,
            'range': v2_rn,
            'divisive': v2_dn,
            'recurrent': v2_rdn,
            'adaptive_gain': v2_ag,
        })
        st.dataframe(df1, use_container_width=True)
        st.dataframe(df2, use_container_width=True)

    st.info("Tip: paste different arrays (e.g., low-biased vs high-biased) to see how context shifts each normalization.")

    # ---------------------------------------
    # Expected Utility (Draft Section)
    # ---------------------------------------
    st.markdown("---")
    st.header("Expected Utility (Draft)")

    st.markdown(
        "This section implements Expected Utility with a sign–power utility function: "
        "`return (1 if v >= 0 else -1) * (abs(v) ** alpha)` and identity probability weighting `w(p)=p`."
    )

    # --- Parameters & inputs ---
    colP1, colP2, colP3 = st.columns(3)
    with colP1:
        alpha = st.slider("Utility curvature α", 0.2, 2.0, 0.88, 0.02)
    with colP2:
        v_lottery = st.number_input("Lottery payoff (v_lottery)", value=100000.0, step=1000.0)
        p_lottery = st.number_input("Lottery probability (p_lottery)", value=0.0001, min_value=0.0, max_value=1.0, step=0.0001, format="%.4f")
    with colP3:
        v_gamble1 = st.number_input("Gamble outcome 1 (gain)", value=55.0, step=1.0)
        v_gamble2 = st.number_input("Gamble outcome 2 (loss)", value=-50.0, step=1.0)
        p_gamble = st.number_input("Gamble probability for outcome 1", value=0.5, min_value=0.0, max_value=1.0, step=0.05)

    # --- Core functions ---
    def utility_function(v, alpha=0.88):
        return (1 if v >= 0 else -1) * (abs(v) ** alpha)

    def EU(p, u1, u2=0):
        return p * u1 + (1 - p) * u2

    # --- Show equations used ---
    colEq1, colEq2 = st.columns(2)
    with colEq1:
        st.latex(r"u(v) = (1 \; \text{if } v \ge 0 \; \text{else } -1)\cdot |v|^{\alpha}")
    with colEq2:
        st.latex(r"EU = p\,u(v_1) + (1-p)\,u(v_2)")

    # --- Plot utility function on [0, 50] ---
    v_min, v_max, v_step = 0, 50, 10
    v = np.arange(v_min, v_max + v_step, v_step)
    u = np.array([utility_function(val, alpha) for val in v], dtype=float)

    fig_u, ax_u = plt.subplots()
    ax_u.plot(v, u, marker='o')
    ax_u.plot([v_min, v_max], [v_min, v_max], color='gray', ls=':', label='u(v)=v (linear ref)')
    ax_u.set_xlabel('Value v')
    ax_u.set_ylabel('Utility u(v)')
    ax_u.set_title('Utility function (0–50)')
    ax_u.legend()
    st.pyplot(fig_u, clear_figure=True)

    # --- Subjective spacing & sensitivity ---
    # Compare u(10)-u(0) vs u(50)-u(40)
    uv0, uv10, uv40, uv50 = [utility_function(x, alpha) for x in (0, 10, 40, 50)]
    span_0_10 = uv10 - uv0
    span_40_50 = uv50 - uv40

    # Marginal sensitivity over a finer grid (v>0)
    vv = np.linspace(0.001, 50, 500)
    # derivative du/dv for gains: alpha * v^(alpha-1)
    dudv = alpha * (vv ** (alpha - 1))
    max_idx = int(np.argmax(dudv))
    most_sensitive_at = vv[max_idx]

    colS1, colS2, colS3 = st.columns(3)
    colS1.metric("Δu (0→10)", f"{span_0_10:.3g}")
    colS2.metric("Δu (40→50)", f"{span_40_50:.3g}")
    colS3.metric("Max sensitivity at v≈", f"{most_sensitive_at:.2f}")

    if np.isclose(span_0_10, span_40_50, rtol=1e-6, atol=1e-9):
        st.info("Subjective differences are ~the same between 0–10 and 40–50.")
    else:
        bigger = "0–10" if span_0_10 > span_40_50 else "40–50"
        st.info(f"Subjective difference is larger for {bigger} (with α={alpha:.2f}).")

    # Optional: show marginal utility curve
    fig_mu, ax_mu = plt.subplots()
    ax_mu.plot(vv, dudv)
    ax_mu.set_xlabel('Value v (gains)')
    ax_mu.set_ylabel('Marginal utility du/dv')
    ax_mu.set_title('Where sensitivity is highest (du/dv)')
    st.pyplot(fig_mu, clear_figure=True)

    # --- EU calculations for the two scenarios ---
    # Lottery: p_lottery chance of v_lottery, else 0
    u_lottery = utility_function(v_lottery, alpha)
    eu_lottery = EU(p_lottery, u1=u_lottery, u2=utility_function(0, alpha))

    # Gamble: p_gamble chance of v_gamble1, (1-p) of v_gamble2
    u_g1 = utility_function(v_gamble1, alpha)
    u_g2 = utility_function(v_gamble2, alpha)
    eu_gamble = EU(p_gamble, u1=u_g1, u2=u_g2)

    colEU1, colEU2 = st.columns(2)
    with colEU1:
        st.subheader("Lottery")
        st.write(f"p = {p_lottery:.4f}, payoff = {v_lottery:.0f}")
        st.latex(r"EU_{lottery} = p\,u(v) + (1-p)\,u(0)")
        st.metric("EU (lottery)", f"{eu_lottery:.3g}")
    with colEU2:
        st.subheader("50–50 Gamble")
        st.write(f"p = {p_gamble:.2f}, outcomes = {v_gamble1:.0f} and {v_gamble2:.0f}")
        st.latex(r"EU_{gamble} = p\,u(v_1) + (1-p)\,u(v_2)")
        st.metric("EU (gamble)", f"{eu_gamble:.3g}")

    # --- Should you pursue each according to EU? ---
    # Compare each EU against status-quo u(0)=0
    u_status_quo = utility_function(0, alpha)

    def decision_text(eu_val: float) -> str:
        if eu_val > u_status:
            return "Pursue (EU > status quo)"
        elif eu_val < u_status:
            return "Do NOT pursue (EU < status quo)"
        return "Indifferent (EU ≈ status quo)"

    u_status = u_status_quo
    colD1, colD2 = st.columns(2)
    colD1.success(decision_text(eu_lottery))
    colD2.success(decision_text(eu_gamble))

    # --- What changed and why? ---
    st.markdown("### What changed and why?")
    st.markdown(
        "- When **α < 1** (concave for gains), marginal utility is higher near 0 and falls with v. "
        "This makes large payoffs (like the lottery) contribute less utility than their face value, "
        "so EU can be modest despite a large prize."
        "- The 50–50 gamble combines a moderate gain and a loss. The sign–power utility makes losses count with the same curvature; "
        "their disutility can outweigh the gain depending on α, shifting the EU and the decision.")


if page == "Expected Utility (EU)":
    st.title("Expected Utility (EU)")
    st.markdown(
        "EU allows **nonlinear utility**. We use a sign–power (CRRA-style) function "
        "implemented exactly as: `return (1 if v >= 0 else -1) * (abs(v) ** alpha)`."
    )

    alpha = st.slider("Curvature α", 0.2, 2.0, 0.8, 0.05)

    _show_eq("Expected Utility of lottery L = {(x_i, p_i)}",
             r"EU(L) = \sum_i p_i \cdot u(x_i)")

    # Equations right above their corresponding graphs (side by side)
    col_eq1, col_eq2 = st.columns(2)
    with col_eq1:
        st.latex(r"u(v) = (1 \; \text{if } v \ge 0 \; \text{else } -1)\cdot |v|^{\alpha}")
        xr = np.linspace(-100, 100, 400)
        u_vals = np.array([(1 if val >= 0 else -1) * (abs(val) ** alpha) for val in xr], dtype=float)
        _plot_simple(xr, u_vals, "Outcome v", "Utility u(v)", f"Utility (α={alpha:.2f})")
    with col_eq2:
        st.latex(r"w(p) = p")
        pr = np.linspace(0, 1, 200)
        _plot_simple(pr, pr, "Probability p", "Weight w(p)", "Identity weighting: w(p)=p")

    st.divider()
    st.subheader("Worked examples (EU)")

    def u_func(v, alpha):
        return (1 if v >= 0 else -1) * (abs(v) ** alpha)

    # Example 1: 0.01% to win 100,000; otherwise 0
    p1 = 0.0001
    EU1 = p1 * u_func(100_000.0, alpha) + (1 - p1) * u_func(0.0, alpha)

    # Example 2: 50% +55, 50% -50
    p2 = 0.5
    EU2 = p2 * u_func(55.0, alpha) + (1 - p2) * u_func(-50.0, alpha)

    colA, colB = st.columns(2)
    with colA:
        st.markdown("**Lottery ticket:** 0.01% chance to win 100,000")
        st.latex(r"EU = 0.0001\,u(100{,}000) + 0.9999\,u(0)")
        st.metric("EU (utils)", f"{EU1:.3g}")
    with colB:
        st.markdown("**50–50 gamble:** +55 / −50")
        st.latex(r"EU = 0.5\,u(55) + 0.5\,u(-50)")
        st.metric("EU (utils)", f"{EU2:.3g}")

if page == "Expected Utility (EU)":
    st.title("Expected Utility (EU)")
    st.markdown("EU allows for subjective and **nonlinear utility**, which EV does not consider. The EU ")

    alpha = st.slider("Curvature (α). α<1: concave, α=1: linear, α>1: convex", 0.2, 2.0, 0.8, 0.05)

    _show_eq("Expected Utility of lottery L = {(x_i, p_i)}", r"\mathrm{EU}(L) = \sum_i p_i \cdot u(x_i)")
    _show_eq("Utility (sign–power)", r"u(x) = \operatorname{sign}(x)\,|x|^{\alpha}")
    _show_eq("Probability weighting (identity)", r"w(p) = p")

    # Visuals
    st.divider()
    st.subheader("Graphics of EU utility and probability weighting functions:")
    col1, col2 = _two_cols()
    with col1:
        xr = np.linspace(-100, 100, 400)
        u = np.sign(xr) * (np.abs(xr) ** alpha)
        _plot_simple(xr, u, "Outcome x", "Utility u(x)", f"Sign–power utility (α={alpha:.2f})")

    with col2:
        pr = np.linspace(0, 1, 200)
        _plot_simple(pr, pr, "Probability p", "Weight w(p)", "Identity weighting: w(p)=p")

    st.divider()
    st.subheader("Worked examples (EU)")

    def u(x):
        x = np.asarray(x, dtype=float)
        return np.sign(x) * (np.abs(x) ** alpha)

    # Example 1: 0.01% to win 100,000; otherwise 0
    p1 = 0.0001
    EU1 = p1 * u(100_000.0) + (1 - p1) * u(0.0)

    # Example 2: 50% +55, 50% -50
    p2 = 0.5
    EU2 = p2 * u(55.0) + (1 - p2) * u(-50.0)

    colA, colB = st.columns(2)
    with colA:
        st.markdown("**Lottery ticket:** 0.01% chance to win 100,000")
        st.latex(r"\\mathrm{EU} = w(p)u(100{,}000) + w(1-p)u(0),\\; w(p)=p")
        st.metric("EU (utils)", f"{EU1:.3g}")
    with colB:
        st.markdown("**50–50 gamble:** +55 / −50")
        st.latex(r"\\mathrm{EU} = 0.5\,u(55) + 0.5\,u(-50)")
        st.metric("EU (utils)", f"{EU2:.3g}")

# ---------------------------------------
# Prospect Theory (PT)
# ---------------------------------------
if page == "Prospect Theory (PT)":
    st.title("Prospect Theory (PT)")
    st.markdown("PT uses a **reference-dependent value function** and **nonlinear probability weighting** which EU does not consider. It has distinct domains with different functions for behaviors if they are considered a LOSS or GAIN.")

    st.subheader("Parameters")
    colA, colB = st.columns(2)
    with colA:
        alpha = st.slider("Curvature for gains (α)", 0.2, 1.5, 0.88, 0.02)
        gamma = st.slider("Weighting (gains) γ", 0.2, 1.5, 0.61, 0.01)
        ref = st.slider("Reference point r", -50.0, 50.0, 0.0, 1.0)
    with colB:
        beta = st.slider("Curvature for losses (β)", 0.2, 1.5, 0.88, 0.02)
        delta = st.slider("Weighting (losses) δ", 0.2, 1.5, 0.69, 0.01)
        lam = st.slider("Loss aversion λ", 0.5, 4.0, 2.25, 0.05)

    _show_eq("Value (reference-dependent)", r"v(x) = \begin{cases}(x-r)^{\alpha}, & x \ge r \\ -\lambda\, (r-x)^{\beta}, & x < r\end{cases}")
    _show_eq("Weighting (TK-1992 form)", r"w_+(p) = \frac{p^{\gamma}}{\left(p^{\gamma} + (1-p)^{\gamma}\right)^{1/\gamma}},\quad w_-(p) = \frac{p^{\delta}}{\left(p^{\delta} + (1-p)^{\delta}\right)^{1/\delta}}")

    # Visuals
    col1, col2 = _two_cols()
    with col1:
        x = np.linspace(-100, 100, 500)
        v = np.where(x >= ref, (x - ref) ** alpha, -lam * (ref - x) ** beta)
        _plot_simple(x, v, "Outcome x", "Value v(x)", "Prospect Theory value function")

    with col2:
        p = np.linspace(0.001, 0.999, 400)
        w_plus = p ** gamma / ( (p ** gamma + (1 - p) ** gamma) ** (1/gamma) )
        w_minus = p ** delta / ( (p ** delta + (1 - p) ** delta) ** (1/delta) )
        _plot_multi(p, [w_plus, w_minus], ["w₊(p) (gains)", "w₋(p) (losses)"], "Probability p", "Weight", "Probability weighting (TK-1992)")

    st.divider()
    st.subheader("Worked examples (PT)")

    def v_fn(x):
        x = np.asarray(x, dtype=float)
        return np.where(x >= ref, (x - ref) ** alpha, -lam * (ref - x) ** beta)

    def w_plus_fn(p):
        p = np.asarray(p, dtype=float)
        return p ** gamma / ((p ** gamma + (1 - p) ** gamma) ** (1 / gamma))

    def w_minus_fn(p):
        p = np.asarray(p, dtype=float)
        return p ** delta / ((p ** delta + (1 - p) ** delta) ** (1 / delta))

    # Example 1: 0.01% to win 100,000; else 0
    p1 = 0.0001
    PT1 = w_plus_fn(p1) * v_fn(100_000.0) + w_minus_fn(1 - p1) * v_fn(0.0)

    # Example 2: 50% +55, 50% -50
    p2 = 0.5
    PT2 = w_plus_fn(p2) * v_fn(55.0) + w_minus_fn(1 - p2) * v_fn(-50.0)

    colA, colB = st.columns(2)
    with colA:
        st.markdown("**Lottery ticket:** 0.01% chance to win 100,000")
        st.latex(r"\\mathrm{PT} = w_+(0.0001)\,v(100{,}000) + w_-(0.9999)\,v(0)")
        st.metric("PT value (utils)", f"{PT1:.3g}")
    with colB:
        st.markdown("**50–50 gamble:** +55 / −50")
        st.latex(r"\\mathrm{PT} = w_+(0.5)\,v(55) + w_-(0.5)\,v(-50)")
        st.metric("PT value (utils)", f"{PT2:.3g}")

# ---------------------------------------
# Normalization Techniques
# ---------------------------------------
if page == "Normalization Techniques":
    st.title("Normalization Techniques (Overview)")
    st.caption("Normalization methods are techniques used to adjust or scale values so they’re easier to compare, interpret, or process. Below are 4 types of normalization methods that can be used in the context of making decisions with various influences.")

    st.divider()
    st.subheader("1) Range normalization")
    st.markdown("Scales each value by the total range. Sensitive to extremes; if one value is big, everything else looks small. Linear mapping.")
    st.latex(r"f(v) = \frac{v}{\max(v) - \min(v)}")
    st.caption("Intuition: How big is this value compared to the total spread?")

    st.divider()
    st.subheader("2) Divisive normalization")
    st.markdown("Scales each value by the average. For example, if a distractor value increases, the denominator increases, reducing sensitivity. Linear mapping.")
    st.latex(r"f(v) = \frac{v}{\text{mean}(v)}")
    st.caption("Intuition: How big is this value compared to a typical (average) value?")

    st.divider()
    st.subheader("3) Recurrent divisive normalization")
    st.markdown("Normalizes by the value itself plus the mean. Outputs bound between 0 and 1. Nonlinear: larger values flatten, emphasizing smaller differences among big numbers.")
    st.latex(r"f(v) = \frac{v}{v + \text{mean}(v)}")
    st.caption("Intuition: Relative strength compared to background context — explains context-dependent perception.")

    st.divider()
    st.subheader("4) Adaptive gain / logistic model of value")
    st.markdown("S-shaped sliding sigmoid. Captures contrast around the mean: small shifts near mean are exaggerated, extremes flatten. Below mean → compressed toward 0; above mean → toward 1.")
    st.latex(r"f(v) = \frac{1}{1+e^{-(v-\text{mean}(v)) \cdot k}}")
    st.caption("Intuition: Contrast enhancement — the brain emphasizes differences near the typical value, ignoring extremes.")

# ---------------------------------------
# Normalization Comparisons
# ---------------------------------------
if page == "Normalization Comparisons":
    st.set_page_config(page_title="Normalization Methods Comparison", layout="wide")

    st.title("Normalization Comparison – Restaurant prices ")
    st.caption("Interactive version of the Google Colab that compares the different normalization methods! Situation: imagine you're choosing  between restaurants with different prices. You can compare what happens when the restaurant group has a larger range, when the average prices tend to be lower vs. higher, and how that plays out with each normalization method.")

    # -----------------------------
    # Helper: parse arrays from text
    # -----------------------------
    def parse_array(s: str) -> np.ndarray:
        toks = [t for t in s.replace(",", " ").split() if t]
        try:
            return np.array([float(t) for t in toks], dtype=float)
        except Exception:
            return np.array([], dtype=float)

    # -----------------------------
    # On-page inputs
    # -----------------------------
    st.header("Inputs")

    def_v1 = "1 2 5 10"
    def_v2 = "1 5 9 10"

    st.info("Tip: paste different arrays (e.g., low-biased vs high-biased) to see how context shifts each normalization.")

    col_in1, col_in2 = st.columns(2)
    with col_in1:
        v1_str = st.text_input("Restaurant Group 1 (comma/space separated)", value=def_v1)
    with col_in2:
        v2_str = st.text_input("Restaurant Group 1 (comma/space separated)", value=def_v2)

    col_in3, col_in4 = st.columns([1,1])
    with col_in3:
        slope = st.slider("Adaptive gain slope k", 0.05, 2.0, 0.7, 0.05)
    with col_in4:
        show_table = st.checkbox("Show numeric table", value=True)

    v1 = parse_array(v1_str)
    v2 = parse_array(v2_str)

    # Guardrail
    if v1.size == 0 or v2.size == 0:
        st.error("Please provide valid numeric arrays for Restaurant Group 1 and Restaurant Group 2.")
        st.stop()
        
    # Inline summary right under inputs
    col_sum = st.columns(4)
    col_sum[0].metric("Mean G1", f"{np.mean(v1):.2f}")
    col_sum[1].metric("Range G1", f"{(np.max(v1) - np.min(v1)):.2f}")
    col_sum[2].metric("Mean G2", f"{np.mean(v2):.2f}")
    col_sum[3].metric("Range G2", f"{(np.max(v2) - np.min(v2)):.2f}")

    # -----------------------------
    # Normalization functions
    # -----------------------------
    def range_normalization(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=float)
        if v.size == 0:
            return v
        denom = v.max() - v.min()
        if denom == 0:
            return np.ones_like(v)
        return v / denom


    def divisive_normalization(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=float)
        if v.size == 0:
            return v
        mu = v.mean()
        if mu == 0:
            return np.zeros_like(v)
        return v / mu


    def recurrent_normalization(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=float)
        if v.size == 0:
            return v
        mu = v.mean()
        return v / (v + mu)


    def adaptive_gain(v: np.ndarray, k: float = 0.7) -> np.ndarray:
        v = np.asarray(v, dtype=float)
        if v.size == 0:
            return v
        mu = v.mean()
        return 1.0 / (1.0 + np.exp(-(v - mu) * k))

    # -----------------------------
    # Compute
    # -----------------------------
    v1_rn  = range_normalization(v1)
    v1_dn  = divisive_normalization(v1)
    v1_rdn = recurrent_normalization(v1)
    v1_ag  = adaptive_gain(v1, slope)

    v2_rn  = range_normalization(v2)
    v2_dn  = divisive_normalization(v2)
    v2_rdn = recurrent_normalization(v2)
    v2_ag  = adaptive_gain(v2, slope)

    # -----------------------------
    # Plots (two panels like your Colab)
    # -----------------------------
    st.markdown("### Plots")
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    # Colors matching your Colab example
    c_rn = "#F8766D"
    c_dn = "#7CAE00"
    c_rdn = "#00BFC4"
    c_ag = "#C77CFF"

    ax[0].plot(v1, v1_rn,  color=c_rn,  marker='o')
    ax[0].plot(v1, v1_dn,  color=c_dn,  marker='o')
    ax[0].plot(v1, v1_rdn, color=c_rdn, marker='o')
    ax[0].plot(v1, v1_ag,  color=c_ag,  marker='o')
    ax[0].legend(['range normalization','divisive normalization','recurrent divisive norm','adaptive gain/logistic'])
    ax[0].set_xlabel('Value')
    ax[0].set_ylabel('Normalization model output')
    ax[0].set_title('Normalization models (Restaurant Group 1)')

    ax[1].plot(v2, v2_rn,  color=c_rn,  marker='o')
    ax[1].plot(v2, v2_dn,  color=c_dn,  marker='o')
    ax[1].plot(v2, v2_rdn, color=c_rdn, marker='o')
    ax[1].plot(v2, v2_ag,  color=c_ag,  marker='o')
    ax[1].tick_params(labelleft=True)
    ax[1].legend(['range normalization','divisive normalization','recurrent divisive norm','adaptive gain/logistic'])
    ax[1].set_xlabel('Value')
    ax[1].set_ylabel('Normalization model output')
    ax[1].set_title('Normalization models (Restaurant Group 2)')

    st.pyplot(fig, clear_figure=True)

    # -----------------------------
    # Optional: table
    # -----------------------------
    if show_table:
        st.markdown("### Numeric comparison table")
        import pandas as pd
        df1 = pd.DataFrame({
            'v1': v1,
            'range': v1_rn,
            'divisive': v1_dn,
            'recurrent': v1_rdn,
            'adaptive_gain': v1_ag,
        })
        df2 = pd.DataFrame({
            'v2': v2,
            'range': v2_rn,
            'divisive': v2_dn,
            'recurrent': v2_rdn,
            'adaptive_gain': v2_ag,
        })
        st.dataframe(df1, use_container_width=True)
        st.dataframe(df2, use_container_width=True)

     # -----------------------------
    # Equations
    # -----------------------------
    st.markdown("### Equations")
    colE1, colE2 = st.columns(2)
    with colE1:
        st.latex(r"\text{Range: } f(v)=\frac{v}{\max(v)-\min(v)}")
        st.latex(r"\text{Divisive: } f(v)=\frac{v}{\operatorname{mean}(v)}")
    with colE2:
        st.latex(r"\text{Recurrent divisive: } f(v)=\frac{v}{v+\operatorname{mean}(v)}")
        st.latex(r"\text{Adaptive gain: } f(v)=\frac{1}{1+e^{-(v-\operatorname{mean}(v))\,k}}")