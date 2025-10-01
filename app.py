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


def desirability_from_price(prices):
    # Convert prices (lower=better) to a nonnegative desirability signal s_i
    # s_i = max(p) - p_i, then rescaled to [0, 1] if needed downstream
    prices = np.array(prices, dtype=float)
    s = prices.max() - prices
    return s


# ---------------------------------------
# Overview
# ---------------------------------------
if page == "Overview":
    st.title("Choice Models under Uncertainty & Normalization Techniques")
    st.markdown(
        """
        This interactive app covers three classic decision models under risk/uncertainty and four 
        normalization schemes used in value coding.

        **Models**
        - **Expected Value (EV):** linear utility, linear probability.
        - **Expected Utility (EU):** nonlinear utility over outcomes.
        - **Prospect Theory (PT):** reference-dependent value and nonlinear probability weighting.

        **Normalization techniques** (applied to an example of choosing between restaurants by price — *lower is better*):
        - **Range normalization**
        - **Divisive normalization**
        - **Recurrent divisive normalization**
        - **Adaptive gain / logistic value**
        """
    )
    st.info("Use the sidebar to visit the model- or normalization-specific pages. Sliders let you change parameters and immediately see the equations and curves update.")

# ---------------------------------------
# Expected Value (EV)
# ---------------------------------------
if page == "Expected Value (EV)":
    st.title("Expected Value (EV)")
    st.markdown(
        "EV assumes **linear utility** and **linear probability weighting**. It simply multiplies outcomes by their probabilities and sums.")

    _show_eq("EV of a lottery L = {(x_i, p_i)}", r"\\mathrm{EV}(L) = \\sum_i p_i \\cdot x_i")
    _show_eq("Utility (linear)", r"u(x) = x")
    _show_eq("Probability weighting (identity)", r"w(p) = p")

    # Visuals
    col1, col2 = _two_cols()
    with col1:
        xr = np.linspace(-100, 100, 400)
        _plot_simple(xr, xr, "Outcome x", "Utility u(x)", "Linear utility: u(x)=x")

    with col2:
        pr = np.linspace(0, 1, 200)
        _plot_simple(pr, pr, "Probability p", "Weight w(p)", "Identity weighting: w(p)=p")

# ---------------------------------------
# Expected Utility (EU)
# ---------------------------------------
if page == "Expected Utility (EU)":
    st.title("Expected Utility (EU)")
    st.markdown("EU allows **nonlinear utility**. Below we use a **power/CRRA-style** shape with optional loss domain via a sign-power form.")

    alpha = st.slider("Curvature (α). α<1: concave (risk-averse), α=1: linear, α>1: convex (risk-seeking)", 0.2, 2.0, 0.8, 0.05)

    _show_eq("Expected Utility of lottery L = {(x_i, p_i)}", r"\\mathrm{EU}(L) = \\sum_i p_i \\cdot u(x_i)")
    _show_eq("Utility (sign–power)", r"u(x) = \\operatorname{sign}(x)\\,|x|^{\\alpha}")
    _show_eq("Probability weighting (identity)", r"w(p) = p")

    # Visuals
    col1, col2 = _two_cols()
    with col1:
        xr = np.linspace(-100, 100, 400)
        u = np.sign(xr) * (np.abs(xr) ** alpha)
        _plot_simple(xr, u, "Outcome x", "Utility u(x)", f"Sign–power utility (α={alpha:.2f})")

    with col2:
        pr = np.linspace(0, 1, 200)
        _plot_simple(pr, pr, "Probability p", "Weight w(p)", "Identity weighting: w(p)=p")

# ---------------------------------------
# Prospect Theory (PT)
# ---------------------------------------
if page == "Prospect Theory (PT)":
    st.title("Prospect Theory (PT)")
    st.markdown("PT uses a **reference-dependent value function** and **nonlinear probability weighting**.")

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

# ---------------------------------------
# Normalization Techniques
# ---------------------------------------
if page == "Normalization Techniques":
    st.title("Normalization Techniques (Restaurant Prices)")
    st.markdown(
        "We transform **prices** into **desirability** scores where higher is better. Because lower prices are preferred, we invert prices via a context-dependent step: ")
    _show_eq("Price → desirability signal", r"s_i = \max_j p_j - p_i")

    st.subheader("Example contexts")
    n = st.slider("Number of restaurants", 3, 15, 5)

    # Two contexts: biased low vs biased high
    colL, colH = st.columns(2)
    with colL:
        st.markdown("**Context A (biased low prices)**")
        low_center = st.slider("A: average price (~)", 5, 20, 10, 1)
        rng_low = np.random.default_rng(1)
        prices_low = np.clip(np.round(rng_low.normal(loc=low_center, scale=2.0, size=n), 2), 1.0, None)
        st.write({f"R{i+1}": float(p) for i, p in enumerate(prices_low)})
    with colH:
        st.markdown("**Context B (biased high prices)**")
        high_center = st.slider("B: average price (~)", 20, 50, 30, 1)
        rng_high = np.random.default_rng(2)
        prices_high = np.clip(np.round(rng_high.normal(loc=high_center, scale=3.0, size=n), 2), 1.0, None)
        st.write({f"R{i+1}": float(p) for i, p in enumerate(prices_high)})

    st.divider()
    st.subheader("1) Range normalization")
    _show_eq("Equation (lower price = better)", r"\tilde{x}_i = \dfrac{\max(p) - p_i}{\max(p) - \min(p)}")

    def range_norm(prices):
        p = np.array(prices, dtype=float)
        denom = p.max() - p.min()
        if denom == 0:
            return np.ones_like(p)
        return (p.max() - p) / denom

    yA = range_norm(prices_low)
    yB = range_norm(prices_high)

    idx = np.arange(1, n + 1)
    col1, col2 = _two_cols()
    with col1:
        _plot_simple(idx, yA, "Restaurant index", "Normalized value", "Range norm – Context A (low prices)")
    with col2:
        _plot_simple(idx, yB, "Restaurant index", "Normalized value", "Range norm – Context B (high prices)")

    st.divider()
    st.subheader("2) Divisive normalization")
    _show_eq("Equation (on desirability s)", r"\tilde{x}_i = \dfrac{s_i}{\sigma + \sum_j s_j},\qquad s_i = \max(p) - p_i")

    sigma = st.slider("Stabilizer σ (divisive)", 0.0, 10.0, 1.0, 0.1)

    def divisive_norm(prices, sigma):
        s = desirability_from_price(prices)
        denom = sigma + s.sum()
        if denom == 0:
            return np.zeros_like(s)
        return s / denom

    yA = divisive_norm(prices_low, sigma)
    yB = divisive_norm(prices_high, sigma)

    col1, col2 = _two_cols()
    with col1:
        _plot_simple(idx, yA, "Restaurant index", "Normalized value", "Divisive norm – Context A")
    with col2:
        _plot_simple(idx, yB, "Restaurant index", "Normalized value", "Divisive norm – Context B")

    st.divider()
    st.subheader("3) Recurrent divisive normalization")
    _show_eq("Fixed point", r"y_i = \dfrac{s_i}{\sigma + \sum_j w_{ij} y_j}\quad\Rightarrow\quad y^{(t+1)}_i = \dfrac{s_i}{\sigma + \sum_j w_{ij} y^{(t)}_j}")

    sigma_rec = st.slider("Stabilizer σ (recurrent)", 0.0, 10.0, 1.0, 0.1)
    w_off = st.slider("Off-diagonal weight w (0=no interaction, 1=strong)", 0.0, 1.0, 0.5, 0.05)
    max_iter = st.slider("Max iterations", 5, 200, 50, 5)
    tol = st.slider("Convergence tol", 1e-6, 1e-2, 1e-4, format="%e")

    def recurrent_divisive_norm(prices, sigma, w_off, max_iter, tol):
        s = desirability_from_price(prices)
        n = s.size
        # Weight matrix: zeros on diagonal, w_off elsewhere
        W = np.full((n, n), w_off)
        np.fill_diagonal(W, 0.0)
        y = np.zeros_like(s)
        for _ in range(max_iter):
            denom = sigma + W @ y
            denom = np.where(denom == 0, np.finfo(float).eps, denom)
            y_new = s / denom
            if np.max(np.abs(y_new - y)) < tol:
                y = y_new
                break
            y = y_new
        return y

    yA = recurrent_divisive_norm(prices_low, sigma_rec, w_off, max_iter, tol)
    yB = recurrent_divisive_norm(prices_high, sigma_rec, w_off, max_iter, tol)

    col1, col2 = _two_cols()
    with col1:
        _plot_simple(idx, yA, "Restaurant index", "Normalized value", "Recurrent divisive norm – Context A")
    with col2:
        _plot_simple(idx, yB, "Restaurant index", "Normalized value", "Recurrent divisive norm – Context B")

    st.divider()
    st.subheader("4) Adaptive gain / logistic value")
    _show_eq("Logistic transform (lower price = higher value)", r"\tilde{x}_i = \frac{1}{1 + \exp\big(k\,[p_i - r]\big)}")

    k = st.slider("Slope k", 0.01, 2.0, 0.3, 0.01)
    use_median = st.checkbox("Reference r = median price in context", value=True)

    def logistic_value(prices, k, r=None):
        p = np.array(prices, dtype=float)
        if r is None:
            r = np.median(p)
        return 1.0 / (1.0 + np.exp(k * (p - r)))

    rA = np.median(prices_low) if use_median else st.number_input("Manual r for A", value=float(np.median(prices_low)))
    rB = np.median(prices_high) if use_median else st.number_input("Manual r for B", value=float(np.median(prices_high)))

    yA = logistic_value(prices_low, k, rA)
    yB = logistic_value(prices_high, k, rB)

    col1, col2 = _two_cols()
    with col1:
        _plot_simple(idx, yA, "Restaurant index", "Value", f"Adaptive gain (logistic) – Context A (r={rA:.2f})")
    with col2:
        _plot_simple(idx, yB, "Restaurant index", "Value", f"Adaptive gain (logistic) – Context B (r={rB:.2f})")

    st.caption("All normalization outputs above are on an arbitrary scale where **higher is better**.")
