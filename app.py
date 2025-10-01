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

        **Normalization techniques** (applied to choosing between restaurants by price — *lower is better*):
        - **Range normalization**
        - **Divisive normalization**
        - **Recurrent divisive normalization**
        - **Adaptive gain / logistic value**
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
        st.markdown("**Lottery ticket:** 0.01% chance to win 100,000; otherwise 0")
        st.latex(r"\\mathrm{EV} = 0.0001 \times 100{,}000 + 0.9999 \times 0 = 10")
        st.metric("EV", f"{ev1:.2f}")
    with colB:
        st.markdown("**50–50 gamble:** +55 with 50%, −50 with 50%")
        st.latex(r"\\mathrm{EV} = 0.5 \times 55 + 0.5 \times (-50) = 2.5")
        st.metric("EV", f"{ev2:.2f}")

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
    st.title("Expected Utility (EU)")
    st.markdown("EU allows **nonlinear utility**. Below we use a **sign–power/CRRA-style** shape.")

    alpha = st.slider("Curvature (α). α<1: concave, α=1: linear, α>1: convex", 0.2, 2.0, 0.8, 0.05)

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
    st.title("Normalization Techniques (w/ example of Restaurant Prices)")
    st.markdown(
        "We transform **prices** into **desirability** scores where higher is better. Because lower prices are preferred, we invert prices via a context-dependent step: ")
    _show_eq("Price → desirability signal", r"s_i = \\max_j p_j - p_i")

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

    idx = np.arange(1, n + 1)

    st.divider()
    st.subheader("1) Range normalization")
    st.markdown("Scales each value by the total range. Sensitive to extremes; if one value is big, everything else looks small. Linear mapping.")
    st.latex(r"f(v) = \frac{v}{\max(v) - \min(v)}")
    st.caption("Intuition: How big is this value compared to the total spread?")

    def range_norm(prices):
        p = np.array(prices, dtype=float)
        denom = p.max() - p.min()
        if denom == 0:
            return np.ones_like(p)
        return (p.max() - p) / denom

    yA = range_norm(prices_low)
    yB = range_norm(prices_high)

    col1, col2 = _two_cols()
    with col1:
        _plot_simple(idx, yA, "Restaurant index", "Normalized value", "Range norm – Context A")
    with col2:
        _plot_simple(idx, yB, "Restaurant index", "Normalized value", "Range norm – Context B")

    st.divider()
    st.subheader("2) Divisive normalization")
    st.markdown("Scales each value by the average. For example, if a distractor value increases, the denominator increases, reducing sensitivity. Linear mapping.")
    st.latex(r"f(v) = \\frac{v}{\\text{mean}(v)}")
    st.caption("Intuition: How big is this value compared to a typical (average) value?")

    sigma = st.slider("Stabilizer σ (divisive)", 0.0, 10.0, 1.0, 0.1)

    def divisive_norm(prices, sigma):
        s = desirability_from_price(prices)
        denom = sigma + s.mean() * len(s)
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
    st.markdown("Normalizes by the value itself plus the mean. Outputs bound between 0 and 1. Nonlinear: larger values flatten, emphasizing smaller differences among big numbers.")
    st.latex(r"f(v) = \\frac{v}{v + \\text{mean}(v)}")
    st.caption("Intuition: Relative strength compared to background context — explains context-dependent perception.")

    def recurrent_divisive_norm(prices):
        v = desirability_from_price(prices)
        mean_v = v.mean()
        return v / (v + mean_v)

    yA = recurrent_divisive_norm(prices_low)
    yB = recurrent_divisive_norm(prices_high)

    col1, col2 = _two_cols()
    with col1:
        _plot_simple(idx, yA, "Restaurant index", "Normalized value", "Recurrent divisive norm – Context A")
    with col2:
        _plot_simple(idx, yB, "Restaurant index", "Normalized value", "Recurrent divisive norm – Context B")

    st.divider()
    st.subheader("4) Adaptive gain / logistic model of value")
    st.markdown("S-shaped sliding sigmoid. Captures contrast around the mean: small shifts near mean are exaggerated, extremes flatten. Below mean → compressed toward 0; above mean → toward 1.")
    st.latex(r"f(v) = \\frac{1}{1+e^{-(v-\\text{mean}(v)) \\cdot k}}")
    st.caption("Intuition: Contrast enhancement — the brain emphasizes differences near the typical value, ignoring extremes.")

    k = st.slider("Slope k", 0.01, 2.0, 0.3, 0.01)

    def logistic_value(prices, k):
        v = desirability_from_price(prices)
        r = np.mean(v)
        return 1.0 / (1.0 + np.exp(-(v - r) * k))

    yA = logistic_value(prices_low, k)
    yB = logistic_value(prices_high, k)

    col1, col2 = _two_cols()
    with col1:
        _plot_simple(idx, yA, "Restaurant index", "Value", "Adaptive gain (logistic) – Context A")
    with col2:
        _plot_simple(idx, yB, "Restaurant index", "Value", "Adaptive gain (logistic) – Context B")

    st.caption("All normalization outputs above are on an arbitrary scale where higher is better.")
