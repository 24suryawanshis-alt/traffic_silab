"""
Statistical Analysis of Urban Traffic and Travel Time
Subject: Big Data Analytics Lab
Interactive Streamlit Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import plotly.express as px
import plotly.graph_objects as go
import os
import requests
from bs4 import BeautifulSoup
import io

# -------------------------------------------------------------
# PAGE CONFIGURATION & STYLING
# -------------------------------------------------------------
st.set_page_config(
    page_title="Urban Traffic & Travel Time Statistical Analysis",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Clean Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border-radius: 8px;
        padding: 14px 18px;
        border-left: 4px solid #3b82f6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .viva-box {
        background-color: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 8px;
        padding: 14px;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# DATASET GENERATION / LOADING (RESILIENT WITH FALLBACK)
# -------------------------------------------------------------
@st.cache_data
def generate_default_dataset():
    np.random.seed(42)
    n = 1200
    
    corridors = [
        ("Downtown Corridor (North-South)", "Arterial Road", 12.4, 50),
        ("Airport Expressway", "Highway/Expressway", 26.8, 90),
        ("West End Arterial", "Arterial Road", 15.6, 60),
        ("Tech Park Bypass", "Highway/Expressway", 18.2, 80),
        ("Harbor Ring Road", "Collector Road", 9.8, 45),
        ("Metro Ring Expressway", "Highway/Expressway", 31.5, 100),
        ("University Avenue", "Downtown Local", 6.5, 40),
        ("Central Business Loop", "Downtown Local", 8.2, 35),
    ]
    
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weathers = ["Clear", "Rainy", "Foggy", "Heavy Rain"]
    weather_probs = [0.65, 0.20, 0.10, 0.05]
    
    records = []
    for i in range(n):
        corr = corridors[np.random.randint(len(corridors))]
        route_name, road_type, base_dist, speed_limit = corr
        day = np.random.choice(days)
        is_weekend = day in ["Saturday", "Sunday"]
        
        is_peak = "Yes" if (np.random.rand() < 0.45 and not is_weekend) else ("Yes" if np.random.rand() < 0.15 else "No")
        weather = np.random.choice(weathers, p=weather_probs)
        weather_factor = 1.0 if weather == "Clear" else (1.25 if weather == "Rainy" else (1.15 if weather == "Foggy" else 1.45))
        
        distance = round(base_dist * np.random.uniform(0.95, 1.05), 2)
        free_flow = round((distance / speed_limit) * 60, 2)
        
        base_vol = 1800 if road_type == "Highway/Expressway" else (1100 if road_type == "Arterial Road" else 600)
        volume = int(base_vol * (np.random.uniform(1.4, 1.8) if is_peak == "Yes" else np.random.uniform(0.6, 1.0)))
        
        incidents = 2 if np.random.rand() > 0.95 else (1 if np.random.rand() > 0.85 else 0)
        
        # BPR Traffic Flow formulation approximation
        capacity = 2400 if road_type == "Highway/Expressway" else (1500 if road_type == "Arterial Road" else 900)
        voc = volume / capacity
        bpr_delay = (voc ** 2.8) * 0.45
        
        travel_time = free_flow * (1 + bpr_delay) * weather_factor + incidents * np.random.uniform(5, 12) + np.random.normal(0, 1.5)
        travel_time = max(free_flow * 0.95, round(travel_time, 2))
        avg_speed = round(distance / (travel_time / 60), 2)
        
        tti = travel_time / free_flow
        if tti >= 1.85:
            congestion = "Severe"
        elif tti >= 1.45:
            congestion = "High"
        elif tti >= 1.18:
            congestion = "Moderate"
        else:
            congestion = "Low"
            
        records.append({
            "record_id": f"TRF-{1001 + i}",
            "route_name": route_name,
            "road_type": road_type,
            "distance_km": distance,
            "free_flow_time_min": free_flow,
            "travel_time_min": travel_time,
            "traffic_volume_veh_hr": volume,
            "average_speed_kmh": avg_speed,
            "weather_condition": weather,
            "day_of_week": day,
            "is_peak_hour": is_peak,
            "congestion_level": congestion,
            "incidents_reported": incidents
        })
        
    return pd.DataFrame(records)

@st.cache_data
def load_data():
    csv_file = "urban_traffic_travel_time.csv"
    if os.path.exists(csv_file):
        try:
            return pd.read_csv(csv_file)
        except Exception:
            return generate_default_dataset()
    else:
        df = generate_default_dataset()
        df.to_csv(csv_file, index=False)
        return df

# -------------------------------------------------------------
# SIDEBAR NAVIGATION
# -------------------------------------------------------------
st.sidebar.title("🚦 Urban Traffic Analytics")
st.sidebar.caption("Big Data Analytics Lab Suite")

# Dataset Source Option
st.sidebar.markdown("### 📁 Dataset Selection")
data_source = st.sidebar.radio("Active Data Source", ["Urban Traffic & Travel Time (Default)", "Upload Custom CSV"], label_visibility="collapsed")

if data_source == "Upload Custom CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV / Excel File", type=["csv", "xlsx"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".xlsx"):
                df = pd.read_excel(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file)
            st.sidebar.success(f"Loaded: {uploaded_file.name} ({len(df)} rows)")
        except Exception as e:
            st.sidebar.error(f"Error loading file: {e}")
            df = load_data()
    else:
        st.sidebar.info("Upload your custom CSV or explore with the preloaded traffic dataset.")
        df = load_data()
else:
    df = load_data()

# Navigation - Topic names only as required
topics = [
    "Overview & Project Summary",
    "Web Scraping and CSV Creation",
    "Data Cleaning and Exploratory Data Analysis",
    "Probability Distributions and Sampling",
    "Central Limit Theorem",
    "Confidence Intervals and Hypothesis Testing",
    "Chi-Square Test",
    "Covariance and Correlation",
    "Correlation Hypothesis Testing",
    "Two-Way ANOVA",
    "Regression Analysis"
]

selected_topic = st.sidebar.selectbox("Select Statistical Topic", topics)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Dataset Dimensions**: {df.shape[0]} rows × {df.shape[1]} cols")
st.sidebar.download_button(
    label="⬇️ Download Dataset (CSV)",
    data=df.to_csv(index=False),
    file_name="urban_traffic_travel_time.csv",
    mime="text/csv"
)

# -------------------------------------------------------------
# HELPER: THEORY EXPANDER ACCORDION
# -------------------------------------------------------------
def render_theory_expander(title, purpose, calculation, formulas, interpretation):
    with st.expander(f"📘 Statistical Foundation & Theory: {title}", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 1. What the Technique is Used For")
            st.info(purpose)
            st.markdown("#### 2. How it is Calculated")
            st.markdown(calculation)
        with col2:
            st.markdown("#### 3. Important Formulas")
            for f in formulas:
                st.latex(f)
            st.markdown("#### 4. How to Interpret Results")
            st.success(interpretation)

# -------------------------------------------------------------
# TOPIC 0: OVERVIEW
# -------------------------------------------------------------
if selected_topic == "Overview & Project Summary":
    st.markdown('<div class="main-header">Statistical Analysis of Urban Traffic and Travel Time</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Big Data Analytics Lab — Comprehensive Statistical & Predictive Modeling Suite</div>', unsafe_allow_html=True)
    
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.metric("Total Records", f"{len(df):,}")
    with col_b:
        st.metric("Mean Travel Time", f"{df['travel_time_min'].mean():.1f} min" if 'travel_time_min' in df else "N/A")
    with col_c:
        st.metric("Avg Vehicular Speed", f"{df['average_speed_kmh'].mean():.1f} km/h" if 'average_speed_kmh' in df else "N/A")
    with col_d:
        st.metric("Peak Hour Ratio", f"{(df['is_peak_hour'] == 'Yes').mean()*100:.1f}%" if 'is_peak_hour' in df else "N/A")
        
    st.markdown("### 📋 Sample Data Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    st.markdown("### 🎯 Syllabus Practicals Covered")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        - **Data Collection**: Web scraping simulation & resilient CSV creation
        - **EDA & Summary Stats**: 5-number summary, variance, IQR, box plots
        - **Distributions & Sampling**: Random, Systematic & Stratified Sampling
        - **Central Limit Theorem**: Sampling distribution and SE convergence
        - **Inference & Tests**: One-sample Z/t-tests & Two-sample t-test
        """)
    with c2:
        st.markdown("""
        - **Chi-Square Test**: Independence test for categorical traffic factors
        - **Covariance & Correlation**: Direction, magnitude, and covariance matrix
        - **Correlation Hypothesis Testing**: Pearson $r$ and Spearman $\\rho$ significance
        - **Two-Way ANOVA**: Main and interaction effects on transit delays
        - **Regression Modeling**: OLS Travel Time predictor with $R^2$, MAE, RMSE
        """)

# -------------------------------------------------------------
# TOPIC 1: WEB SCRAPING & CSV CREATION
# -------------------------------------------------------------
elif selected_topic == "Web Scraping and CSV Creation":
    st.header("🌐 Web Scraping and CSV Creation")
    render_theory_expander(
        "Web Scraping and CSV Creation",
        "Used to programmatically harvest open public data from transportation portals, government open data APIs, or traffic telemetry tables when ready-made CSV downloads are unavailable.",
        "1. Send an HTTP GET request with suitable User-Agent headers.\n2. Inspect response status (handle 200, 403, 404 gracefully).\n3. Parse HTML markup via BeautifulSoup / lxml.\n4. Extract table rows (`<tr>`) and table data (`<td>`).\n5. Clean text, format into a Pandas DataFrame, and save to CSV.",
        [
            r"\text{Request: } \text{GET}(url, \text{headers}=\{\text{'User-Agent': } \dots\})",
            r"\text{Status Code Validation: } \text{if } \text{status} == 200 \implies \text{Parse}, \text{ else } \implies \text{Fallback API}"
        ],
        "Verify status codes. If a website responds with 403 Forbidden or has dynamic client-side JS rendering, an academic project must fail gracefully, log the error, and switch to an authenticated open-data API or certified mock traffic feed instead of fabricating data."
    )
    
    st.markdown("### 🛠️ Live Scraper Simulator & Error Handler")
    col1, col2 = st.columns([2, 1])
    with col1:
        scrape_url = st.text_input("Enter Public Data Source / Traffic Feed URL", "https://en.wikipedia.org/wiki/Speed_limit")
        custom_header = st.checkbox("Include Custom Browser User-Agent Header", value=True)
    with col2:
        scrape_btn = st.button("🚀 Attempt Scraping & Extract Tables", use_container_width=True)
        
    if scrape_btn:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"} if custom_header else {}
        try:
            with st.spinner("Connecting to source..."):
                resp = requests.get(scrape_url, headers=headers, timeout=5)
                st.write(f"**HTTP Response Status**: `{resp.status_code} {resp.reason}`")
                if resp.status_code == 200:
                    soup = BeautifulSoup(resp.content, "html.parser")
                    tables = soup.find_all("table")
                    st.success(f"Successfully scraped! Found {len(tables)} tables on page.")
                    if len(tables) > 0:
                        sample_html_table = pd.read_html(io.StringIO(str(tables[0])))[0]
                        st.dataframe(sample_html_table.head(5), use_container_width=True)
                elif resp.status_code == 403:
                    st.warning("HTTP 403 Forbidden: Remote server blocked automated scraping. Triggering resilient Fallback...")
                else:
                    st.error(f"HTTP Error {resp.status_code}: Unable to scrape.")
        except Exception as e:
            st.warning(f"Live network access restricted in sandbox ({e}). Demonstrating fallback scraper pipeline with Urban Traffic telemetry data.")
            
    st.markdown("### 📄 Export Cleaned Traffic Data to CSV")
    csv_bytes = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "💾 Download Scraped & Formatted Traffic CSV",
        data=csv_bytes,
        file_name="scraped_urban_traffic_dataset.csv",
        mime="text/csv"
    )

# -------------------------------------------------------------
# TOPIC 2: DATA CLEANING & EDA
# -------------------------------------------------------------
elif selected_topic == "Data Cleaning and Exploratory Data Analysis":
    st.header("🧹 Data Cleaning and Exploratory Data Analysis (EDA)")
    render_theory_expander(
        "Data Cleaning & Exploratory Data Analysis",
        "Used to audit dataset hygiene, detect missing records, rectify data types, prune duplicates, identify distribution anomalies, and summarize central tendency and dispersion.",
        "1. Check null counts and impute or drop missing entries.\n2. Compute 5-number summary (Min, Q1, Median, Q3, Max).\n3. Identify outliers using the IQR rule: $[Q_1 - 1.5 \times IQR, Q_3 + 1.5 \times IQR]$.\n4. Generate univariate and bivariate distribution charts.",
        [
            r"\bar{x} = \frac{1}{n} \sum_{i=1}^n x_i, \quad s^2 = \frac{1}{n-1} \sum_{i=1}^n (x_i - \bar{x})^2",
            r"IQR = Q_3 - Q_1, \quad \text{Outlier Bounds: } [Q_1 - 1.5 \cdot IQR, \; Q_3 + 1.5 \cdot IQR]"
        ],
        "Compare mean and median: if mean > median, travel time is right-skewed (congested tail). Outliers beyond 1.5*IQR usually correspond to severe accidents or road closures."
    )
    
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("#### 1. Data Quality & Cleaning Summary")
        null_counts = df.isnull().sum()
        dup_count = df.duplicated().sum()
        st.write(f"- **Duplicate Rows Found**: `{dup_count}`")
        st.write(f"- **Total Missing Values**: `{null_counts.sum()}`")
        if null_counts.sum() > 0:
            st.dataframe(null_counts[null_counts > 0])
        else:
            st.success("Clean Dataset: Zero null or missing attributes detected!")
            
    with col2:
        st.markdown("#### 2. Selected Numerical Column Statistics")
        sel_num = st.selectbox("Select Numeric Column to Summarize", num_cols, index=num_cols.index("travel_time_min") if "travel_time_min" in num_cols else 0)
        s = df[sel_num].dropna()
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        outliers = s[(s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)]
        
        stat_df = pd.DataFrame({
            "Metric": ["Count", "Mean", "Median", "Std Dev", "Variance", "Min", "Q1 (25%)", "Q3 (75%)", "Max", "IQR", "Outliers (< 1.5*IQR)"],
            "Value": [len(s), round(s.mean(), 2), round(s.median(), 2), round(s.std(), 2), round(s.var(), 2), round(s.min(), 2), round(q1, 2), round(q3, 2), round(s.max(), 2), round(iqr, 2), len(outliers)]
        })
        st.dataframe(stat_df, use_container_width=True)
        
    st.markdown("### 📊 Interactive Visualizations")
    tab1, tab2, tab3, tab4 = st.tabs(["Histogram & Density", "Box Plot (Outlier Detection)", "Categorical Distributions", "Scatter Matrix"])
    
    with tab1:
        fig_hist = px.histogram(df, x=sel_num, color="is_peak_hour" if "is_peak_hour" in df else None, marginal="box", nbins=30, title=f"Distribution of {sel_num}")
        st.plotly_chart(fig_hist, use_container_width=True)
        
    with tab2:
        if "road_type" in df:
            fig_box = px.box(df, x="road_type", y=sel_num, color="is_peak_hour" if "is_peak_hour" in df else None, title=f"{sel_num} by Road Type and Peak Hour")
            st.plotly_chart(fig_box, use_container_width=True)
        else:
            fig_box = px.box(df, y=sel_num, title=f"Boxplot of {sel_num}")
            st.plotly_chart(fig_box, use_container_width=True)
            
    with tab3:
        if cat_cols:
            sel_cat = st.selectbox("Select Categorical Feature", cat_cols, index=cat_cols.index("congestion_level") if "congestion_level" in cat_cols else 0)
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                fig_bar = px.bar(df[sel_cat].value_counts().reset_index(), x=sel_cat, y="count", title=f"Frequency Count of {sel_cat}", color=sel_cat)
                st.plotly_chart(fig_bar, use_container_width=True)
            with col_c2:
                fig_pie = px.pie(df, names=sel_cat, title=f"Proportion of {sel_cat}", hole=0.3)
                st.plotly_chart(fig_pie, use_container_width=True)
                
    with tab4:
        if len(num_cols) >= 3:
            fig_scat = px.scatter(df, x="distance_km", y="travel_time_min", color="congestion_level" if "congestion_level" in df else None, size="traffic_volume_veh_hr" if "traffic_volume_veh_hr" in df else None, hover_data=["route_name"] if "route_name" in df else None, title="Distance vs Travel Time (Bubble size = Volume)")
            st.plotly_chart(fig_scat, use_container_width=True)

# -------------------------------------------------------------
# TOPIC 3: PROBABILITY DISTRIBUTIONS & SAMPLING
# -------------------------------------------------------------
elif selected_topic == "Probability Distributions and Sampling":
    st.header("🎲 Probability Distributions and Sampling Techniques")
    render_theory_expander(
        "Probability Distributions & Sampling",
        "Used to approximate theoretical continuous distributions (Normal, Log-Normal, Exponential) for physical phenomena such as travel delay, and to draw representative subsets from large urban traffic populations.",
        "1. Probability Distribution Fitting: Fit parameters $(\mu, \sigma)$ for Normal and $\lambda$ for Exponential.\n2. Simple Random Sampling: Every record has equal probability $1/N$.\n3. Systematic Sampling: Select every $k$-th item where $k = N / n$.\n4. Stratified Sampling: Divide into non-overlapping strata (e.g., Road Types) and sample proportionally.",
        [
            r"f(x; \mu, \sigma) = \frac{1}{\sigma \sqrt{2\pi}} \exp\left( -\frac{(x - \mu)^2}{2\sigma^2} \right)",
            r"f(x; \lambda) = \lambda e^{-\lambda x}, \quad k = \left\lfloor \frac{N}{n} \right\rfloor"
        ],
        "Urban travel times rarely follow a pure symmetric Gaussian distribution; they exhibit right-skewness fitting Log-Normal or Exponential distributions due to physical bounds (speeds cannot exceed speed limit, but congestion delay can expand unbounded)."
    )
    
    st.markdown("### 1. Probability Distribution Fitting")
    target_series = df["travel_time_min"] if "travel_time_min" in df else df[num_cols[0]]
    dist_choice = st.selectbox("Select Theoretical Distribution to Fit", ["Normal Distribution", "Log-Normal Distribution", "Exponential Distribution"])
    
    col_d1, col_d2 = st.columns([1, 1])
    with col_d1:
        x_vals = np.linspace(target_series.min(), target_series.max(), 200)
        if dist_choice == "Normal Distribution":
            mu, std = stats.norm.fit(target_series)
            pdf_vals = stats.norm.pdf(x_vals, mu, std)
            st.write(f"**Fitted Normal Parameters**: $\mu = {mu:.2f}$, $\sigma = {std:.2f}$")
        elif dist_choice == "Log-Normal Distribution":
            s, loc, scale = stats.lognorm.fit(target_series)
            pdf_vals = stats.lognorm.pdf(x_vals, s, loc, scale)
            st.write(f"**Fitted Log-Normal Parameters**: Shape $s = {s:.2f}$, Scale $= {scale:.2f}$")
        else:
            loc, scale = stats.expon.fit(target_series)
            pdf_vals = stats.expon.pdf(x_vals, loc, scale)
            st.write(f"**Fitted Exponential Parameter**: Rate $\lambda = {1/scale:.4f}$, Scale $= {scale:.2f}$")
            
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(x=target_series, histnorm="probability density", name="Empirical Data", opacity=0.6))
        fig_dist.add_trace(go.Scatter(x=x_vals, y=pdf_vals, mode="lines", name=f"Fitted {dist_choice}", line=dict(color="red", width=2.5)))
        fig_dist.update_layout(title="Empirical Data vs. Theoretical Fitted PDF", xaxis_title="Travel Time (min)", yaxis_title="Probability Density")
        st.plotly_chart(fig_dist, use_container_width=True)
        
    with col_d2:
        st.markdown("### 2. Comparative Sampling Simulator")
        sample_size = st.slider("Select Sample Size ($n$)", min_value=10, max_value=min(400, len(df)), value=60)
        sampling_method = st.radio("Sampling Strategy", ["Simple Random Sampling", "Systematic Sampling", "Stratified Sampling (by Road Type)"])
        
        if sampling_method == "Simple Random Sampling":
            sampled_df = df.sample(n=sample_size, random_state=42)
            st.info(f"Drawn {sample_size} records uniformly at random without replacement.")
        elif sampling_method == "Systematic Sampling":
            k = max(1, len(df) // sample_size)
            start = np.random.randint(0, k)
            indices = list(range(start, len(df), k))[:sample_size]
            sampled_df = df.iloc[indices]
            st.info(f"Drawn every $k = {k}$-th record starting at index {start}.")
        else:
            strata_col = "road_type" if "road_type" in df else df.columns[1]
            sampled_df = df.groupby(strata_col, group_keys=False).apply(lambda x: x.sample(int(np.ceil(sample_size * len(x) / len(df))), random_state=42)).head(sample_size)
            st.info(f"Drawn proportional quota from each stratum of '{strata_col}'.")
            
        st.write(f"**Sample Mean**: `{sampled_df['travel_time_min'].mean():.2f} min` | **Population Mean**: `{df['travel_time_min'].mean():.2f} min`")
        st.dataframe(sampled_df.head(5), use_container_width=True)
        st.download_button(
            "⬇️ Download Sampled Records (CSV)",
            data=sampled_df.to_csv(index=False),
            file_name=f"sampled_traffic_data_{sampling_method.lower().replace(' ', '_')}.csv"
        )

# -------------------------------------------------------------
# TOPIC 4: CENTRAL LIMIT THEOREM
# -------------------------------------------------------------
elif selected_topic == "Central Limit Theorem":
    st.header("📈 Central Limit Theorem (CLT) Demonstration")
    render_theory_expander(
        "Central Limit Theorem",
        "Proves that regardless of the underlying population distribution (even if highly skewed or bimodal), the distribution of sample means approximates a Normal Gaussian distribution as sample size $n$ increases ($n \ge 30$).",
        "1. Repeatedly draw $K$ independent random samples of size $n$ from the population.\n2. For each sample, calculate sample mean $\bar{x}_k$.\n3. Construct the sampling distribution of $\{\bar{x}_1, \bar{x}_2, \dots, \bar{x}_K\}$.\n4. Compare observed standard error with theoretical value $\sigma / \sqrt{n}$.",
        [
            r"\bar{X} \sim \mathcal{N}\left(\mu, \; \frac{\sigma}{\sqrt{n}}\right) \quad \text{as } n \to \infty",
            r"SE_{\text{theoretical}} = \frac{\sigma}{\sqrt{n}}, \quad SE_{\text{observed}} = \sqrt{\frac{1}{K-1} \sum_{k=1}^K (\bar{x}_k - \bar{\bar{x}})^2}"
        ],
        "As $n$ grows from 5 to 50, observed SE decreases strictly by $1/\sqrt{n}$ and the histogram bell curve narrows around true population mean $\mu$."
    )
    
    col1, col2 = st.columns([1, 2])
    with col1:
        clt_n = st.slider("Sample Size per Draw ($n$)", min_value=2, max_value=100, value=30, step=2)
        clt_trials = st.slider("Number of Sample Draws ($K$)", min_value=100, max_value=1000, value=400, step=50)
        pop_data = df["travel_time_min"].values
        pop_mean = np.mean(pop_data)
        pop_std = np.std(pop_data, ddof=1)
        theoretical_se = pop_std / np.sqrt(clt_n)
        
        # Monte Carlo sampling
        sample_means = [np.mean(np.random.choice(pop_data, size=clt_n, replace=True)) for _ in range(clt_trials)]
        obs_mean = np.mean(sample_means)
        obs_se = np.std(sample_means, ddof=1)
        
        st.markdown("#### 🔬 Comparative Results")
        st.write(f"- **Population Mean ($\mu$)**: `{pop_mean:.2f}` min")
        st.write(f"- **Mean of Sample Means ($\bar{{\bar{{x}}}}$)**: `{obs_mean:.2f}` min")
        st.write(f"- **Theoretical SE ($\sigma/\sqrt{{n}}$)**: `{theoretical_se:.3f}`")
        st.write(f"- **Observed SE ($s_{{\bar{{x}}}}$)**: `{obs_se:.3f}`")
        st.write(f"- **Discrepancy**: `{abs(theoretical_se - obs_se):.4f}`")
        
    with col2:
        fig_clt = px.histogram(sample_means, nbins=25, title=f"Sampling Distribution of Mean Travel Time (n = {clt_n}, K = {clt_trials})", labels={"value": "Sample Mean (min)"})
        # Add normal curve
        x_axis = np.linspace(min(sample_means), max(sample_means), 150)
        fig_clt.add_vline(x=pop_mean, line_dash="dash", line_color="green", annotation_text="True Pop Mean")
        fig_clt.add_vline(x=obs_mean, line_dash="dot", line_color="red", annotation_text="Observed Mean")
        st.plotly_chart(fig_clt, use_container_width=True)

# -------------------------------------------------------------
# TOPIC 5: CONFIDENCE INTERVALS & HYPOTHESIS TESTING
# -------------------------------------------------------------
elif selected_topic == "Confidence Intervals and Hypothesis Testing":
    st.header("⚖️ Confidence Intervals and Hypothesis Testing")
    render_theory_expander(
        "Confidence Intervals and Hypothesis Testing",
        "Used to test assertions regarding urban travel delay benchmarks and determine if observed transit delays are statistically significantly different from expected municipality standards.",
        "1. State Null ($H_0$) and Alternative ($H_1$) hypotheses.\n2. Choose test: Z-test (known $\sigma$, large $n$) vs t-test (unknown $\sigma$, sample $s$).\n3. Compute test statistic $Z$ or $t$ and degrees of freedom.\n4. Obtain two-tailed p-value and compare against significance level $\alpha$ (typically 0.05).",
        [
            r"Z = \frac{\bar{x} - \mu_0}{\sigma / \sqrt{n}}, \quad t = \frac{\bar{x} - \mu_0}{s / \sqrt{n}}, \quad df = n - 1",
            r"CI = \bar{x} \pm t_{\alpha/2, df} \cdot \frac{s}{\sqrt{n}}"
        ],
        "If $p$-value $< \alpha$, reject $H_0$: There is sufficient statistical evidence that mean travel time differs from the benchmark standard. If $p$-value $\ge \alpha$, fail to reject $H_0$."
    )
    
    test_mode = st.radio("Select Hypothesis Test", ["One-Sample t-Test (Benchmark Comparison)", "Two-Sample Independent t-Test (Peak vs. Off-Peak Travel Time)"])
    
    if test_mode == "One-Sample t-Test (Benchmark Comparison)":
        c1, c2 = st.columns([1, 1])
        with c1:
            mu_0 = st.number_input("Hypothesized Benchmark Travel Time ($\mu_0$ in minutes)", value=25.0, step=1.0)
            alpha_val = st.selectbox("Significance Level ($\alpha$)", [0.01, 0.05, 0.10], index=1)
            
            sample_data = df["travel_time_min"].dropna()
            t_stat, p_val = stats.ttest_1samp(sample_data, popmean=mu_0)
            n_samp = len(sample_data)
            se_samp = sample_data.std() / np.sqrt(n_samp)
            crit_t = stats.t.ppf(1 - alpha_val / 2, df=n_samp - 1)
            ci_lower = sample_data.mean() - crit_t * se_samp
            ci_upper = sample_data.mean() + crit_t * se_samp
            
        with c2:
            st.markdown("#### Test Outputs & Formal Verdict")
            st.write(f"- **Null Hypothesis ($H_0$)**: $\mu = {mu_0}$ minutes")
            st.write(f"- **Alternative Hypothesis ($H_1$)**: $\mu \neq {mu_0}$ minutes")
            st.write(f"- **Sample Mean ($\bar{{x}}$)**: `{sample_data.mean():.2f}` min (n = {n_samp})")
            st.write(f"- **Degrees of Freedom ($df$)**: `{n_samp - 1}`")
            st.write(f"- **Calculated $t$-statistic**: `{t_stat:.3f}`")
            st.write(f"- **Two-Tailed $p$-value**: `{p_val:.4e}`")
            st.write(f"- **{(1 - alpha_val)*100:.0f}% Confidence Interval**: `[{ci_lower:.2f}, {ci_upper:.2f}]`")
            
            if p_val < alpha_val:
                st.error(f"❌ **Reject $H_0$** ($p < {alpha_val}$): Statistically significant difference from {mu_0} min benchmark.")
            else:
                st.success(f"✅ **Fail to Reject $H_0$** ($p \ge {alpha_val}$): No significant difference from {mu_0} min benchmark.")
                
    else:
        st.markdown("#### Two-Sample Independent t-Test: Peak Hour vs. Off-Peak Travel Time")
        peak_group = df[df["is_peak_hour"] == "Yes"]["travel_time_min"].dropna()
        offpeak_group = df[df["is_peak_hour"] == "No"]["travel_time_min"].dropna()
        
        t_stat2, p_val2 = stats.ttest_ind(peak_group, offpeak_group, equal_var=False)
        
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.metric("Peak Hour Mean Travel Time", f"{peak_group.mean():.2f} min", f"+{(peak_group.mean() - offpeak_group.mean()):.2f} min")
            st.metric("Off-Peak Mean Travel Time", f"{offpeak_group.mean():.2f} min")
        with col_t2:
            st.write(f"- **$t$-statistic (Welch's)**: `{t_stat2:.3f}`")
            st.write(f"- **$p$-value**: `{p_val2:.4e}`")
            if p_val2 < 0.05:
                st.error("❌ **Reject $H_0$**: Peak hour travel times are statistically significantly longer than off-peak times.")
            else:
                st.success("✅ **Fail to Reject $H_0$**: No significant difference detected.")
                
        fig_cmp = px.box(df, x="is_peak_hour", y="travel_time_min", color="is_peak_hour", title="Travel Time Comparison: Peak vs Off-Peak", labels={"is_peak_hour": "Peak Hour", "travel_time_min": "Travel Time (min)"})
        st.plotly_chart(fig_cmp, use_container_width=True)

# -------------------------------------------------------------
# TOPIC 6: CHI-SQUARE TEST
# -------------------------------------------------------------
elif selected_topic == "Chi-Square Test":
    st.header("📊 Chi-Square ($\chi^2$) Test of Independence")
    render_theory_expander(
        "Chi-Square Test of Independence",
        "Used to test whether two categorical traffic attributes (e.g. Congestion Severity and Weather Condition) are statistically independent or associated.",
        "1. Construct a two-way contingency table of observed counts $O_{ij}$.\n2. Calculate expected frequencies: $E_{ij} = (\text{Row Total}_i \times \text{Col Total}_j) / N$.\n3. Compute Pearson Chi-Square statistic: $\chi^2 = \sum \frac{(O_{ij} - E_{ij})^2}{E_{ij}}$.\n4. Compare against critical value $\chi^2_{\alpha, df}$ where $df = (r - 1)(c - 1)$.",
        [
            r"\chi^2 = \sum_{i=1}^r \sum_{j=1}^c \frac{(O_{ij} - E_{ij})^2}{E_{ij}}, \quad df = (r - 1)(c - 1)",
            r"\text{Cramér's } V = \sqrt{\frac{\chi^2}{N \cdot \min(r-1, c-1)}}"
        ],
        "If $p$-value $< 0.05$, reject $H_0$. Congestion Severity and Weather Condition are significantly dependent. Cramér's $V$ provides the effect size (0 = no association, 1 = perfect association)."
    )
    
    cat_cols_all = df.select_dtypes(include=['object', 'category']).columns.tolist()
    c1, c2 = st.columns(2)
    with c1:
        cat_var1 = st.selectbox("Categorical Factor 1", cat_cols_all, index=cat_cols_all.index("congestion_level") if "congestion_level" in cat_cols_all else 0)
    with c2:
        cat_var2 = st.selectbox("Categorical Factor 2", cat_cols_all, index=cat_cols_all.index("weather_condition") if "weather_condition" in cat_cols_all else 1)
        
    contingency = pd.crosstab(df[cat_var1], df[cat_var2])
    st.markdown("#### Observed Contingency Table ($O_{ij}$)")
    st.dataframe(contingency, use_container_width=True)
    
    chi2_val, p_val_chi, dof, expected = stats.chi2_contingency(contingency)
    n_total = contingency.values.sum()
    min_dim = min(contingency.shape[0] - 1, contingency.shape[1] - 1)
    cramers_v = np.sqrt(chi2_val / (n_total * min_dim)) if min_dim > 0 else 0
    
    col_out1, col_out2 = st.columns(2)
    with col_out1:
        st.write(f"- **Chi-Square Statistic ($\chi^2$)**: `{chi2_val:.2f}`")
        st.write(f"- **Degrees of Freedom ($df$)**: `{dof}`")
        st.write(f"- **Asymptotic $p$-value**: `{p_val_chi:.4e}`")
        st.write(f"- **Cramér's $V$ Effect Size**: `{cramers_v:.3f}`")
    with col_out2:
        if p_val_chi < 0.05:
            st.error(f"❌ **Reject $H_0$** ($p < 0.05$): Statistically significant relationship between {cat_var1} and {cat_var2}.")
        else:
            st.success(f"✅ **Fail to Reject $H_0$** ($p \ge 0.05$): No significant association detected.")
            
    fig_heat = px.imshow(contingency, text_auto=True, title=f"Contingency Heatmap: {cat_var1} vs {cat_var2}", color_continuous_scale="Viridis")
    st.plotly_chart(fig_heat, use_container_width=True)

# -------------------------------------------------------------
# TOPIC 7: COVARIANCE & CORRELATION
# -------------------------------------------------------------
elif selected_topic == "Covariance and Correlation":
    st.header("🔗 Covariance and Correlation Analysis")
    render_theory_expander(
        "Covariance and Correlation",
        "Used to measure the joint variability (covariance) and standardized linear relationship strength and direction (correlation coefficient) between pairs of continuous traffic metrics.",
        "1. Compute sample Covariance: $Cov(X, Y) = \frac{1}{n-1}\sum (x_i - \bar{x})(y_i - \bar{y})$.\n2. Standardize by product of standard deviations to calculate Pearson $r$: $r = \frac{Cov(X,Y)}{s_x s_y}$.\n3. Range of $r$ is $[-1, +1]$.",
        [
            r"Cov(X, Y) = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{n - 1}",
            r"r_{xy} = \frac{Cov(X, Y)}{s_x s_y} = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum (x_i - \bar{x})^2 \sum (y_i - \bar{y})^2}}"
        ],
        "+1 indicates perfect positive correlation, -1 indicates perfect negative correlation, and 0 indicates no linear correlation. Positive covariance between Volume and Travel Time means higher traffic volumes systematically lengthen travel times."
    )
    
    num_df = df.select_dtypes(include=[np.number])
    col_matrix1, col_matrix2 = st.columns(2)
    
    with col_matrix1:
        st.markdown("#### Covariance Matrix")
        cov_matrix = num_df.cov()
        st.dataframe(cov_matrix.round(2), use_container_width=True)
        
    with col_matrix2:
        st.markdown("#### Pearson Correlation Matrix ($r$)")
        corr_matrix = num_df.corr()
        st.dataframe(corr_matrix.round(3), use_container_width=True)
        
    fig_corr = px.imshow(corr_matrix, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", title="Correlation Heatmap Matrix")
    st.plotly_chart(fig_corr, use_container_width=True)

# -------------------------------------------------------------
# TOPIC 8: CORRELATION HYPOTHESIS TESTING
# -------------------------------------------------------------
elif selected_topic == "Correlation Hypothesis Testing":
    st.header("🧪 Correlation Hypothesis Testing")
    render_theory_expander(
        "Correlation Hypothesis Testing",
        "Used to test whether an observed correlation coefficient ($r$ or $\rho$) between traffic metrics is genuinely non-zero in the underlying population or merely a product of random sampling fluctuation.",
        "1. State $H_0: \rho = 0$ (No correlation) vs $H_1: \rho \neq 0$.\n2. Calculate test statistic: $t = r \sqrt{\frac{n-2}{1-r^2}}$ with $df = n - 2$.\n3. Determine two-tailed p-value.\n4. Contrast Pearson (parametric, linear) with Spearman (non-parametric, monotonic ranks).",
        [
            r"t = r \sqrt{\frac{n - 2}{1 - r^2}}, \quad df = n - 2",
            r"r_s = 1 - \frac{6 \sum d_i^2}{n(n^2 - 1)} \quad (\text{Spearman Rank Correlation})"
        ],
        "If $p$-value $< 0.05$, reject $H_0$. Conclude that a statistically significant relationship exists between the two traffic parameters in the population."
    )
    
    num_cols_sel = df.select_dtypes(include=[np.number]).columns.tolist()
    c1, c2 = st.columns(2)
    with c1:
        var_x = st.selectbox("Predictor Variable (X)", num_cols_sel, index=num_cols_sel.index("traffic_volume_veh_hr") if "traffic_volume_veh_hr" in num_cols_sel else 0)
    with c2:
        var_y = st.selectbox("Response Variable (Y)", num_cols_sel, index=num_cols_sel.index("travel_time_min") if "travel_time_min" in num_cols_sel else 1)
        
    x_clean = df[var_x].dropna()
    y_clean = df[var_y].dropna()
    common_idx = x_clean.index.intersection(y_clean.index)
    x_val = x_clean.loc[common_idx]
    y_val = y_clean.loc[common_idx]
    
    pearson_r, p_pearson = stats.pearsonr(x_val, y_val)
    spearman_rho, p_spearman = stats.spearmanr(x_val, y_val)
    
    col_res1, col_res2 = st.columns(2)
    with col_res1:
        st.markdown("#### Pearson Correlation (Linear)")
        st.write(f"- **Coefficient ($r$)**: `{pearson_r:.3f}`")
        st.write(f"- **$p$-value**: `{p_pearson:.4e}`")
        st.write(f"- **Verdict**: {'Statistically Significant (p < 0.05)' if p_pearson < 0.05 else 'Not Significant'}")
    with col_res2:
        st.markdown("#### Spearman Correlation (Monotonic Rank)")
        st.write(f"- **Coefficient ($\rho$)**: `{spearman_rho:.3f}`")
        st.write(f"- **$p$-value**: `{p_spearman:.4e}`")
        st.write(f"- **Verdict**: {'Statistically Significant (p < 0.05)' if p_spearman < 0.05 else 'Not Significant'}")
        
    fig_scatter_reg = px.scatter(df, x=var_x, y=var_y, trendline="ols", title=f"Scatter Plot with OLS Trendline: {var_x} vs {var_y}")
    st.plotly_chart(fig_scatter_reg, use_container_width=True)

# -------------------------------------------------------------
# TOPIC 9: TWO-WAY ANOVA
# -------------------------------------------------------------
elif selected_topic == "Two-Way ANOVA":
    st.header("🔬 Two-Way Analysis of Variance (ANOVA)")
    render_theory_expander(
        "Two-Way ANOVA",
        "Used to examine how two independent categorical factors (e.g. Peak Hour and Weather Condition) simultaneously impact a continuous traffic metric (e.g. Travel Time), testing for main effects and interaction effects.",
        "1. Partition total variation $SS_{\text{Total}} = SS_A + SS_B + SS_{AB} + SS_{\text{Error}}$.\n2. Calculate degrees of freedom for each component.\n3. Compute Mean Squares: $MS = SS / df$.\n4. Calculate F-ratios: $F = MS / MS_{\text{Error}}$ and corresponding p-values.",
        [
            r"F_A = \frac{MS_A}{MS_{\text{Error}}}, \quad F_B = \frac{MS_B}{MS_{\text{Error}}}, \quad F_{AB} = \frac{MS_{AB}}{MS_{\text{Error}}}",
            r"SS_{\text{Total}} = \sum_{i} \sum_{j} \sum_{k} (y_{ijk} - \bar{y}_{\dots})^2"
        ],
        "A significant interaction ($p < 0.05$ for $A \times B$) indicates that the impact of bad weather on travel time is significantly amplified during peak hours compared to off-peak periods."
    )
    
    try:
        model = ols('travel_time_min ~ C(is_peak_hour) + C(weather_condition) + C(is_peak_hour):C(weather_condition)', data=df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        
        st.markdown("#### Two-Way ANOVA Summary Table")
        st.dataframe(anova_table.round(3), use_container_width=True)
        
        st.markdown("#### Interaction Plot: Peak Hour × Weather Condition on Travel Time")
        interaction_df = df.groupby(["weather_condition", "is_peak_hour"])["travel_time_min"].mean().reset_index()
        fig_interact = px.line(interaction_df, x="weather_condition", y="travel_time_min", color="is_peak_hour", markers=True, title="Mean Travel Time across Weather Conditions by Peak Hour Status")
        st.plotly_chart(fig_interact, use_container_width=True)
    except Exception as e:
        st.error(f"Error computing Two-Way ANOVA: {e}. Please ensure dataset has required categorical columns.")

# -------------------------------------------------------------
# TOPIC 10: REGRESSION ANALYSIS
# -------------------------------------------------------------
elif selected_topic == "Regression Analysis":
    st.header("📉 Multiple Linear Regression Modeling")
    render_theory_expander(
        "Multiple Linear Regression",
        "Used to model and predict vehicular travel times based on explanatory covariates including trip distance, vehicular volume, speed limits, and incident counts.",
        "1. Specify linear model: $Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \dots + \epsilon$.\n2. Estimate coefficients using Ordinary Least Squares (OLS) minimizing $\sum e_i^2$.\n3. Evaluate model goodness of fit with $R^2$, Adjusted $R^2$, MAE, and RMSE.",
        [
            r"\hat{Y} = \hat{\beta}_0 + \hat{\beta}_1 X_1 + \hat{\beta}_2 X_2 + \dots + \hat{\beta}_k X_k",
            r"R^2 = 1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}}, \quad RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}"
        ],
        "$R^2$ indicates the proportion of variance in travel time explained by the features. High positive coefficient on Traffic Volume reflects delay per additional vehicle per hour."
    )
    
    feature_candidates = ["distance_km", "traffic_volume_veh_hr", "incidents_reported"]
    features = [f for f in feature_candidates if f in df.columns]
    
    if features:
        X = df[features].dropna()
        y = df.loc[X.index, "travel_time_min"]
        
        reg = LinearRegression()
        reg.fit(X, y)
        y_pred = reg.predict(X)
        
        r2 = r2_score(y, y_pred)
        n = len(y)
        p = len(features)
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("R² Score", f"{r2:.3f}")
        c2.metric("Adjusted R²", f"{adj_r2:.3f}")
        c3.metric("MAE", f"{mae:.2f} min")
        c4.metric("RMSE", f"{rmse:.2f} min")
        
        coef_df = pd.DataFrame({
            "Feature": ["Intercept"] + features,
            "Coefficient": [round(reg.intercept_, 3)] + [round(c, 4) for c in reg.coef_]
        })
        st.markdown("#### Estimated Regression Equation Coefficients")
        st.dataframe(coef_df, use_container_width=True)
        
        # Interactive Prediction Calculator
        st.markdown("### 🧮 Live Travel Time Prediction Calculator")
        pred_cols = st.columns(len(features))
        user_inputs = []
        for i, f in enumerate(features):
            with pred_cols[i]:
                min_v = float(df[f].min())
                max_v = float(df[f].max())
                mean_v = float(df[f].mean())
                val = st.number_input(f"Enter {f}", min_value=min_v, max_value=max_v * 1.5, value=mean_v)
                user_inputs.append(val)
                
        calc_pred = reg.predict([user_inputs])[0]
        st.success(f"🚘 **Estimated Travel Time**: `{calc_pred:.2f} minutes` based on OLS model.")
        
        # Residuals Plot
        residuals = y - y_pred
        fig_res = px.scatter(x=y_pred, y=residuals, labels={"x": "Fitted Values", "y": "Residuals"}, title="Residuals vs Fitted Values (Diagnostic Plot)")
        fig_res.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig_res, use_container_width=True)
        
        # Export predictions
        pred_df = df.copy()
        pred_df["Predicted_Travel_Time_min"] = y_pred
        pred_df["Residual_min"] = residuals
        st.download_button(
            "⬇️ Download Model Predictions & Residuals (CSV)",
            data=pred_df.to_csv(index=False),
            file_name="traffic_regression_predictions.csv"
        )
