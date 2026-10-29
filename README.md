# Statistical Analysis of Urban Traffic and Travel Time
**Big Data Analytics Lab Project**
*Deployed on Streamlit Community Cloud*

---

## 📌 Project Overview
This project performs an end-to-end statistical analysis of urban traffic congestion, vehicular volumes, and travel times across metropolitan corridors. It demonstrates key concepts of Big Data Analytics, Inferential Statistics, Sampling Theory, Hypothesis Testing, and Predictive Modeling through a unified, interactive Streamlit web application.

---

## 🎯 Syllabus & Practical Modules Implemented
1. **Web Scraping & CSV Creation**: Automated data collection simulation with HTTP fallback and error handling.
2. **Data Cleaning & Exploratory Data Analysis (EDA)**: Handling missing values, outlier detection (IQR), 5-number summary, histograms, and box plots.
3. **Probability Distributions & Sampling**: Normal & Exponential distribution fitting; Simple Random, Systematic, and Stratified sampling.
4. **Central Limit Theorem (CLT)**: Monte Carlo sampling demonstration with observed vs theoretical Standard Errors.
5. **Confidence Intervals & Hypothesis Testing**: One-sample Z-test, One-sample t-test, and Two-sample independent t-test (Peak vs. Off-Peak).
6. **Chi-Square Test**: Test of Independence between Congestion Level and Weather Conditions / Road Types.
7. **Covariance & Correlation**: Direction and strength of relationships between Volume, Speed, Distance, and Travel Time.
8. **Correlation Hypothesis Testing**: Pearson ($r$) and Spearman ($\rho$) significance tests with two-tailed p-values.
9. **Two-Way ANOVA**: Evaluating the effects of Peak Hours and Weather Conditions on Travel Time.
10. **Multiple Linear Regression**: Ordinary Least Squares (OLS) model for travel time prediction with $R^2$, MAE, and RMSE evaluation.

---

## 🚀 Local Installation & Execution

### Prerequisites
- Python 3.10 or higher
- Git installed on your system

### Steps
```bash
# 1. Clone your repository
git clone https://github.com/<YOUR_USERNAME>/urban-traffic-analytics.git
cd urban-traffic-analytics

# 2. Create and activate a virtual environment (optional but recommended)
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS / Linux:
source venv/bin/activate

# 3. Install required packages
pip install -r requirements.txt

# 4. Run the Streamlit web application
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`.

---

## 🌐 Deploy to Streamlit Community Cloud
1. Push your project files (`app.py`, `requirements.txt`, `urban_traffic_travel_time.csv`, `README.md`) to a public GitHub repository.
2. Visit [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
3. Click **"New app"**.
4. Select your repository, set the branch to `main`, and specify the main file as `app.py`.
5. Click **"Deploy!"**. Your app will be live within 2 minutes with a shareable public URL!

---

## 📊 Dataset Dictionary
- `record_id`: Unique identifier for each traffic sensor observation.
- `route_name`: Metropolitan corridor name.
- `road_type`: Classification (Highway/Expressway, Arterial Road, Collector Road, Downtown Local).
- `distance_km`: Length of corridor segment (in kilometers).
- `free_flow_time_min`: Baseline travel time at posted speed limit (minutes).
- `travel_time_min`: Observed actual travel time including congestion delays (minutes).
- `traffic_volume_veh_hr`: Number of vehicles passing sensor per hour (veh/hr).
- `average_speed_kmh`: Mean vehicular transit speed (km/h).
- `weather_condition`: Ambient weather (Clear, Rainy, Foggy, Heavy Rain).
- `day_of_week`: Day of observation.
- `is_peak_hour`: Binary flag ('Yes' for 08:00-10:00 or 17:00-19:00, 'No' otherwise).
- `congestion_level`: Categorical severity ('Low', 'Moderate', 'High', 'Severe').
- `incidents_reported`: Number of traffic incidents or stalls reported.
