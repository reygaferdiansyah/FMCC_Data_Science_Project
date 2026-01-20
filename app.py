import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ======================================================
# 1. PLATINUM DARK CONFIGURATION
# ======================================================
st.set_page_config(
    page_title="OLIST Intelligence HQ",
    layout="wide",
    page_icon="💎",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .block-container { padding-top: 2rem; }
    div[data-testid="stMetricValue"] { font-size: 28px; font-weight: 700; color: #FAFAFA; }
    div[data-testid="stMetricLabel"] { font-size: 14px; color: #A0AEC0; }
    h1, h2, h3 { font-family: 'Segoe UI', sans-serif; font-weight: 600; }
    .streamlit-expanderHeader { background-color: #161B26; color: #FAFAFA; border-radius: 5px; }
    div[data-testid="stDataFrame"] { border: 1px solid #2d3748; border-radius: 10px; }
    
    /* Custom Card Style */
    .metric-card {
        background-color: #1E293B;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ======================================================
# 2. DATA LOADING
# ======================================================
@st.cache_data
def load_data():
    rfm, forecast = None, None
    try:
        rfm = pd.read_csv("olist_rfm_final_for_engine.csv")
    except:
        st.error("⚠️ Error: 'olist_rfm_final_for_engine.csv' not found.")
    try:
        forecast = pd.read_csv("olist_weekly_engineered.csv")
        forecast["date"] = pd.to_datetime(forecast["date"])
    except:
        st.error("⚠️ Error: 'olist_weekly_engineered.csv' not found.")
    return rfm, forecast

@st.cache_resource
def load_brain():
    model, sx, sy = None, None, None
    try:
        model = load_model("olist_revenue_lstm.h5", compile=False)
        sx = joblib.load("scaler_X.joblib")
        sy = joblib.load("scaler_y.joblib")
    except Exception as e:
        st.warning(f"⚠️ Model AI not ready: {e}")
    features = ['n_orders', 'month_sin', 'month_cos', 'week_sin', 'week_cos', 
                'is_black_friday', 'is_december', 'is_peak_season', 
                'orders_x_bf', 'orders_x_peak']
    return model, sx, sy, features

rfm_df, hist_df = load_data()
model, scaler_x, scaler_y, FEATURES = load_brain()

# ======================================================
# 3. DYNAMIC FUTURE ENGINE
# ======================================================
def generate_future_input(start_date, baseline_orders, periods):
    future_dates = pd.date_range(start=start_date + pd.Timedelta(days=7), periods=periods, freq='W-MON')
    
    future_rows = []
    np.random.seed(42) 
    
    for i, date in enumerate(future_dates):
        month = date.month
        week = date.isocalendar().week
        m_sin = np.sin(2 * np.pi * month / 12)
        m_cos = np.cos(2 * np.pi * month / 12)
        w_sin = np.sin(2 * np.pi * week / 53)
        w_cos = np.cos(2 * np.pi * week / 53)
        
        is_dec = 1 if month == 12 else 0
        is_peak = 1 if month in [11, 12] else 0
        is_bf = 1 if (month == 11 and 23 <= date.day <= 29) else 0
        
        noise = np.random.normal(0, 0.10) 
        growth = 1 + (i * 0.002) 
        simulated_orders = baseline_orders * (1 + noise) * growth
        
        if is_bf: simulated_orders *= 1.5 
            
        row = [simulated_orders, m_sin, m_cos, w_sin, w_cos, is_bf, is_dec, is_peak, simulated_orders * is_bf, simulated_orders * is_peak]
        future_rows.append(row)
        
    return np.array(future_rows), future_dates

# ======================================================
# 4. STRATEGIC DECISION ENGINE
# ======================================================
def analyze_strategy(change_pct):
    if change_pct < 0:
        strategy = "🚨 REVENUE DROP ALERT"
        risk_level = "HIGH RISK"
        risk_color = "#e74c3c" # Red
        desc = "Decline detected vs last month. Focus on Retention & Win-Back campaigns."
        target_segments = ["At Risk", "Lost", "Hibernating"]
    elif change_pct == 0:
        strategy = "⚖️ STABLE / INSUFFICIENT DATA"
        risk_level = "NEUTRAL"
        risk_color = "#A0AEC0" # Grey
        desc = "Revenue is stagnant or insufficient historical data for comparison."
        target_segments = ["Promising"]
    else:
        strategy = "📈 GROWTH SIGNAL"
        risk_level = "LOW RISK"
        risk_color = "#2ecc71" # Green
        desc = "Positive trend maintained. Opportunity for Upselling & Expansion."
        target_segments = ["Champions", "Loyal Customers"]

    return strategy, risk_level, risk_color, desc, target_segments

# ======================================================
# 5. UI: SIDEBAR
# ======================================================
st.sidebar.title("💎 OLIST INTELLIGENCE")
st.sidebar.markdown("---")

# --- MENU NAVIGATION ---
menu = st.sidebar.radio("Module Access:", 
    ["About Me", "Model Performance", "Customer Insights", "Future Sight (AI Forecast)"]
)
st.sidebar.markdown("---")
st.sidebar.caption("System vFinal (Pro English) | Bi-LSTM")

# ======================================================
# 6. MODULES
# ======================================================

# --- MODULE 0: ABOUT ME (PROFILE) ---
if menu == "About Me":
    st.title("👋 About Me")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 👨‍💻 Professional Summary")
        st.write("""
        **Bachelor of Computer Science graduate** pursuing a **Master’s in Information Technology at BINUS University**, 
        specializing in **Data Science and AI Engineering**. 
        
        Experienced in data analysis and building ML/DL models. Completed the Data Science Bootcamp at **Dibimbing.id** and earned the **Microsoft Azure AI Certification**. 
        Gained hands-on experience as a Data Science Intern at **Urbansolv**, developing IoT-based forecasting and analytics solutions. 
        Passionate about creating data-driven and AI-powered solutions with real-world impact.
        """)
        
        st.markdown("### 💼 Work Experience")
        
        st.markdown("##### **Data Science (Contract)** @ URBANSOLV")
        st.caption("September 2025 – Present")
        st.info("""
        * Delivering monitoring project for client **Pertamina** focused on building Power BI dashboards.
        * Integrating operational data into automated pipelines.
        * Defining KPIs with Pertamina teams to enhance operational transparency and field supervision effectiveness.
        """)

        st.markdown("##### **Data Science Intern** @ URBANSOLV")
        st.caption("July 2025 – September 2025")
        st.write("""
        * Developed **ML/DL models** for time series forecasting & classification using IoT water sensor data.
        * Conducted **web scraping** to collect and structure external lead data.
        * Performed **geospatial analysis** using GeoPandas and QGIS.
        * Applied clustering techniques (K-Means, Agglomerative) for customer segmentation.
        * Built interactive dashboards using Streamlit, Power BI, and Plotly Dash.
        """)

        st.markdown("### 🎓 Education")
        st.write("""
        * **Master of Information Technology** - BINUS University (Nov 2024 - Present)
        * **Data Science Bootcamp** - Dibimbing.id (Nov 2024 - June 2025)
        * **Bachelor of Information Technology** - STIKOM CKI (2019 - 2023) | GPA: 3.75
        """)

    with col2:
        st.markdown("### 🛠️ Technical Skills")
        st.success("""
        **Languages & Tools:**
        * Python, SQL
        * Power BI, Streamlit
        * Jupyter Notebook, Git
        
        **Core Competencies:**
        * Machine Learning & Deep Learning
        * IoT Integration & Analytics
        * Cloud (Azure, AWS)
        * Data Visualization
        * Geospatial Analysis
        """)
        
        st.markdown("### 📬 Contact")
        st.markdown("""
        - 💼 [LinkedIn](https://www.linkedin.com/in/reyga-ferdiansyah)  
        - 🛠️ [GitHub](https://github.com/reygaferdiansyah)  
        - 📧 [reygafp@gmail.com](mailto:reygafp@gmail.com)
        - 📱 [WhatsApp +62 851-5651-4266](https://wa.me/6285156514266)
        """)

# --- MODULE 1: MODEL PERFORMANCE ---
elif menu == "Model Performance":
    st.title("🤖 Model Performance Audit")
    st.markdown("Evaluating **Bi-LSTM Model** Accuracy: *Actual vs Predicted* on Historical Data.")

    if model and hist_df is not None and FEATURES:
        df_eval = hist_df.copy().sort_values('date')
        train_size = int(len(df_eval) * 0.8)
        train_df = df_eval.iloc[:train_size]
        test_df = df_eval.iloc[train_size:]
        
        try:
            X_train = scaler_x.transform(train_df[FEATURES])
            X_test = scaler_x.transform(test_df[FEATURES])
            X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
            X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
            
            y_train_pred = scaler_y.inverse_transform(model.predict(X_train)).flatten()
            y_test_pred = scaler_y.inverse_transform(model.predict(X_test)).flatten()
            
            train_df['Predicted'] = y_train_pred
            test_df['Predicted'] = y_test_pred
            y_test_real = test_df['revenue'].values
            
            mae = mean_absolute_error(y_test_real, y_test_pred)
            rmse = np.sqrt(mean_squared_error(y_test_real, y_test_pred))
            mape = np.mean(np.abs((y_test_real - y_test_pred) / y_test_real)) * 100
            r2 = r2_score(y_test_real, y_test_pred)
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("R2 Score (Accuracy)", f"{r2:.2%}", help="How well the model explains variance (Closer to 100% is better)")
            c2.metric("Test MAE", f"R$ {mae:,.0f}", help="Mean Absolute Error")
            c3.metric("Test RMSE", f"R$ {rmse:,.0f}", help="Root Mean Squared Error")
            c4.metric("Test MAPE", f"{mape:.2f}%", help="Mean Absolute Percentage Error")
            
            st.markdown("---")
            
            st.subheader("📉 Actual vs Predicted Timeline (Test Data)")
            fig_eval = go.Figure()
            fig_eval.add_trace(go.Scatter(x=train_df['date'], y=train_df['revenue'], mode='lines', name='Training Data', line=dict(color='#4A5568', width=1)))
            fig_eval.add_trace(go.Scatter(x=test_df['date'], y=test_df['revenue'], mode='lines', name='Actual Test Data', line=dict(color='#3182CE', width=2)))
            fig_eval.add_trace(go.Scatter(x=test_df['date'], y=test_df['Predicted'], mode='lines', name='AI Prediction', line=dict(color='#F6E05E', width=2, dash='dot')))
            fig_eval.update_layout(template="plotly_dark", height=500, hovermode="x unified", title="Model Performance Overview")
            st.plotly_chart(fig_eval, use_container_width=True)
            
            c_left, c_right = st.columns([1, 1])
            with c_left:
                st.subheader("🎯 Prediction Correlation")
                fig_scatter = px.scatter(x=y_test_real, y=y_test_pred, labels={'x': 'Actual Value', 'y': 'Predicted Value'}, template="plotly_dark", title=f"Correlation (R2: {r2:.2f})", color_discrete_sequence=['#00CC96'])
                fig_scatter.add_shape(type="line", x0=min(y_test_real), y0=min(y_test_real), x1=max(y_test_real), y1=max(y_test_real), line=dict(color="#E74C3C", dash="dash"))
                st.plotly_chart(fig_scatter, use_container_width=True)
            with c_right:
                st.subheader("🔍 Error Distribution (Residuals)")
                residuals = y_test_real - y_test_pred
                fig_res = ff.create_distplot([residuals], ['Residuals'], bin_size=(max(residuals)-min(residuals))/20, show_hist=True, show_rug=False, colors=['#3B82F6'])
                fig_res.update_layout(template="plotly_dark", showlegend=False, height=400, title="Zero = Perfect Prediction")
                fig_res.add_shape(type="line", x0=0, y0=0, x1=0, y1=1, xref='x', yref='paper', line=dict(color="#E74C3C", width=2, dash="dot"))
                st.plotly_chart(fig_res, use_container_width=True)

        except Exception as e:
            st.error(f"Error evaluating model: {str(e)}")
    else:
        st.error("⚠️ Model or data not found.")

# --- MODULE 2: CUSTOMER INSIGHTS ---
elif menu == "Customer Insights":
    st.title("🧬 Customer Insights & Overview")
    
    if rfm_df is not None:
        st.subheader("🧬 Customer Segmentation DNA")
        seg_choice = st.selectbox("Select Segment to Analyze:", rfm_df['Segment'].unique())
        subset = rfm_df[rfm_df['Segment'] == seg_choice]
        st.caption(f"Visualizing behavior pattern for **{seg_choice}** segment.")
        
        fig_dna = px.scatter(
            rfm_df.sample(min(3000, len(rfm_df))), 
            x='Recency', y='Monetary', color='Segment', size='Frequency', 
            log_y=True, template="plotly_dark", color_discrete_sequence=px.colors.qualitative.Bold,
            title="Recency vs Monetary Pattern (Size = Frequency)"
        )
        fig_dna.update_layout(height=500, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_dna, use_container_width=True)
        
        with st.expander(f"📄 View Data Table: {seg_choice} (Top 100)", expanded=False):
            st.dataframe(subset.head(100), use_container_width=True)

        st.markdown("---")
        st.subheader("📊 Executive Command Center")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Customers", f"{rfm_df['customer_id'].nunique():,}")
        k2.metric("Total Revenue", f"R$ {rfm_df['Monetary'].sum()/1e6:,.1f}M")
        k3.metric("Avg Ticket", f"R$ {rfm_df['Monetary'].mean():,.0f}")
        k4.metric("Active Segments", f"{rfm_df['Segment'].nunique()}")
        
        c1, c2 = st.columns([1.5, 1])
        with c1:
            st.markdown("**Revenue Contribution by Segment**")
            rev_by_seg = rfm_df.groupby('Segment')['Monetary'].sum().reset_index()
            fig_bar = px.bar(rev_by_seg, x='Segment', y='Monetary', color='Segment', text_auto='.2s', template="plotly_dark", color_discrete_sequence=px.colors.qualitative.Bold)
            fig_bar.update_layout(showlegend=False, height=400, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_bar, use_container_width=True)
        with c2:
            st.markdown("**Customer Distribution**")
            count_by_seg = rfm_df['Segment'].value_counts().reset_index()
            count_by_seg.columns = ['Segment', 'Count']
            fig_pie = px.pie(count_by_seg, values='Count', names='Segment', hole=0.5, template="plotly_dark", color_discrete_sequence=px.colors.qualitative.Bold)
            fig_pie.update_layout(height=400, paper_bgcolor="rgba(0,0,0,0)", showlegend=True, legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(fig_pie, use_container_width=True)

# --- MODULE 3: FUTURE SIGHT (FORECAST) ---
elif menu == "Future Sight (AI Forecast)":
    st.title("🔮 AI Revenue Projection")
    st.markdown("Interactive: **Chart points follow the Slider Horizon. Click a point to inspect specifics.**")
    
    c1, c2 = st.columns(2)
    with c1: 
        horizon = st.slider("Forecast Horizon (Weeks)", 4, 52, 12)
    with c2: 
        view_mode = st.radio("Resolution:", ["Daily", "Weekly", "Monthly"], horizontal=True)

    if model and hist_df is not None:
        valid_hist = hist_df[hist_df['revenue'] > 1000].copy()
        last_4_weeks = valid_hist.tail(4)
        avg_n_orders = last_4_weeks['n_orders'].mean()
        last_valid_date = last_4_weeks['date'].iloc[-1]
        
        with st.spinner('Calculating Projections...'):
            X_raw, future_dates = generate_future_input(last_valid_date, avg_n_orders, horizon)
            X_scaled = scaler_x.transform(X_raw)
            X_reshaped = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
            y_pred = scaler_y.inverse_transform(model.predict(X_reshaped, verbose=0)).flatten()
            y_pred = np.maximum(y_pred, 0)

        total_horizon_val = np.sum(y_pred)
        
        plot_hist = None
        plot_future = None
        plot_mode = 'lines'
        
        if view_mode == "Daily":
            hist_resampled = valid_hist.set_index('date').resample('D').interpolate(method='linear')
            hist_resampled['revenue'] = hist_resampled['revenue'] / 7 
            
            df_temp = pd.DataFrame({'date': future_dates, 'revenue': y_pred}).set_index('date')
            future_resampled = df_temp.resample('D').interpolate(method='linear')
            future_resampled['revenue'] = future_resampled['revenue'] / 7
            
            noise = np.random.normal(0, future_resampled['revenue'] * 0.1, len(future_resampled))
            future_resampled['revenue'] += noise
            is_weekend = future_resampled.index.dayofweek >= 5
            future_resampled.loc[is_weekend, 'revenue'] *= 0.65
            
            plot_hist = hist_resampled.tail(90)
            plot_future = future_resampled
            plot_mode = 'lines'

        elif view_mode == "Weekly":
            hist_resampled = valid_hist.set_index('date').resample('W-MON').sum()
            df_temp = pd.DataFrame({'date': future_dates, 'revenue': y_pred}).set_index('date')
            future_resampled = df_temp.resample('W-MON').sum()
            plot_hist = hist_resampled.tail(24)
            plot_future = future_resampled
            plot_mode = 'lines+markers'

        else: # Monthly
            hist_resampled = valid_hist.set_index('date').resample('ME').sum()
            df_temp = pd.DataFrame({'date': future_dates, 'revenue': y_pred}).set_index('date')
            future_resampled = df_temp.resample('ME').sum()
            plot_hist = hist_resampled.tail(12)
            plot_future = future_resampled
            plot_mode = 'lines+markers'

        # Loop logic
        prev_vals = []
        prev_dates = []
        changes = []
        full_hist = hist_resampled.sort_index()
        
        for idx, row in plot_future.iterrows():
            curr_date = idx
            tgt_date = curr_date - pd.DateOffset(months=1)
            try:
                n_idx = full_hist.index.get_indexer([tgt_date], method='nearest')[0]
                fd = full_hist.index[n_idx]
                if abs((fd - tgt_date).days) > 5: fv = 0
                else: fv = full_hist.iloc[n_idx]['revenue']
            except: fv = 0
            
            if fv <= 1: pct = 0.0
            else: pct = (row['revenue'] - fv) / fv
            
            prev_vals.append(fv)
            prev_dates.append(fd)
            changes.append(pct)
            
        plot_future['prev_revenue'] = prev_vals
        plot_future['prev_date_ref'] = prev_dates
        plot_future['mom_change'] = changes

        chart_key = f"chart_{horizon}_{view_mode}"

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=plot_hist.index, y=plot_hist['revenue'], mode='lines', name='History', line=dict(color='#A0AEC0', width=2), hoverinfo='x+y'))
        fig.add_trace(go.Scatter(x=plot_future.index, y=plot_future['revenue'], mode=plot_mode, name='AI Forecast', line=dict(color='#00CC96', width=3), marker=dict(size=8, opacity=0.7),
            customdata=np.stack((plot_future['prev_revenue'], plot_future['mom_change'], plot_future['prev_date_ref'].astype(str)), axis=-1),
            hovertemplate="<b>Date:</b> %{x}<br><b>Forecast:</b> R$ %{y:,.0f}<br><b>Vs Last Month:</b> %{customdata[1]:.2%}<extra></extra>"
        ))
        
        try:
            fig.add_trace(go.Scatter(x=[plot_hist.index[-1], plot_future.index[0]], y=[plot_hist['revenue'].iloc[-1], plot_future['revenue'].iloc[0]], mode='lines', showlegend=False, line=dict(color='#00CC96', width=2, dash='dot'), hoverinfo='skip'))
        except: pass

        fig.update_layout(title=f"{view_mode} Forecast", xaxis_title="Timeline", yaxis_title="Revenue (R$)", template="plotly_dark", height=500, hovermode="closest", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", clickmode='event+select')

        selection = st.plotly_chart(fig, use_container_width=True, on_select="rerun", selection_mode="points", key=chart_key)

        if selection and len(selection.selection.points) > 0:
            point = selection.selection.points[0]
            if "x" in point:
                clicked_dt = pd.to_datetime(point["x"])
                if clicked_dt in plot_future.index: selected_date = clicked_dt
                else: selected_date = plot_future.index[-1]
        else:
            selected_date = plot_future.index[-1]

        selected_val = plot_future.loc[selected_date]['revenue']
        change_pct = plot_future.loc[selected_date]['mom_change']
        prev_lbl = plot_future.loc[selected_date]['prev_date_ref'].strftime('%d %b %Y')
        
        strategy, risk_lvl, risk_col, desc, target_list = analyze_strategy(change_pct)
        date_lbl = selected_date.strftime('%d %b %Y')

        st.markdown(f"### 🎯 Focus: {date_lbl}")
        m1, m2 = st.columns([1, 2])
        with m1: st.metric("Projected Revenue", f"R$ {selected_val:,.0f}", f"{change_pct:.2%} vs {prev_lbl}")
        with m2: st.markdown(f"<div class='metric-card' style='border-left: 5px solid {risk_col};'><h3 style='margin:0; color:{risk_col};'>{strategy}</h3><p style='color:white; margin-top:5px;'>{desc}</p></div>", unsafe_allow_html=True)

        st.subheader(f"🚀 Action Plan for {date_lbl}")
        if rfm_df is not None:
            for i, seg in enumerate(target_list):
                matched_seg = next((s for s in rfm_df['Segment'].unique() if seg.lower() in s.lower()), None)
                if not matched_seg: continue
                with st.expander(f"📂 Open Segment: {matched_seg}", expanded=True):
                    target_list_df = rfm_df[rfm_df['Segment'] == matched_seg]
                    if not target_list_df.empty:
                        csv = target_list_df[['customer_id']].to_csv(index=False).encode('utf-8')
                        btn_key = f"btn_{matched_seg}_{selected_date.strftime('%Y%m%d')}_{i}"
                        st.download_button(label=f"📥 Export List", data=csv, file_name=f"TARGET_{matched_seg}.csv", mime='text/csv', type='primary', key=btn_key)
                    else: st.info(f"No users in '{matched_seg}'.")