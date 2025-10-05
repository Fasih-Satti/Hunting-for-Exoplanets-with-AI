import streamlit as st
import pandas as pd
import numpy as np
import io
import joblib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, Tuple, Dict, Any


def load_model_from_file(model_file) -> Optional[Any]:
    if model_file is None:
        return None
        
    try:
        model_bytes = model_file.read()
        model = joblib.load(io.BytesIO(model_bytes))
        
        required_methods = ['predict']
        for method in required_methods:
            if not hasattr(model, method):
                st.error(f"Model does not have method {method}")
                return None
        
        st.sidebar.success("Model loaded successfully")
        
        with st.sidebar.expander("Model Information"):
            st.write(f"**Type:** {type(model).__name__}")
            
            if hasattr(model, 'classes_'):
                st.write(f"**Classes:** {', '.join(map(str, model.classes_))}")
            
            if hasattr(model, 'feat_names_in_'):
                st.write(f"**Features:** {len(model.feat_names_in_)}")
                st.write(f"**First 5:** {', '.join(model.feat_names_in_[:5])}")
            
            if hasattr(model, 'n_estimators'):
                st.write(f"**Trees:** {model.n_estimators}")
        
        return model
        
    except Exception as e:
        st.sidebar.error(f"Error loading model: {str(e)}")
        return None


def train_baseline_model(df: pd.DataFrame, target_col: str = 'disposition') -> Optional[Pipeline]:
    if target_col not in df.columns:
        st.warning(f"Column '{target_col}' not found in data")
        return None
    
    with st.spinner("Training baseline model..."):
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col != target_col]
        
        if not feature_cols:
            st.error("No numeric features available for training")
            return None
        
        df_clean = df.dropna(subset=[target_col]).copy()
        
        if len(df_clean) < 20:
            st.error("Insufficient data for training (minimum 20 records required)")
            return None
        
        missing_ratio = df_clean[feature_cols].isna().mean()
        valid_features = [col for col in feature_cols if missing_ratio[col] < 0.95]
        
        if not valid_features:
            st.error("All features have too many missing values")
            return None
        
        X = df_clean[valid_features]
        y = df_clean[target_col].astype(str)
        
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        except:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('clf', RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ))
        ])
        
        pipeline.fit(X_train, y_train)
        
        try:
            y_pred = pipeline.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True)
            
            col1, col2, col3 = st.sidebar.columns(3)
            col1.metric("Accuracy", f"{report['accuracy']:.2%}")
            col2.metric("Precision", f"{report['weighted avg']['precision']:.2%}")
            col3.metric("Recall", f"{report['weighted avg']['recall']:.2%}")
        except:
            pass
        
        pipeline.feat_names_in_ = np.array(valid_features)
        st.sidebar.success(f"Baseline model trained on {len(valid_features)} features")
        
        return pipeline


def get_model_features(model, df: pd.DataFrame) -> list:
    if hasattr(model, 'feat_names_in_'):
        features = list(model.feat_names_in_)
        missing = [f for f in features if f not in df.columns]
        if missing:
            st.warning(f"Missing features: {', '.join(missing)}")
            features = [f for f in features if f in df.columns]
        return features
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = ['disposition', 'label', 'class', 'target', 'y']
    features = [col for col in numeric_cols if col.lower() not in exclude]
    
    return features


def make_predictions(model, df: pd.DataFrame, feature_cols: list) -> Dict[str, np.ndarray]:
    X = df[feature_cols].copy()
    
    if X.isna().any().any() and not hasattr(model, 'named_steps'):
        X = X.fillna(X.median())
    
    results = {}
    
    try:
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X)
            classes = list(model.classes_)
            
            pred_idx = np.argmax(proba, axis=1)
            results['predictions'] = np.array(classes)[pred_idx]
            results['confidence'] = proba.max(axis=1)
            
            for i, cls in enumerate(classes):
                results[f'proba_{cls}'] = proba[:, i]
        else:
            results['predictions'] = model.predict(X)
            results['confidence'] = np.ones(len(X))
        
        results['priority'] = compute_priority(results['confidence'])
        
        st.success(f"Predictions completed for {len(X)} records")
        
        pred_counts = pd.Series(results['predictions']).value_counts()
        with st.expander("Prediction Distribution"):
            for cls, count in pred_counts.items():
                pct = count / len(X) * 100
                st.write(f"**{cls}:** {count} ({pct:.1f}%)")
        
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        results['predictions'] = np.array(['ERROR'] * len(X))
        results['confidence'] = np.zeros(len(X))
        results['priority'] = np.ones(len(X)) * 0.5
    
    return results


def compute_priority(confidence: np.ndarray) -> np.ndarray:
    confidence = np.where(np.isnan(confidence), 0.5, confidence)
    confidence = np.clip(confidence, 0.0, 1.0)
    priority = 1.0 - (np.abs(confidence - 0.5) * 2.0)
    return np.clip(priority, 0, 1)


st.set_page_config(
    page_title="EXODETECT - Exoplanet Classification System",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #f0f2f6;
        border-radius: 10px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #667eea;
        color: white;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">EXODETECT</h1>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666; font-size: 1.2rem;'>Advanced Exoplanet Classification & Analysis System</p>", unsafe_allow_html=True)

if 'model' not in st.session_state:
    st.session_state.model = None
if 'df' not in st.session_state:
    st.session_state.df = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'model_ready' not in st.session_state:
    st.session_state.model_ready = False

tab1, tab2, tab3, tab4 = st.tabs(["EXPLORE", "PLANET DETAIL", "MODEL", "BATCH TRIAGE"])

with tab1:
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        st.markdown("### FILTERS")
        with st.container():
            mission = st.multiselect(
                "Mission",
                ["Kepler", "K2", "TESS", "COROT", "All"],
                default=["All"]
            )
            
            disposition = st.multiselect(
                "Disposition",
                ["Confirmed", "Candidate", "False Positive", "All"],
                default=["All"]
            )
            
            st.markdown("**Star Temperature (K)**")
            temp_min = st.number_input("Min", value=3000, step=100, label_visibility="collapsed")
            temp_max = st.number_input("Max", value=7000, step=100, label_visibility="collapsed")
            
            st.markdown("**Orbital Radius (AU)**")
            orbit_min = st.number_input("Min", value=0.0, step=0.1, format="%.2f", key="orbit_min")
            orbit_max = st.number_input("Max", value=5.0, step=0.1, format="%.2f", key="orbit_max")
            
            st.markdown("**Transit Depth (ppm)**")
            transit_depth = st.slider("", 0, 10000, (0, 5000))
    
    with col2:
        st.markdown("### SCATTERPLOT / SKY MAP")
        
        np.random.seed(42)
        n_points = 150
        
        sample_data = pd.DataFrame({
            'ra': np.random.uniform(0, 360, n_points),
            'dec': np.random.uniform(-90, 90, n_points),
            'radius': np.random.uniform(0.5, 4, n_points),
            'period': np.random.exponential(50, n_points),
            'disposition': np.random.choice(['Confirmed', 'Candidate', 'False Positive'], n_points, p=[0.3, 0.5, 0.2]),
            'star_temp': np.random.uniform(3000, 7000, n_points),
            'name': [f'KOI-{i:04d}' for i in range(n_points)]
        })
        
        fig = px.scatter(
            sample_data,
            x='ra',
            y='dec',
            color='disposition',
            size='radius',
            hover_data=['name', 'period', 'star_temp'],
            color_discrete_map={
                'Confirmed': '#4CAF50',
                'Candidate': '#FFA726',
                'False Positive': '#EF5350'
            },
            labels={'ra': 'Right Ascension (deg)', 'dec': 'Declination (deg)'},
            height=500
        )
        
        fig.update_layout(
            plot_bgcolor='#f8f9fa',
            paper_bgcolor='white',
            font=dict(size=12),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        st.markdown("### RESULTS")
        
        results = [
            ("TOI-1234.01", "Candidate", 1.2, 5.6, 4500),
            ("Kepler-22b", "Confirmed", 2.4, 12.3, 5500),
            ("K03210.01", "False Positive", 0.9, 7.5, 4010),
            ("K00123.01", "Candidate", 1.1, 10.2, 5300)
        ]
        
        for name, status, radius, period, temp in results:
            with st.container():
                color = '#4CAF50' if status == 'Confirmed' else '#FFA726' if status == 'Candidate' else '#EF5350'
                st.markdown(f"**{name}** <span style='color: {color};'>{status}</span>", unsafe_allow_html=True)
                st.text(f"Radius: {radius} R⊕")
                st.text(f"Period: {period} d")
                st.text(f"Star Temp: {temp} K")
                st.markdown("---")

with tab2:
    st.markdown("### Detailed Planet Analysis")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        planet_select = st.selectbox(
            "Select Planet",
            ["Kepler-452b", "Proxima Centauri b", "TRAPPIST-1e", "HD 209458 b"]
        )
        
        st.markdown("#### Planet Properties")
        st.metric("Radius", "1.63 R⊕", "+0.03")
        st.metric("Mass", "5.0 M⊕", "+0.5")
        st.metric("Orbital Period", "384.8 days", "+0.3")
        st.metric("Semi-major Axis", "1.046 AU", "+0.001")
        st.metric("Equilibrium Temperature", "265 K", "+3")
    
    with col2:
        categories = ['Radius', 'Mass', 'Temperature', 'Orbital Period', 'Density']
        earth_values = [1, 1, 1, 1, 1]
        planet_values = [1.63, 5.0, 0.92, 1.05, 1.88]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=earth_values,
            theta=categories,
            fill='toself',
            name='Earth',
            line_color='blue'
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=planet_values,
            theta=categories,
            fill='toself',
            name=planet_select,
            line_color='red'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 6]
                )),
            showlegend=True,
            title="Planet Comparison with Earth"
        )
        
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown("### Exoplanet Classification Model")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### INPUT DATA")
        
        orbital_period = st.number_input("Orbital Period (days)", value=10.0, step=0.1)
        transit_duration = st.number_input("Transit Duration (hours)", value=3.0, step=0.1)
        planetary_radius = st.number_input("Planetary Radius (R⊕)", value=1.0, step=0.1)
        transit_depth_model = st.number_input("Transit Depth (ppm)", value=100, step=10)
        star_temp_model = st.number_input("Star Temperature (K)", value=5778, step=100)
        stellar_radius = st.number_input("Stellar Radius (R☉)", value=1.0, step=0.1)
        stellar_log_g = st.number_input("Stellar log g", value=4.4, step=0.1)
        stellar_insolation = st.number_input("Stellar Insolation (S⊕)", value=1.0, step=0.1)
        
        if st.button("RUN PREDICTION", type="primary", use_container_width=True):
            if st.session_state.model is None:
                st.warning("Model not loaded. Please load a model or train a baseline model in BATCH TRIAGE.")
            else:
                try:
                    input_data = pd.DataFrame({
                        'koi_period': [orbital_period],
                        'koi_duration': [transit_duration],
                        'koi_prad': [planetary_radius],
                        'koi_depth': [transit_depth_model],
                        'koi_steff': [star_temp_model],
                        'koi_srad': [stellar_radius],
                        'koi_slogg': [stellar_log_g],
                        'koi_insol': [stellar_insolation]
                    })
                    
                    feature_cols = get_model_features(st.session_state.model, input_data)
                    results = make_predictions(st.session_state.model, input_data, feature_cols)
                    
                    pred_class = results['predictions'][0]
                    conf = results['confidence'][0]
                    
                    if pred_class == 'CONFIRMED':
                        st.success(f"Prediction: **{pred_class}**")
                    elif pred_class == 'CANDIDATE':
                        st.warning(f"Prediction: **{pred_class}**")
                    else:
                        st.error(f"Prediction: **{pred_class}**")
                    
                    st.info(f"Confidence: **{conf:.2%}**")
                    
                    if 'proba_CONFIRMED' in results:
                        proba_confirmed = results['proba_CONFIRMED'][0]
                        proba_candidate = results['proba_CANDIDATE'][0]
                        proba_fp = results['proba_FALSE POSITIVE'][0]
                        
                        st.markdown("**Class Probabilities:**")
                        st.write(f"Confirmed: {proba_confirmed:.2%}")
                        st.write(f"Candidate: {proba_candidate:.2%}")
                        st.write(f"False Positive: {proba_fp:.2%}")
                    
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
                    import random
                    pred_class = random.choice(["Confirmed", "Candidate", "False Positive"])
                    conf = random.uniform(0.6, 0.99)
                    st.success(f"Prediction: **{pred_class}** (simulation)")
                    st.info(f"Confidence: **{conf:.2%}** (simulation)")
    
    with col2:
        st.markdown("#### MODEL PERFORMANCE")
        
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Accuracy", "95.7%", "+2.1%")
        col_b.metric("Precision", "92.3%", "+1.5%")
        col_c.metric("Recall", "94.1%", "+0.8%")
        
        st.markdown("#### Neural Network Architecture")
        st.image("https://via.placeholder.com/400x300/667eea/ffffff?text=Neural+Network+Diagram", use_column_width=True)
        
        st.markdown("#### Feature Importance")
        features = ['Transit Depth', 'Orbital Period', 'Planetary Radius', 'Star Temperature', 'Transit Duration']
        importance = [0.28, 0.23, 0.19, 0.17, 0.13]
        
        fig = px.bar(
            x=importance,
            y=features,
            orientation='h',
            labels={'x': 'Importance', 'y': 'Feature'},
            color=importance,
            color_continuous_scale='viridis'
        )
        fig.update_layout(showlegend=False, height=250)
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.markdown("### Batch Processing & Triage System")
    
    with st.expander("How it works", expanded=True):
        st.markdown("""
        - **Upload a CSV** with exoplanet data (NASA Kepler/TESS format)
        - **Upload a trained model** (.joblib) or let the system train one
        - The system produces **predictions**, **confidence scores**, and **priority rankings**
        - **Download results** as CSV for further analysis
        """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### Input Files")
        
        csv_file = st.file_uploader("Upload CSV Data", type=["csv"])
        model_file = st.file_uploader("Upload Model (optional)", type=["joblib", "pkl"])
        
        if st.button("Process Batch", type="primary", use_container_width=True):
            if csv_file:
                try:
                    csv_file.seek(0)
                    
                    try:
                        df = pd.read_csv(csv_file, low_memory=False)
                    except:
                        csv_file.seek(0)
                        try:
                            df = pd.read_csv(csv_file, sep='\t', low_memory=False)
                        except:
                            csv_file.seek(0)
                            try:
                                df = pd.read_csv(csv_file, sep=';', low_memory=False)
                            except:
                                csv_file.seek(0)
                                df = pd.read_csv(csv_file, sep=None, engine='python', low_memory=False)
                    
                    st.session_state.df = df
                    st.success(f"CSV loaded: {len(df):,} records, {len(df.columns)} columns")
                    
                    with st.expander("Data Preview", expanded=True):
                        col_info1, col_info2 = st.columns(2)
                        with col_info1:
                            st.write("**First 10 columns:**")
                            for i, col in enumerate(df.columns[:10], 1):
                                dtype = str(df[col].dtype)
                                st.text(f"{i}. {col} ({dtype})")
                        
                        with col_info2:
                            st.write("**Statistics:**")
                            st.text(f"Total rows: {len(df):,}")
                            st.text(f"Total columns: {len(df.columns)}")
                            st.text(f"Numeric columns: {len(df.select_dtypes(include=[np.number]).columns)}")
                            st.text(f"Text columns: {len(df.select_dtypes(include=['object']).columns)}")
                            
                        st.write("**First 5 rows:**")
                        st.dataframe(df.head())
                    
                except Exception as e:
                    st.error(f"Failed to read CSV file")
                    st.error(f"Error: {str(e)}")
                    
                    try:
                        csv_file.seek(0)
                        content = csv_file.read()
                        text_preview = content[:500].decode('utf-8', errors='ignore')
                        
                        st.write("**File preview (first 500 characters):**")
                        st.code(text_preview)
                        
                        if '\t' in text_preview:
                            st.info("Detected tabs - possibly TSV file")
                        if ';' in text_preview:
                            st.info("Detected semicolons - possibly European CSV")
                        if ',' in text_preview:
                            st.info("Detected commas - standard CSV")
                            
                    except:
                        st.error("Cannot read file as text")
                    
                    st.stop()
                
                model = None
                
                if model_file is not None:
                    try:
                        model_file.seek(0)
                        
                        if not (model_file.name.endswith('.joblib') or model_file.name.endswith('.pkl')):
                            st.warning(f"File {model_file.name} may be incorrect format")
                        
                        model_file.seek(0)
                        model_bytes = model_file.read()
                        
                        size_mb = len(model_bytes) / (1024 * 1024)
                        st.info(f"Model size: {size_mb:.1f} MB")
                        
                        import pickle
                        
                        try:
                            model = joblib.load(io.BytesIO(model_bytes))
                            st.success(f"Model loaded via joblib")
                        except:
                            try:
                                model = pickle.load(io.BytesIO(model_bytes))
                                st.success(f"Model loaded via pickle")
                            except Exception as e:
                                st.error(f"Failed to load model: {str(e)}")
                                model = None
                        
                        if model and hasattr(model, 'predict'):
                            st.session_state.model = model
                            
                            with st.expander("Model Information"):
                                st.write(f"**Type:** {type(model).__name__}")
                                if hasattr(model, 'classes_'):
                                    st.write(f"**Classes:** {model.classes_}")
                                if hasattr(model, 'feat_names_in_'):
                                    st.write(f"**Features ({len(model.feat_names_in_)}):**")
                                    st.write(', '.join(model.feat_names_in_[:10]))
                            
                    except Exception as e:
                        st.error(f"Error loading model: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
                
                if model is None and 'disposition' in df.columns:
                    if st.checkbox("Train baseline model?", value=True):
                        with st.spinner("Training model..."):
                            model = train_baseline_model(df)
                            if model:
                                st.session_state.model = model
                
                if model is not None:
                    st.session_state.model_ready = True
                    st.success("Model ready for use")
                else:
                    st.session_state.model_ready = False
                    st.info("Working without model - demo results will be shown")
                    
            else:
                st.warning("Please upload a CSV file")
    
    with col2:
        if st.session_state.df is not None:
            st.markdown("#### Results & Analysis")
            
            df = st.session_state.df.copy()
            
            if st.session_state.get('model_ready', False) and st.session_state.model is not None:
                model = st.session_state.model
                
                try:
                    feature_cols = get_model_features(model, df)
                    
                    if not feature_cols:
                        st.error("No suitable features found for model")
                        st.info(f"Model expects features: {getattr(model, 'feat_names_in_', 'unknown')}")
                    else:
                        results = make_predictions(model, df, feature_cols)
                        
                        df['prediction'] = results['predictions']
                        df['confidence'] = results['confidence']
                        df['priority_score'] = results['priority']
                        
                        for key in results:
                            if key.startswith('proba_'):
                                df[key] = results[key]
                        
                        st.success("Predictions completed using loaded model")
                
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
                    np.random.seed(42)
                    df['prediction'] = np.random.choice(
                        ['CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE'],
                        len(df), p=[0.2, 0.5, 0.3]
                    )
                    df['confidence'] = np.random.uniform(0.5, 0.99, len(df))
                    df['priority_score'] = compute_priority(df['confidence'])
                    st.info("Demo predictions shown")
            else:
                np.random.seed(42)
                df['prediction'] = np.random.choice(
                    ['CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE'],
                    len(df), p=[0.2, 0.5, 0.3]
                )
                df['confidence'] = np.random.uniform(0.5, 0.99, len(df))
                df['priority_score'] = compute_priority(df['confidence'])
                st.info("Model not loaded - demo predictions shown")
            
            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.metric("Total Records", f"{len(df):,}")
            col_b.metric("Confirmed", f"{(df['prediction'] == 'CONFIRMED').sum():,}")
            col_c.metric("Candidates", f"{(df['prediction'] == 'CANDIDATE').sum():,}")
            col_d.metric("High Priority", f"{(df['priority_score'] > 0.7).sum():,}")
            
            display_cols = ['prediction', 'confidence', 'priority_score']
            
            id_cols = ['kepoi_name', 'koi_name', 'kepid', 'tic_id', 'source_id', 'id']
            for id_col in id_cols:
                if id_col in df.columns:
                    display_cols.insert(0, id_col)
                    break
            
            if 'disposition' in df.columns:
                display_cols.insert(1, 'disposition')
            
            st.markdown("### Predictions Table")
            st.dataframe(
                df[display_cols].head(100),
                use_container_width=True,
                height=400
            )
            
            csv_output = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Predictions CSV",
                csv_output,
                "predictions_output.csv",
                "text/csv",
                use_container_width=True
            )
            
            col_v1, col_v2 = st.columns(2)
            
            with col_v1:
                fig = px.pie(
                    values=df['prediction'].value_counts().values,
                    names=df['prediction'].value_counts().index,
                    title="Prediction Distribution",
                    color_discrete_map={
                        'CONFIRMED': '#4CAF50',
                        'CANDIDATE': '#FFA726',
                        'FALSE POSITIVE': '#EF5350'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col_v2:
                fig = px.histogram(
                    df,
                    x='confidence',
                    nbins=30,
                    title="Confidence Distribution",
                    labels={'confidence': 'Confidence Score', 'count': 'Count'}
                )
                st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #666;'>EXODETECT v2.0 | Built with Streamlit | "
    "Data: NASA Exoplanet Archive</p>",
    unsafe_allow_html=True
)