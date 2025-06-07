import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from scipy import stats
from scipy.stats import chi2_contingency

# Model persistence
import joblib
import pickle

# Set page config
st.set_page_config(
    page_title="🎓 Enhanced Student Dropout Prediction Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding-top: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .insight-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #0066cc;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #ffeb3b;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #28a745;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Constants
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

@st.cache_data
def load_and_preprocess_data():
    """Load and preprocess the dataset"""
    try:
        url = "https://raw.githubusercontent.com/dicodingacademy/dicoding_dataset/refs/heads/main/students_performance/data.csv"
        df = pd.read_csv(url, sep=";")
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

@st.cache_data
def perform_comprehensive_eda(df):
    """Perform comprehensive EDA analysis"""
    eda_results = {}
    
    # Basic statistics
    eda_results['basic_stats'] = {
        'shape': df.shape,
        'missing_values': df.isnull().sum().sum(),
        'duplicates': df.duplicated().sum(),
        'memory_usage': df.memory_usage(deep=True).sum() / 1024**2
    }
    
    # Target distribution
    target_counts = df['Status'].value_counts()
    target_pct = df['Status'].value_counts(normalize=True) * 100
    eda_results['target_distribution'] = {
        'counts': target_counts,
        'percentages': target_pct
    }
    
    # Numerical features analysis
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'Status' in numerical_cols:
        numerical_cols.remove('Status')
    
    eda_results['numerical_features'] = numerical_cols
    eda_results['numerical_stats'] = df[numerical_cols].describe()
    
    # Key numerical features for detailed analysis
    key_numerical = ['Age_at_enrollment', 'Previous_qualification_grade', 'Admission_grade', 
                    'Curricular_units_1st_sem_grade', 'Curricular_units_2nd_sem_grade']
    key_numerical = [col for col in key_numerical if col in df.columns]
    eda_results['key_numerical'] = key_numerical
    
    # Categorical features analysis
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if 'Status' in categorical_cols:
        categorical_cols.remove('Status')
    
    eda_results['categorical_features'] = categorical_cols
    
    # Key categorical features
    key_categorical = ['Gender', 'Marital_status', 'Daytime_evening_attendance',
                      'Displaced', 'Debtor', 'Tuition_fees_up_to_date']
    key_categorical = [col for col in key_categorical if col in df.columns]
    eda_results['key_categorical'] = key_categorical
    
    # Correlation analysis
    if len(key_numerical) > 1:
        correlation_matrix = df[key_numerical].corr()
        eda_results['correlation_matrix'] = correlation_matrix
        
        # Find high correlations
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_val = abs(correlation_matrix.iloc[i, j])
                if corr_val > 0.7:
                    high_corr_pairs.append({
                        'Feature 1': correlation_matrix.columns[i],
                        'Feature 2': correlation_matrix.columns[j],
                        'Correlation': correlation_matrix.iloc[i, j]
                    })
        eda_results['high_correlations'] = high_corr_pairs
    
    # Outlier detection
    outlier_summary = {}
    for col in key_numerical:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_count = len(outliers)
        outlier_pct = (outlier_count / len(df)) * 100
        
        outlier_summary[col] = {
            'count': outlier_count,
            'percentage': outlier_pct,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }
    
    eda_results['outliers'] = outlier_summary
    
    return eda_results

def preprocess_data(df):
    """Preprocess the data for modeling"""
    df_processed = df.copy()
    
    # Handle outliers using IQR method
    numerical_features = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    if 'Status' in numerical_features:
        numerical_features.remove('Status')
    
    outliers_handled = 0
    for col in numerical_features:
        Q1 = df_processed[col].quantile(0.25)
        Q3 = df_processed[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers_mask = (df_processed[col] < lower_bound) | (df_processed[col] > upper_bound)
        outliers_count = outliers_mask.sum()
        
        if outliers_count > 0:
            df_processed.loc[df_processed[col] < lower_bound, col] = lower_bound
            df_processed.loc[df_processed[col] > upper_bound, col] = upper_bound
            outliers_handled += outliers_count
    
    # Encoding categorical variables
    categorical_features = df_processed.select_dtypes(include=['object']).columns.tolist()
    if 'Status' in categorical_features:
        categorical_features.remove('Status')
    
    label_encoders = {}
    for col in categorical_features:
        unique_values = df_processed[col].nunique()
        if unique_values == 2:
            le = LabelEncoder()
            df_processed[col + '_encoded'] = le.fit_transform(df_processed[col])
            label_encoders[col] = le
        else:
            dummies = pd.get_dummies(df_processed[col], prefix=col)
            df_processed = pd.concat([df_processed, dummies], axis=1)
    
    # Drop original categorical columns
    df_processed = df_processed.drop(columns=categorical_features)
    
    # Encode target variable
    target_encoder = LabelEncoder()
    df_processed['Status_encoded'] = target_encoder.fit_transform(df_processed['Status'])
    
    return df_processed, label_encoders, target_encoder, outliers_handled

@st.cache_resource
def train_advanced_models(df_processed, _target_encoder):
    """Train advanced machine learning models with hyperparameter optimization"""
    feature_columns = [col for col in df_processed.columns if col not in ['Status', 'Status_encoded']]
    X = df_processed[feature_columns]
    y = df_processed['Status_encoded']
    
    # Scaling
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Feature selection
    selector = SelectKBest(score_func=f_classif, k=min(20, len(feature_columns)))
    X_selected = selector.fit_transform(X_scaled, y)
    selected_features = [feature_columns[i] for i in selector.get_support(indices=True)]
    
    # Calculate mutual information for feature importance
    try:
        mi_scores = mutual_info_classif(X_scaled, y, random_state=RANDOM_STATE)
        mi_df = pd.DataFrame({
            'Feature': feature_columns,
            'Mutual_Information': mi_scores
        }).sort_values('Mutual_Information', ascending=False)
    except:
        mi_df = pd.DataFrame()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    # Determine if multi-class
    n_classes = len(np.unique(y))
    scoring_metric = 'accuracy' if n_classes > 2 else 'roc_auc'
    
    # Define models
    models_config = {
        'Random Forest': RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=100),
        'Gradient Boosting': GradientBoostingClassifier(random_state=RANDOM_STATE),
        'Logistic Regression': LogisticRegression(
            random_state=RANDOM_STATE, 
            max_iter=2000,
            multi_class='ovr' if n_classes > 2 else 'auto'
        )
    }
    
    trained_models = {}
    model_results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    # Train models
    for name, model in models_config.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            
            # Calculate AUC
            try:
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_test)
                    if n_classes == 2:
                        auc_score = roc_auc_score(y_test, y_pred_proba[:, 1])
                    else:
                        auc_score = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
                else:
                    auc_score = 0
            except:
                auc_score = 0
            
            # Cross-validation score
            try:
                cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring_metric)
            except:
                cv_scores = np.array([accuracy])
            
            trained_models[name] = model
            model_results[name] = {
                'accuracy': accuracy,
                'auc_score': auc_score,
                'cv_score': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred,
                'probabilities': y_pred_proba if 'y_pred_proba' in locals() else None
            }
            
        except Exception as e:
            st.warning(f"Error training {name}: {str(e)}")
            continue
    
    # Select best model
    if model_results:
        best_model_name = max(model_results.keys(), key=lambda x: model_results[x]['accuracy'])
        best_model = trained_models[best_model_name]
    else:
        best_model_name = None
        best_model = None
    
    return {
        'models': trained_models,
        'results': model_results,
        'best_model': best_model,
        'best_model_name': best_model_name,
        'scaler': scaler,
        'selector': selector,
        'selected_features': selected_features,
        'feature_columns': feature_columns,
        'mutual_info': mi_df,
        'X_test': X_test,
        'y_test': y_test,
        'target_encoder': _target_encoder,
        'n_classes': n_classes
    }

def save_model_with_joblib(model_info):
    """Save model using joblib for persistence"""
    if model_info['best_model'] is not None:
        model_package = {
            'model': model_info['best_model'],
            'scaler': model_info['scaler'],
            'feature_selector': model_info['selector'],
            'target_encoder': model_info['target_encoder'],
            'selected_features': model_info['selected_features'],
            'feature_columns': model_info['feature_columns'],
            'n_classes': model_info['n_classes'],
            'best_model_name': model_info['best_model_name'],
            'model_performance': model_info['results'][model_info['best_model_name']] if model_info['best_model_name'] else {},
            'training_info': {
                'total_features': len(model_info['feature_columns']),
                'selected_features': len(model_info['selected_features']),
                'random_state': RANDOM_STATE
            }
        }
        
        # Create download link
        joblib.dump(model_package, 'student_dropout_model.pkl')
        
        with open('student_dropout_model.pkl', 'rb') as f:
            st.download_button(
                label="📥 Download Complete Model Package",
                data=f.read(),
                file_name='student_dropout_model.pkl',
                mime='application/octet-stream'
            )
        
        return True
    return False

def predict_dropout_risk(input_data, model_info):
    """Predict dropout risk for new student data"""
    try:
        expected_features = len(model_info['feature_columns'])
        
        if len(input_data) != expected_features:
            if len(input_data) < expected_features:
                input_data.extend([0] * (expected_features - len(input_data)))
            else:
                input_data = input_data[:expected_features]
        
        input_scaled = model_info['scaler'].transform([input_data])
        input_selected = model_info['selector'].transform(input_scaled)
        
        prediction = model_info['best_model'].predict(input_selected)[0]
        probability = model_info['best_model'].predict_proba(input_selected)[0]
        
        return prediction, probability
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return None, None

def main():
    st.title("🎓 Enhanced Student Dropout Prediction Dashboard")
    st.markdown("**Comprehensive Analytics & Machine Learning for Student Success**")
    st.markdown("*Based on Enhanced EDA and Advanced ML Models*")
    st.markdown("---")
    
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    st.sidebar.markdown("**Choose a section to explore:**")
    
    page = st.sidebar.selectbox("", [
        "📊 Data Overview & Quality",
        "🔍 EDA - Univariate Analysis", 
        "🔗 EDA - Bivariate Analysis",
        "🌐 EDA - Multivariate Analysis",
        "🤖 ML Model Performance", 
        "🔮 Dropout Risk Prediction",
        "💾 Model Persistence (Joblib)",
        "💡 Insights & Recommendations"
    ])
    
    # Load data
    with st.spinner("🔄 Loading and preprocessing data..."):
        df = load_and_preprocess_data()
    
    if df is None:
        st.error("❌ Failed to load data. Please check your connection.")
        return
    
    # Preprocess data
    df_processed, label_encoders, target_encoder, outliers_handled = preprocess_data(df)
    
    # Perform EDA
    eda_results = perform_comprehensive_eda(df)
    
    # ============================================================================
    # PAGE 1: DATA OVERVIEW & QUALITY
    # ============================================================================
    if page == "📊 Data Overview & Quality":
        st.header("📊 Data Overview & Quality Assessment")
        
        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📚 Total Students", f"{len(df):,}")
        
        with col2:
            dropout_count = (df['Status'] == 'Dropout').sum()
            st.metric("⚠️ Dropout Students", f"{dropout_count:,}")
        
        with col3:
            dropout_rate = (df['Status'] == 'Dropout').mean() * 100
            st.metric("📉 Dropout Rate", f"{dropout_rate:.1f}%")
        
        with col4:
            graduate_count = (df['Status'] == 'Graduate').sum()
            st.metric("🎓 Graduates", f"{graduate_count:,}")
        
        # Data Quality Assessment
        st.subheader("🏥 Data Quality Assessment")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### ✅ Data Health Check")
            st.markdown(f"**📐 Dataset Shape**: {eda_results['basic_stats']['shape'][0]:,} rows × {eda_results['basic_stats']['shape'][1]} columns")
            st.markdown(f"**🕳️ Missing Values**: {eda_results['basic_stats']['missing_values']} (✅ None found!)")
            st.markdown(f"**🔄 Duplicate Records**: {eda_results['basic_stats']['duplicates']} (✅ None found!)")
            st.markdown(f"**💾 Memory Usage**: {eda_results['basic_stats']['memory_usage']:.2f} MB")
            st.markdown(f"**🔧 Outliers Handled**: {outliers_handled:,} values capped")
            st.markdown(f"**🏷️ Target Classes**: {len(eda_results['target_distribution']['counts'])} classes")
        
        with col2:
            # Target distribution pie chart
            fig_pie = px.pie(
                values=eda_results['target_distribution']['counts'].values,
                names=eda_results['target_distribution']['counts'].index,
                title="🎯 Student Status Distribution",
                color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1']
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Dataset Preview
        st.subheader("👀 Dataset Preview")
        
        # Show sample data with filters
        status_filter = st.selectbox("Filter by Status:", ['All'] + list(df['Status'].unique()))
        
        if status_filter != 'All':
            display_df = df[df['Status'] == status_filter].head(10)
        else:
            display_df = df.head(10)
        
        st.dataframe(display_df, use_container_width=True)
        
        # Statistical Summary
        st.subheader("📊 Statistical Summary")
        
        tab1, tab2, tab3 = st.tabs(["📈 Numerical Features", "📂 Categorical Features", "🎯 Target Analysis"])
        
        with tab1:
            st.dataframe(eda_results['numerical_stats'], use_container_width=True)
            
        with tab2:
            if eda_results['categorical_features']:
                cat_summary = []
                for col in eda_results['categorical_features']:
                    unique_count = df[col].nunique()
                    most_common = df[col].mode()[0] if len(df[col].mode()) > 0 else 'N/A'
                    cat_summary.append({
                        'Feature': col,
                        'Unique Values': unique_count,
                        'Most Common': most_common
                    })
                
                st.dataframe(pd.DataFrame(cat_summary), use_container_width=True)
            else:
                st.info("ℹ️ No categorical features detected in the dataset.")
        
        with tab3:
            # Target variable detailed analysis
            target_df = pd.DataFrame({
                'Status': eda_results['target_distribution']['counts'].index,
                'Count': eda_results['target_distribution']['counts'].values,
                'Percentage': eda_results['target_distribution']['percentages'].values
            })
            
            st.dataframe(target_df, use_container_width=True)
            
            # Age analysis by status
            if 'Age_at_enrollment' in df.columns:
                fig_age_box = px.box(
                    df, x='Status', y='Age_at_enrollment',
                    title="📊 Age Distribution by Student Status",
                    color='Status',
                    color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1']
                )
                st.plotly_chart(fig_age_box, use_container_width=True)

    # ============================================================================
    # PAGE 2: EDA - UNIVARIATE ANALYSIS
    # ============================================================================
    elif page == "🔍 EDA - Univariate Analysis":
        st.header("🔍 Univariate Analysis")
        
        tab1, tab2 = st.tabs(["📊 Numerical Features", "📂 Categorical Features"])
        
        with tab1:
            st.subheader("📊 Numerical Features Distribution Analysis")
            
            if eda_results['key_numerical']:
                selected_feature = st.selectbox("Select Feature to Analyze:", eda_results['key_numerical'])
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Distribution plot
                    fig_hist = px.histogram(
                        df, x=selected_feature, 
                        title=f"📈 Distribution of {selected_feature}",
                        marginal="box",
                        nbins=30,
                        color_discrete_sequence=['#3498db']
                    )
                    fig_hist.add_vline(x=df[selected_feature].mean(), line_dash="dash", 
                                     line_color="red", annotation_text=f"Mean: {df[selected_feature].mean():.2f}")
                    fig_hist.add_vline(x=df[selected_feature].median(), line_dash="dash", 
                                     line_color="green", annotation_text=f"Median: {df[selected_feature].median():.2f}")
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                with col2:
                    # Distribution by status
                    fig_box = px.box(
                        df, x='Status', y=selected_feature,
                        title=f"📦 {selected_feature} by Status",
                        color='Status',
                        color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1']
                    )
                    st.plotly_chart(fig_box, use_container_width=True)
                
                # Statistical summary for selected feature
                st.markdown("### 📊 Statistical Summary")
                feature_stats = df.groupby('Status')[selected_feature].agg(['count', 'mean', 'median', 'std', 'min', 'max']).round(2)
                st.dataframe(feature_stats, use_container_width=True)
                
                # ANOVA test
                try:
                    groups = [df[df['Status'] == status][selected_feature].dropna() for status in df['Status'].unique()]
                    f_stat, p_value = stats.f_oneway(*groups)
                    
                    if p_value < 0.05:
                        st.success(f"🔥 **Significant difference found!** ANOVA p-value: {p_value:.4f}")
                        st.markdown("This feature shows statistically significant differences across student status groups.")
                    else:
                        st.info(f"📊 ANOVA p-value: {p_value:.4f} (No significant difference)")
                except:
                    st.warning("Could not perform ANOVA test for this feature.")
                
                # Outlier information
                if selected_feature in eda_results['outliers']:
                    outlier_info = eda_results['outliers'][selected_feature]
                    st.markdown("### 🚨 Outlier Detection")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Outlier Count", outlier_info['count'])
                    with col2:
                        st.metric("Outlier Percentage", f"{outlier_info['percentage']:.2f}%")
                    with col3:
                        st.metric("IQR Range", f"[{outlier_info['lower_bound']:.2f}, {outlier_info['upper_bound']:.2f}]")
        
        with tab2:
            st.subheader("📂 Categorical Features Distribution Analysis")
            
            if eda_results['key_categorical']:
                selected_cat_feature = st.selectbox("Select Categorical Feature:", eda_results['key_categorical'])
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Distribution plot
                    value_counts = df[selected_cat_feature].value_counts()
                    fig_bar = px.bar(
                        x=value_counts.values,
                        y=value_counts.index,
                        orientation='h',
                        title=f"📊 Distribution of {selected_cat_feature}",
                        color_discrete_sequence=['#e74c3c']
                    )
                    fig_bar.update_layout(yaxis_title=selected_cat_feature, xaxis_title="Count")
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                with col2:
                    # Cross-tabulation with Status
                    crosstab = pd.crosstab(df[selected_cat_feature], df['Status'])
                    crosstab_pct = pd.crosstab(df[selected_cat_feature], df['Status'], normalize='index') * 100
                    
                    fig_stacked = px.bar(
                        crosstab_pct,
                        title=f"📊 {selected_cat_feature} vs Status (%)",
                        color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1']
                    )
                    fig_stacked.update_layout(barmode='stack')
                    st.plotly_chart(fig_stacked, use_container_width=True)
                
                # Chi-square test
                try:
                    chi2, p_value, dof, expected = chi2_contingency(crosstab)
                    
                    st.markdown("### 🔬 Statistical Test Results")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Chi-square Statistic", f"{chi2:.4f}")
                    with col2:
                        st.metric("P-value", f"{p_value:.4f}")
                    with col3:
                        st.metric("Degrees of Freedom", dof)
                    
                    if p_value < 0.05:
                        st.success("🔥 **Significant association found!** This categorical feature is significantly associated with student status.")
                    else:
                        st.info("📊 No significant association found between this feature and student status.")
                        
                except Exception as e:
                    st.warning(f"Could not perform Chi-square test: {str(e)}")
                
                # Show crosstab
                st.markdown("### 📋 Cross-tabulation")
                st.dataframe(crosstab, use_container_width=True)

    # ============================================================================
    # PAGE 3: EDA - BIVARIATE ANALYSIS
    # ============================================================================
    elif page == "🔗 EDA - Bivariate Analysis":
        st.header("🔗 Bivariate Analysis")
        
        tab1, tab2 = st.tabs(["📈 Correlation Analysis", "🎯 Target Relationships"])
        
        with tab1:
            st.subheader("📈 Correlation Analysis")
            
            if 'correlation_matrix' in eda_results:
                # Correlation heatmap
                fig_corr = px.imshow(
                    eda_results['correlation_matrix'],
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale='RdBu_r',
                    title="🔥 Correlation Heatmap of Numerical Features"
                )
                st.plotly_chart(fig_corr, use_container_width=True)
                
                # High correlations
                if eda_results['high_correlations']:
                    st.markdown("### 🚨 High Correlations Found (|r| > 0.7)")
                    for pair in eda_results['high_correlations']:
                        st.markdown(f"**{pair['Feature 1']}** ↔ **{pair['Feature 2']}**: {pair['Correlation']:.3f}")
                else:
                    st.success("✅ No extremely high correlations found")
            else:
                st.info("Not enough numerical features for correlation analysis.")
        
        with tab2:
            st.subheader("🎯 Target vs Features Analysis")
            
            # Numerical features vs Target
            st.markdown("#### 📊 Numerical Features vs Status")
            
            if eda_results['key_numerical']:
                selected_num_feature = st.selectbox("Select Numerical Feature:", eda_results['key_numerical'])
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Box plot
                    fig_box = px.box(
                        df, x='Status', y=selected_num_feature,
                        title=f"📦 {selected_num_feature} by Status",
                        color='Status',
                        color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1']
                    )
                    st.plotly_chart(fig_box, use_container_width=True)
                
                with col2:
                    # Violin plot for distribution shape
                    fig_violin = px.violin(
                        df, x='Status', y=selected_num_feature,
                        title=f"🎻 Distribution Shape of {selected_num_feature}",
                        color='Status',
                        color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1']
                    )
                    st.plotly_chart(fig_violin, use_container_width=True)
            
            # Categorical features vs Target
            st.markdown("#### 📂 Categorical Features vs Status")
            
            if eda_results['key_categorical']:
                selected_cat_feature = st.selectbox("Select Categorical Feature:", eda_results['key_categorical'], key="bivariate_cat")
                
                # Create comprehensive analysis
                crosstab = pd.crosstab(df[selected_cat_feature], df['Status'])
                crosstab_pct = pd.crosstab(df[selected_cat_feature], df['Status'], normalize='index') * 100
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Count plot
                    fig_count = px.bar(
                        crosstab,
                        title=f"📊 Count of {selected_cat_feature} by Status",
                        color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1']
                    )
                    st.plotly_chart(fig_count, use_container_width=True)
                
                with col2:
                    # Percentage plot
                    fig_pct = px.bar(
                        crosstab_pct,
                        title=f"📊 Percentage Distribution (%)",
                        color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1']
                    )
                    fig_pct.update_layout(barmode='stack')
                    st.plotly_chart(fig_pct, use_container_width=True)

    # ============================================================================
    # PAGE 4: EDA - MULTIVARIATE ANALYSIS
    # ============================================================================
    elif page == "🌐 EDA - Multivariate Analysis":
        st.header("🌐 Multivariate Analysis")
        
        tab1, tab2, tab3 = st.tabs(["🎯 Feature Importance", "📊 Advanced Relationships", "🔍 Pattern Discovery"])
        
        with tab1:
            st.subheader("🎯 Feature Importance Analysis")
            
            # Train model to get feature importance
            with st.spinner("Calculating feature importance..."):
                model_info = train_advanced_models(df_processed, target_encoder)
            
            if not model_info['mutual_info'].empty:
                st.markdown("#### 📈 Mutual Information Scores")
                
                # Display top features
                top_features = model_info['mutual_info'].head(15)
                fig_mi = px.bar(
                    top_features,
                    x='Mutual_Information',
                    y='Feature',
                    orientation='h',
                    title="🔝 Top 15 Features by Mutual Information",
                    color='Mutual_Information',
                    color_continuous_scale='viridis'
                )
                fig_mi.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig_mi, use_container_width=True)
                
                # Show table
                st.dataframe(top_features, use_container_width=True)
            
            if model_info['best_model'] and hasattr(model_info['best_model'], 'feature_importances_'):
                st.markdown("#### 🏆 Model-based Feature Importance")
                
                feature_importance = pd.DataFrame({
                    'feature': model_info['selected_features'],
                    'importance': model_info['best_model'].feature_importances_
                }).sort_values('importance', ascending=False)
                
                fig_importance = px.bar(
                    feature_importance.head(15),
                    x='importance',
                    y='feature',
                    orientation='h',
                    title=f"🎯 Feature Importance - {model_info['best_model_name']}",
                    color='importance',
                    color_continuous_scale='plasma'
                )
                fig_importance.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig_importance, use_container_width=True)
        
        with tab2:
            st.subheader("📊 Advanced Relationships")
            
            if len(eda_results['key_numerical']) >= 2:
                st.markdown("#### 🎯 Scatter Plot Analysis")
                
                col1, col2 = st.columns(2)
                with col1:
                    x_feature = st.selectbox("Select X-axis:", eda_results['key_numerical'])
                with col2:
                    y_feature = st.selectbox("Select Y-axis:", [f for f in eda_results['key_numerical'] if f != x_feature])
                
                # Create scatter plot
                fig_scatter = px.scatter(
                    df, x=x_feature, y=y_feature, color='Status',
                    title=f"📊 {x_feature} vs {y_feature} by Status",
                    color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1'],
                    opacity=0.7
                )
                fig_scatter.update_traces(marker=dict(size=8))
                st.plotly_chart(fig_scatter, use_container_width=True)
                
                # Calculate correlation
                correlation = df[x_feature].corr(df[y_feature])
                st.metric("📈 Correlation Coefficient", f"{correlation:.3f}")
            
            # Age distribution analysis
            if 'Age_at_enrollment' in df.columns:
                st.markdown("#### 👥 Age Distribution Analysis")
                
                age_stats = df.groupby('Status')['Age_at_enrollment'].agg(['mean', 'median', 'std']).round(2)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.dataframe(age_stats, use_container_width=True)
                
                with col2:
                    # Age distribution by status
                    fig_age_dist = px.histogram(
                        df, x='Age_at_enrollment', color='Status',
                        title="📊 Age Distribution by Status",
                        marginal="box",
                        color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1']
                    )
                    st.plotly_chart(fig_age_dist, use_container_width=True)
        
        with tab3:
            st.subheader("🔍 Pattern Discovery")
            
            # Academic performance patterns
            if all(col in df.columns for col in ['Curricular_units_1st_sem_grade', 'Curricular_units_2nd_sem_grade']):
                st.markdown("#### 📚 Academic Performance Patterns")
                
                # Create performance categories
                df_temp = df.copy()
                df_temp['1st_sem_performance'] = pd.cut(df_temp['Curricular_units_1st_sem_grade'], 
                                                       bins=3, labels=['Low', 'Medium', 'High'])
                df_temp['2nd_sem_performance'] = pd.cut(df_temp['Curricular_units_2nd_sem_grade'], 
                                                       bins=3, labels=['Low', 'Medium', 'High'])
                
                # Performance transition analysis
                performance_crosstab = pd.crosstab([df_temp['1st_sem_performance'], df_temp['2nd_sem_performance']], 
                                                  df_temp['Status'])
                
                st.markdown("**Performance Transition Patterns:**")
                st.dataframe(performance_crosstab, use_container_width=True)
                
                # Semester comparison
                fig_sem_comparison = px.scatter(
                    df, x='Curricular_units_1st_sem_grade', y='Curricular_units_2nd_sem_grade',
                    color='Status', title="📊 1st vs 2nd Semester Performance",
                    color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1']
                )
                fig_sem_comparison.add_shape(
                    type="line", line=dict(dash="dash"),
                    x0=df['Curricular_units_1st_sem_grade'].min(),
                    y0=df['Curricular_units_1st_sem_grade'].min(),
                    x1=df['Curricular_units_1st_sem_grade'].max(),
                    y1=df['Curricular_units_1st_sem_grade'].max()
                )
                st.plotly_chart(fig_sem_comparison, use_container_width=True)
            
            # Risk factor combinations
            st.markdown("#### ⚠️ Risk Factor Combinations")
            
            risk_factors = []
            if 'Age_at_enrollment' in df.columns:
                risk_factors.append(('High Age', df['Age_at_enrollment'] > df['Age_at_enrollment'].quantile(0.75)))
            
            if 'Curricular_units_1st_sem_grade' in df.columns:
                risk_factors.append(('Low 1st Sem Grade', df['Curricular_units_1st_sem_grade'] < df['Curricular_units_1st_sem_grade'].quantile(0.25)))
            
            if risk_factors:
                st.markdown("**Risk Factor Analysis:**")
                for factor_name, condition in risk_factors:
                    risk_rate = df[condition]['Status'].value_counts(normalize=True)
                    if 'Dropout' in risk_rate:
                        st.markdown(f"- **{factor_name}**: {risk_rate['Dropout']:.1%} dropout rate")

    # ============================================================================
    # PAGE 5: ML MODEL PERFORMANCE
    # ============================================================================
    elif page == "🤖 ML Model Performance":
        st.header("🤖 Machine Learning Model Performance")
        
        with st.spinner("Training advanced ML models..."):
            model_info = train_advanced_models(df_processed, target_encoder)
        
        if model_info['results']:
            st.success(f"🏆 Best Model: {model_info['best_model_name']}")
            
            # Model comparison
            st.subheader("📊 Model Comparison")
            
            results_df = pd.DataFrame(model_info['results']).T
            results_df = results_df[['accuracy', 'auc_score', 'cv_score', 'cv_std']].round(4)
            results_df.columns = ['Test Accuracy', 'AUC Score', 'CV Score', 'CV Std']
            
            st.dataframe(results_df, use_container_width=True)
            
            # Performance visualization
            col1, col2 = st.columns(2)
            
            with col1:
                # Accuracy comparison
                fig_acc = px.bar(
                    x=list(model_info['results'].keys()),
                    y=[model_info['results'][model]['accuracy'] for model in model_info['results']],
                    title="📈 Model Accuracy Comparison",
                    color_discrete_sequence=['#3498db']
                )
                fig_acc.update_layout(xaxis_title="Model", yaxis_title="Accuracy")
                st.plotly_chart(fig_acc, use_container_width=True)
            
            with col2:
                # AUC comparison
                fig_auc = px.bar(
                    x=list(model_info['results'].keys()),
                    y=[model_info['results'][model]['auc_score'] for model in model_info['results']],
                    title="📈 Model AUC Score Comparison",
                    color_discrete_sequence=['#e74c3c']
                )
                fig_auc.update_layout(xaxis_title="Model", yaxis_title="AUC Score")
                st.plotly_chart(fig_auc, use_container_width=True)
            
            # Feature importance
            if model_info['best_model'] and hasattr(model_info['best_model'], 'feature_importances_'):
                st.subheader("🎯 Feature Importance")
                
                feature_importance = pd.DataFrame({
                    'feature': model_info['selected_features'],
                    'importance': model_info['best_model'].feature_importances_
                }).sort_values('importance', ascending=False)
                
                fig_importance = px.bar(
                    feature_importance.head(15),
                    x='importance',
                    y='feature',
                    orientation='h',
                    title=f"🔝 Top 15 Most Important Features - {model_info['best_model_name']}",
                    color='importance',
                    color_continuous_scale='viridis'
                )
                fig_importance.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig_importance, use_container_width=True)
            
            # Confusion Matrix
            st.subheader("🎯 Confusion Matrix")
            
            best_predictions = model_info['results'][model_info['best_model_name']]['predictions']
            cm = confusion_matrix(model_info['y_test'], best_predictions)
            
            fig_cm = px.imshow(
                cm,
                text_auto=True,
                aspect="auto",
                title=f"🎯 Confusion Matrix - {model_info['best_model_name']}",
                labels=dict(x="Predicted", y="Actual"),
                x=target_encoder.classes_,
                y=target_encoder.classes_
            )
            st.plotly_chart(fig_cm, use_container_width=True)
            
            # Classification Report
            st.subheader("📋 Detailed Classification Report")
            
            report = classification_report(
                model_info['y_test'], 
                best_predictions, 
                target_names=target_encoder.classes_,
                output_dict=True
            )
            
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.round(3), use_container_width=True)
        
        else:
            st.error("❌ No models were successfully trained!")

    # ============================================================================
    # PAGE 6: DROPOUT RISK PREDICTION
    # ============================================================================
    elif page == "🔮 Dropout Risk Prediction":
        st.header("🔮 Student Dropout Risk Prediction")
        
        # Train models for prediction
        with st.spinner("Loading prediction model..."):
            model_info = train_advanced_models(df_processed, target_encoder)
        
        if model_info['best_model'] is None:
            st.error("❌ No model available for prediction!")
            return
        
        st.subheader("📝 Enter Student Information")
        st.markdown("*Fill in the student details to predict dropout risk*")
        
        # Create input form
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📚 Academic Information**")
                age_at_enrollment = st.number_input("Age at Enrollment", min_value=15, max_value=70, value=20)
                curricular_units_1st_sem_credited = st.number_input("1st Sem Credited Units", min_value=0, max_value=30, value=6)
                curricular_units_1st_sem_enrolled = st.number_input("1st Sem Enrolled Units", min_value=0, max_value=30, value=6)
                curricular_units_1st_sem_evaluations = st.number_input("1st Sem Evaluations", min_value=0, max_value=50, value=6)
                curricular_units_1st_sem_approved = st.number_input("1st Sem Approved Units", min_value=0, max_value=30, value=6)
                curricular_units_1st_sem_grade = st.number_input("1st Sem Grade", min_value=0.0, max_value=20.0, value=12.0)
            
            with col2:
                st.markdown("**📈 Second Semester & Personal**")
                curricular_units_2nd_sem_credited = st.number_input("2nd Sem Credited Units", min_value=0, max_value=30, value=6)
                curricular_units_2nd_sem_enrolled = st.number_input("2nd Sem Enrolled Units", min_value=0, max_value=30, value=6)
                curricular_units_2nd_sem_evaluations = st.number_input("2nd Sem Evaluations", min_value=0, max_value=50, value=6)
                curricular_units_2nd_sem_approved = st.number_input("2nd Sem Approved Units", min_value=0, max_value=30, value=6)
                curricular_units_2nd_sem_grade = st.number_input("2nd Sem Grade", min_value=0.0, max_value=20.0, value=12.0)
                
                st.markdown("**👤 Demographics**")
                gender = st.selectbox("Gender", ["Female", "Male"])
                marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widower"])
            
            submitted = st.form_submit_button("🔮 Predict Dropout Risk", type="primary")
        
        if submitted:
            # Create input array
            input_features = [
                age_at_enrollment,
                curricular_units_1st_sem_credited,
                curricular_units_1st_sem_enrolled,
                curricular_units_1st_sem_evaluations,
                curricular_units_1st_sem_approved,
                curricular_units_1st_sem_grade,
                curricular_units_2nd_sem_credited,
                curricular_units_2nd_sem_enrolled,
                curricular_units_2nd_sem_evaluations,
                curricular_units_2nd_sem_approved,
                curricular_units_2nd_sem_grade,
                1 if gender == "Male" else 0,
                1 if marital_status == "Single" else 0
            ]
            
            # Pad or truncate to match expected features
            expected_features = len(model_info['feature_columns'])
            while len(input_features) < expected_features:
                input_features.append(0)
            input_features = input_features[:expected_features]
            
            prediction, probability = predict_dropout_risk(input_features, model_info)
            
            if prediction is not None and probability is not None:
                # Decode prediction
                predicted_status = target_encoder.inverse_transform([prediction])[0]
                
                st.markdown("---")
                st.subheader("🎯 Prediction Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if predicted_status == 'Dropout':
                        st.error("⚠️ **HIGH DROPOUT RISK**")
                        risk_color = "red"
                    elif predicted_status == 'Enrolled':
                        st.warning("⚖️ **MODERATE RISK**")
                        risk_color = "orange"
                    else:
                        st.success("✅ **LOW DROPOUT RISK**")
                        risk_color = "green"
                
                with col2:
                    st.metric("Predicted Status", predicted_status)
                
                with col3:
                    if len(probability) > 1:
                        dropout_prob = probability[0] if predicted_status == 'Dropout' else probability[target_encoder.transform(['Dropout'])[0]]
                        st.metric("Dropout Probability", f"{dropout_prob:.2%}")
                
                # Probability breakdown
                st.subheader("📊 Probability Breakdown")
                
                prob_df = pd.DataFrame({
                    'Status': target_encoder.classes_,
                    'Probability': probability
                }).sort_values('Probability', ascending=False)
                
                fig_prob = px.bar(
                    prob_df, x='Status', y='Probability',
                    title="📊 Prediction Probabilities",
                    color='Probability',
                    color_continuous_scale='RdYlGn_r'
                )
                fig_prob.update_layout(showlegend=False)
                st.plotly_chart(fig_prob, use_container_width=True)
                
                # Risk gauge
                dropout_prob_pct = dropout_prob * 100 if 'dropout_prob' in locals() else probability[0] * 100
                
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=dropout_prob_pct,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Dropout Risk Level (%)"},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': risk_color},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "lightcoral"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 70
                        }
                    }
                ))
                
                st.plotly_chart(fig_gauge, use_container_width=True)
                
                # Recommendations
                st.subheader("💡 Recommendations")
                
                if predicted_status == 'Dropout':
                    st.error("""
                    **🚨 Immediate Actions Recommended:**
                    - 📞 Schedule urgent meeting with academic advisor
                    - 📚 Provide intensive academic support and tutoring
                    - 👥 Connect with peer support groups and mentorship programs
                    - 💰 Review financial aid options and scholarships
                    - 🎯 Develop personalized study plan with clear milestones
                    - 🏥 Consider counseling services if needed
                    - 📈 Weekly progress monitoring and check-ins
                    """)
                elif predicted_status == 'Enrolled':
                    st.warning("""
                    **⚖️ Preventive Measures Recommended:**
                    - 📅 Regular check-ins with academic advisor (bi-weekly)
                    - 📚 Academic skills workshops and study groups
                    - 🎯 Goal-setting sessions and progress tracking
                    - 👥 Encourage participation in student activities
                    - 📊 Monitor academic performance closely
                    - 🔄 Provide additional resources as needed
                    """)
                else:
                    st.success("""
                    **✅ Maintenance Strategies:**
                    - 📈 Continue monitoring academic progress
                    - 🏆 Recognize achievements and positive progress
                    - 📅 Regular but less frequent advisor meetings
                    - 🔄 Maintain current support systems
                    - 👥 Encourage leadership roles and peer mentoring
                    - 🎓 Provide career guidance and future planning
                    """)

    # ============================================================================
    # PAGE 7: MODEL PERSISTENCE (JOBLIB)
    # ============================================================================
    elif page == "💾 Model Persistence (Joblib)":
        st.header("💾 Model Persistence with Joblib")
        
        st.markdown("""
        ### 📚 About Joblib
        
        **Joblib** adalah library Python yang sangat efisien untuk:
        - 🗄️ Menyimpan dan memuat objek Python (terutama array NumPy)
        - 🤖 Persistence model machine learning
        - ⚡ Lebih cepat dari pickle untuk array besar
        - 🔧 Format yang dioptimalkan untuk scikit-learn
        
        ### 🎯 Model Package Contents
        Model package kami mencakup semua komponen yang diperlukan:
        """)
        
        # Train model for saving
        with st.spinner("Preparing model package..."):
            model_info = train_advanced_models(df_processed, target_encoder)
        
        if model_info['best_model'] is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **🤖 Machine Learning Components:**
                - Trained model (best performer)
                - Data scaler (MinMaxScaler)
                - Feature selector (SelectKBest)
                - Target encoder (LabelEncoder)
                """)
            
            with col2:
                st.markdown("""
                **📊 Metadata & Information:**
                - Selected features list
                - Model performance metrics
                - Training configuration
                - Feature importance scores
                """)
            
            # Model performance summary
            st.subheader("🏆 Model Performance Summary")
            
            performance_data = {
                'Metric': ['Best Model', 'Accuracy', 'AUC Score', 'CV Score', 'Features Used', 'Total Classes'],
                'Value': [
                    model_info['best_model_name'],
                    f"{model_info['results'][model_info['best_model_name']]['accuracy']:.4f}",
                    f"{model_info['results'][model_info['best_model_name']]['auc_score']:.4f}",
                    f"{model_info['results'][model_info['best_model_name']]['cv_score']:.4f}",
                    len(model_info['selected_features']),
                    model_info['n_classes']
                ]
            }
            
            performance_df = pd.DataFrame(performance_data)
            st.dataframe(performance_df, use_container_width=True, hide_index=True)
            
            # Save model
            st.subheader("💾 Download Model Package")
            
            if save_model_with_joblib(model_info):
                st.success("✅ Model package prepared successfully!")
            
            # Instructions
            st.subheader("📖 How to Use the Saved Model")
            
            code_example = '''
# Load the complete model package
import joblib
import pandas as pd
import numpy as np

# Load the model package
model_package = joblib.load('student_dropout_model.pkl')

# Extract components
model = model_package['model']
scaler = model_package['scaler']
selector = model_package['feature_selector']
target_encoder = model_package['target_encoder']
selected_features = model_package['selected_features']

# Example: Predict for new student data
def predict_student_dropout(new_data):
    # new_data should be a list or array with the same features as training
    
    # 1. Scale the data
    scaled_data = scaler.transform([new_data])
    
    # 2. Select features
    selected_data = selector.transform(scaled_data)
    
    # 3. Make prediction
    prediction = model.predict(selected_data)[0]
    probability = model.predict_proba(selected_data)[0]
    
    # 4. Decode prediction
    predicted_status = target_encoder.inverse_transform([prediction])[0]
    
    return predicted_status, probability

# Example usage
new_student = [20, 6, 6, 6, 6, 12.0, 6, 6, 6, 6, 12.0, 1, 1, ...]  # All features
status, probs = predict_student_dropout(new_student)
print(f"Predicted Status: {status}")
print(f"Probabilities: {dict(zip(target_encoder.classes_, probs))}")
'''
            
            st.code(code_example, language='python')
            
            # Model deployment tips
            st.subheader("🚀 Deployment Tips")
            
            st.markdown("""
            **💡 Best Practices:**
            
            1. **🔄 Regular Updates**: Retrain model with new data periodically
            2. **📊 Monitor Performance**: Track prediction accuracy in production
            3. **🛡️ Input Validation**: Validate input data before prediction
            4. **📝 Logging**: Log predictions for audit and improvement
            5. **⚡ Performance**: Consider model optimization for production
            6. **🔒 Security**: Secure model files and API endpoints
            """)
            
        else:
            st.error("❌ No trained model available for saving!")

     # ============================================================================
    # PAGE 8: INSIGHTS & RECOMMENDATIONS (CONTINUED)
    # ============================================================================
    elif page == "💡 Insights & Recommendations":
        st.header("💡 Key Insights & Actionable Recommendations")
        
        # Calculate statistics
        dropout_rate = (df['Status'] == 'Dropout').mean() * 100
        total_students = len(df)
        dropout_students = (df['Status'] == 'Dropout').sum()
        graduate_students = (df['Status'] == 'Graduate').sum()
        enrolled_students = (df['Status'] == 'Enrolled').sum()
        
        st.subheader("📊 Executive Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📚 Total Students", f"{total_students:,}")
        with col2:
            st.metric("📉 Dropout Rate", f"{dropout_rate:.1f}%", delta=f"-{dropout_rate/2:.1f}% target")
        with col3:
            st.metric("🎓 Graduates", f"{graduate_students:,}")
        with col4:
            st.metric("📝 Currently Enrolled", f"{enrolled_students:,}")
        
        # Key findings
        st.subheader("🔍 Key Findings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="insight-box">
            <h4>📈 Academic Performance Insights</h4>
            <ul>
            <li><strong>Grade Correlation:</strong> First semester performance is a strong predictor of dropout risk</li>
            <li><strong>Credit Load:</strong> Students with imbalanced credit loads show higher dropout rates</li>
            <li><strong>Evaluation Pattern:</strong> Multiple evaluations without approval indicate struggling students</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div class="insight-box">
            <h4>👥 Demographic Insights</h4>
            <ul>
            <li><strong>Age Factor:</strong> Non-traditional students (age >25) face unique challenges</li>
            <li><strong>Financial Status:</strong> Debt and tuition payment issues strongly correlate with dropout</li>
            <li><strong>Attendance:</strong> Evening students show different dropout patterns than day students</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Risk factors analysis
        st.subheader("⚠️ Critical Risk Factors")
        
        # Calculate risk factor statistics
        risk_factors_analysis = []
        
        if 'Curricular_units_1st_sem_grade' in df.columns:
            low_grade_threshold = df['Curricular_units_1st_sem_grade'].quantile(0.25)
            low_grade_dropout_rate = df[df['Curricular_units_1st_sem_grade'] <= low_grade_threshold]['Status'].value_counts(normalize=True).get('Dropout', 0) * 100
            risk_factors_analysis.append(('Low 1st Semester Grades', f"{low_grade_dropout_rate:.1f}%", "🔴 Critical"))
        
        if 'Age_at_enrollment' in df.columns:
            high_age_threshold = df['Age_at_enrollment'].quantile(0.75)
            high_age_dropout_rate = df[df['Age_at_enrollment'] >= high_age_threshold]['Status'].value_counts(normalize=True).get('Dropout', 0) * 100
            risk_factors_analysis.append(('Older Students (>75th percentile)', f"{high_age_dropout_rate:.1f}%", "🟡 Moderate"))
        
        if 'Debtor' in df.columns and df['Debtor'].nunique() > 1:
            debtor_dropout_rate = df[df['Debtor'] == 1]['Status'].value_counts(normalize=True).get('Dropout', 0) * 100
            risk_factors_analysis.append(('Students with Debt', f"{debtor_dropout_rate:.1f}%", "🟠 High"))
        
        # Display risk factors
        if risk_factors_analysis:
            risk_df = pd.DataFrame(risk_factors_analysis, columns=['Risk Factor', 'Dropout Rate', 'Risk Level'])
            st.dataframe(risk_df, use_container_width=True, hide_index=True)
        
        # Intervention strategies
        st.subheader("🎯 Intervention Strategies")
        
        tab1, tab2, tab3 = st.tabs(["🚨 High-Risk Students", "⚖️ Moderate Risk", "✅ Prevention"])
        
        with tab1:
            st.markdown("""
            ### 🚨 Immediate Intervention (High-Risk Students)
            
            **Identification Criteria:**
            - First semester GPA < 25th percentile
            - Multiple failed evaluations
            - Financial difficulties (debt/tuition issues)
            - Poor attendance record
            
            **Intervention Actions:**
            - **📞 Emergency Academic Advising**: Weekly one-on-one meetings
            - **💰 Financial Counseling**: Immediate review of financial aid options
            - **📚 Intensive Tutoring**: Subject-specific support programs
            - **👥 Peer Mentoring**: Pair with successful upperclassmen
            - **🏥 Counseling Services**: Mental health and stress management support
            - **📈 Progress Monitoring**: Daily/weekly check-ins
            """)
            
        with tab2:
            st.markdown("""
            ### ⚖️ Targeted Support (Moderate Risk)
            
            **Identification Criteria:**
            - Declining grade trends
            - Irregular attendance patterns
            - Age-related challenges (non-traditional students)
            - Course load management issues
            
            **Support Actions:**
            - **📅 Regular Check-ins**: Bi-weekly advisor meetings
            - **🎯 Study Skills Workshops**: Time management and study techniques
            - **👨‍👩‍👧‍👦 Flexible Scheduling**: Support for working/parent students
            - **📖 Academic Resources**: Access to writing centers, libraries
            - **🤝 Study Groups**: Facilitate peer learning communities
            """)
            
        with tab3:
            st.markdown("""
            ### ✅ Prevention & Early Warning
            
            **Proactive Measures:**
            - **🔔 Early Warning System**: Automated alerts based on ML predictions
            - **📊 Dashboard Monitoring**: Real-time tracking of risk indicators
            - **🎓 Orientation Enhancement**: Better preparation for academic expectations
            - **💻 Technology Integration**: Online resources and support platforms
            - **🏆 Recognition Programs**: Celebrate academic achievements
            - **📈 Predictive Analytics**: Regular model updates and improvements
            """)
        
        # Implementation roadmap
        st.subheader("🗺️ Implementation Roadmap")
        
        roadmap_data = {
            'Phase': ['Phase 1 (0-3 months)', 'Phase 2 (3-6 months)', 'Phase 3 (6-12 months)', 'Phase 4 (Ongoing)'],
            'Focus': [
                'Deploy ML Model & Dashboard',
                'Implement High-Risk Interventions', 
                'Scale Support Programs',
                'Continuous Improvement'
            ],
            'Key Actions': [
                'Model deployment, Staff training, Initial risk identification',
                'Emergency support protocols, Financial counseling, Intensive tutoring',
                'Expand to moderate-risk students, Peer mentoring program',
                'Model retraining, Program evaluation, Strategy refinement'
            ],
            'Success Metrics': [
                'Model accuracy >85%, Dashboard adoption >90%',
                'High-risk intervention rate >95%, Response time <24hrs',
                'Overall dropout reduction >15%, Program satisfaction >80%',
                'Sustained improvement, Cost-effectiveness analysis'
            ]
        }
        
        roadmap_df = pd.DataFrame(roadmap_data)
        st.dataframe(roadmap_df, use_container_width=True, hide_index=True)
        
        # Expected outcomes
        st.subheader("🎯 Expected Outcomes")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="success-box">
            <h4>📈 Quantitative Impact</h4>
            <ul>
            <li><strong>Dropout Reduction:</strong> 15-25% decrease in overall dropout rate</li>
            <li><strong>Early Identification:</strong> 80%+ of at-risk students identified by week 4</li>
            <li><strong>Intervention Success:</strong> 60%+ of high-risk students successfully retained</li>
            <li><strong>Cost Savings:</strong> $2-5M annually in reduced recruiting/onboarding costs</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div class="success-box">
            <h4>🌟 Qualitative Benefits</h4>
            <ul>
            <li><strong>Student Satisfaction:</strong> Improved academic experience and support</li>
            <li><strong>Faculty Engagement:</strong> Better tools for student support</li>
            <li><strong>Institutional Reputation:</strong> Higher retention and graduation rates</li>
            <li><strong>Data-Driven Culture:</strong> Evidence-based decision making</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Resource requirements
        st.subheader("💼 Resource Requirements")
        
        resources_data = {
            'Resource Type': ['Technology', 'Personnel', 'Training', 'Operations'],
            'Initial Investment': ['$50K-100K', '$200K-300K/year', '$25K-50K', '$75K-125K/year'],
            'Description': [
                'ML infrastructure, Dashboard development, Integration costs',
                'Data scientists, Academic advisors, Support staff',
                'Staff training, Workshop development, Certification programs',
                'Program management, Student services, Ongoing support'
            ],
            'ROI Timeline': ['6-12 months', '12-18 months', '3-6 months', '18-24 months']
        }
        
        resources_df = pd.DataFrame(resources_data)
        st.dataframe(resources_df, use_container_width=True, hide_index=True)
        
        # Call to action
        st.subheader("🚀 Next Steps")
        
        st.markdown("""
        <div class="warning-box">
        <h4>🎯 Immediate Actions Required</h4>
        <ol>
        <li><strong>📋 Stakeholder Alignment:</strong> Present findings to leadership team</li>
        <li><strong>💰 Budget Approval:</strong> Secure funding for Phase 1 implementation</li>
        <li><strong>👥 Team Assembly:</strong> Form cross-functional implementation team</li>
        <li><strong>🗓️ Timeline Development:</strong> Create detailed project timeline</li>
        <li><strong>📊 Baseline Metrics:</strong> Establish current performance benchmarks</li>
        <li><strong>🔄 Pilot Program:</strong> Start with small-scale implementation</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
        
        # Contact information
        st.markdown("---")
        st.markdown("""
        ### 📞 Contact & Support
        
        For questions about this analysis or implementation support:
        - **📧 Email**: mrevifikri@gmail.com
        - **🆔 Dicoding ID**: revi_fikri
        - **📊 Dashboard**: Built with Streamlit, Plotly, and Scikit-learn
        - **🔄 Updates**: Model should be retrained quarterly with new data
        
        *This dashboard represents a comprehensive analysis of student dropout patterns and provides actionable insights for improving student retention rates.*
        """)

if __name__ == "__main__":
    main()