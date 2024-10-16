import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_absolute_error, mean_squared_error
from joblib import dump, load
from utils import recomienda_tfid


# Page configuration
st.set_page_config(page_title="DeepInsightz", page_icon=":bar_chart:", layout="wide")

# Custom CSS for styling similar to the inspiration
st.markdown("""
<style>
[data-testid="block-container"] {
    padding-left: 2rem;
    padding-right: 2rem;
    padding-top: 1rem;
    padding-bottom: 0rem;
    margin-bottom: -7rem;
}
[data-testid="stVerticalBlock"] {
    padding-left: 0rem;
    padding-right: 0rem;
}
[data-testid="stMetric"] {
    background-color: #393939;
    text-align: center;
    padding: 15px 0;
}
[data-testid="stMetricLabel"] {
  display: flex;
  justify-content: center;
  align-items: center;
}
[data-testid="stMetricDeltaIcon-Up"] {
    position: relative;
    left: 38%;
    -webkit-transform: translateX(-50%);
    -ms-transform: translateX(-50%);
    transform: translateX(-50%);
}
[data-testid="stMetricDeltaIcon-Down"] {
    position: relative;
    left: 38%;
    -webkit-transform: translateX(-50%);
    -ms-transform: translateX(-50%);
    transform: translateX(-50%);
}
</style>
""", unsafe_allow_html=True)

# Load CSV files at the top
df = pd.read_csv("df_clean.csv")
nombres_proveedores = pd.read_csv("nombres_proveedores.csv", sep=';')
euros_proveedor = pd.read_csv("euros_proveedor.csv", sep=',')
ventas_clientes = pd.read_csv("ventas_clientes.csv", sep=',')
customer_clusters = pd.read_csv('predicts/customer_clusters.csv')  # Load the customer clusters here
df_agg_2024 = pd.read_csv('predicts/df_agg_2024.csv') 

# Ensure customer codes are strings
df['CLIENTE'] = df['CLIENTE'].astype(str)
nombres_proveedores['codigo'] = nombres_proveedores['codigo'].astype(str)
euros_proveedor['CLIENTE'] = euros_proveedor['CLIENTE'].astype(str)
customer_clusters['cliente_id'] = customer_clusters['cliente_id'].astype(str)  # Ensure customer IDs are strings
fieles_df = pd.read_csv("clientes_relevantes.csv")
cestas = pd.read_csv("cestas.csv")
productos = pd.read_csv("productos.csv")
df_agg_2024['cliente_id'] = df_agg_2024['cliente_id'].astype(str)

# Convert all columns except 'CLIENTE' to float in euros_proveedor
for col in euros_proveedor.columns:
    if col != 'CLIENTE':
        euros_proveedor[col] = pd.to_numeric(euros_proveedor[col], errors='coerce')

# Check for NaN values after conversion
if euros_proveedor.isna().any().any():
    st.warning("Some values in euros_proveedor couldn't be converted to numbers. Please review the input data.")

# Ignore the last two columns of df
df = df.iloc[:, :-2]

# Function to get supplier name
def get_supplier_name(code):
    code = str(code)  # Ensure code is a string
    name = nombres_proveedores[nombres_proveedores['codigo'] == code]['nombre'].values
    return name[0] if len(name) > 0 else code

# Function to create radar chart with square root transformation
def radar_chart(categories, values, amounts, title):
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
    
    # Apply square root transformation
    sqrt_values = np.sqrt(values)
    sqrt_amounts = np.sqrt(amounts)
    
    max_sqrt_value = max(sqrt_values)
    normalized_values = [v / max_sqrt_value for v in sqrt_values]
    
    # Adjust scaling for spend values
    max_sqrt_amount = max(sqrt_amounts)
    scaling_factor = 0.7  # Adjust this value to control how much the spend values are scaled up
    normalized_amounts = [min((a / max_sqrt_amount) * scaling_factor, 1.0) for a in sqrt_amounts]
    
    normalized_values += normalized_values[:1]
    ax.plot(angles, normalized_values, 'o-', linewidth=2, color='#FF69B4', label='% Units (sqrt)')
    ax.fill(angles, normalized_values, alpha=0.25, color='#FF69B4')
    
    normalized_amounts += normalized_amounts[:1]
    ax.plot(angles, normalized_amounts, 'o-', linewidth=2, color='#4B0082', label='% Spend (sqrt)')
    ax.fill(angles, normalized_amounts, alpha=0.25, color='#4B0082')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=8, wrap=True)
    ax.set_ylim(0, 1)
    
    circles = np.linspace(0, 1, 5)
    for circle in circles:
        ax.plot(angles, [circle]*len(angles), '--', color='gray', alpha=0.3, linewidth=0.5)
    
    ax.set_yticklabels([])
    ax.spines['polar'].set_visible(False)
    
    plt.title(title, size=16, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    return fig

# Navigation menu
with st.sidebar:
    st.sidebar.title("DeepInsightz")
    page = st.sidebar.selectbox("Select the tool you want to use", ["Summary", "Customer Analysis", "Articles Recommendations"])



if page == "Summary":
    # st.title("Welcome to DeepInsightz")
    # st.markdown("""
    #     ### Data-driven Customer Clustering
    #     We analyzed thousands of customers and suppliers to help businesses make smarter sales decisions.
    # """)

    # Create layout with three columns
    col1, col2, col3 = st.columns((1.5, 4.5, 2), gap='medium')

    # Left Column (Red): Metrics and Donut Charts
    with col1:
        st.markdown('#### Key Metrics')
        st.metric(label="Texas", value="29.0 M", delta="+367 K", delta_color="normal")
        st.metric(label="New York", value="19.5 M", delta="-77 K", delta_color="inverse")
        
        st.markdown('#### States Migration')
        
        # Create a placeholder for your own donut charts with Plotly
        donut_fig = px.pie(values=[70, 30], names=['Inbound', 'Outbound'], hole=0.7)
        donut_fig.update_traces(textinfo='percent+label')
        donut_fig.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(donut_fig, use_container_width=True)


    # Middle Column (White): 3D Cluster Model and Bar Chart
    with col2:
        st.markdown('#### 3D Customer Clusters')
        
        # Replace with your own customer cluster visualization
        np.random.seed(42)
        df_cluster = pd.DataFrame({
            'Cluster': np.random.choice(['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4'], 100),
            'x': np.random.randn(100),
            'y': np.random.randn(100),
            'z': np.random.randn(100)
        })
        fig_cluster = px.scatter_3d(df_cluster, x='x', y='y', z='z', color='Cluster')
        st.plotly_chart(fig_cluster, use_container_width=True)
        
        st.markdown('#### Sales by Cluster')
        
        # Replace with your own sales data
        sales_data = {'Cluster': ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4'],
                    'Sales': [400000, 600000, 500000, 300000]}
        df_sales = pd.DataFrame(sales_data)
        fig_sales = px.bar(df_sales, x='Cluster', y='Sales', title="Sales by Cluster")
        st.plotly_chart(fig_sales, use_container_width=True)

    # Right Column (Blue): Key Metrics Overview and Data Preparation Summary
    with col3:
        st.markdown('#### Key Metrics Overview')
        st.write("""
            - **Customers Analyzed**: 4,000
            - **Suppliers Analyzed**: 400
            - **Invoice Lines Processed**: 800,000
        """)

        st.markdown('#### Data Preparation Summary')
        st.write("""
            - Cleaned and standardized product codes and descriptions.
            - Excluded customers with fewer than 12 purchases or sales below €1,200.
        """)

# Customer Analysis Page
elif page == "Customer Analysis":
    st.title("Customer Analysis")
    st.markdown("Use the tools below to explore your customer data.")

    partial_code = st.text_input("Enter part of Customer Code (or leave empty to see all)")
    if partial_code:
        filtered_customers = df[df['CLIENTE'].str.contains(partial_code)]
    else:
        filtered_customers = df
    customer_list = filtered_customers['CLIENTE'].unique()
    customer_code = st.selectbox("Select Customer Code", customer_list)

    if st.button("Calcular"):
        if customer_code:
            # Find Customer's Cluster
            customer_match = customer_clusters[customer_clusters['cliente_id'] == customer_code]

            if not customer_match.empty:
                cluster = customer_match['cluster_id'].values[0]
                st.write(f"Customer {customer_code} belongs to cluster {cluster}")

                # Load the Corresponding Model
                model_path = f'models/modelo_cluster_{cluster}.txt'
                gbm = lgb.Booster(model_file=model_path)
                st.write(f"Loaded model for cluster {cluster}")

                # Inspect the model
                st.write("### Model Information:")
                st.write(f"Number of trees: {gbm.num_trees()}")
                st.write(f"Number of features: {gbm.num_feature()}")
                st.write("Feature names:")
                st.write(gbm.feature_name())

                # Load predict data for that cluster
                predict_data = pd.read_csv(f'predicts/predict_cluster_{cluster}.csv')
                
                # Convert cliente_id to string
                predict_data['cliente_id'] = predict_data['cliente_id'].astype(str)
                
                st.write("### Predict Data DataFrame:")
                st.write(predict_data.head())
                st.write(f"Shape: {predict_data.shape}")

                # Filter for the specific customer
                customer_code_str = str(customer_code)
                customer_data = predict_data[predict_data['cliente_id'] == customer_code_str]
                
                # Add debug statements
                st.write(f"Unique customer IDs in predict data: {predict_data['cliente_id'].unique()}")
                st.write(f"Customer code we're looking for: {customer_code_str}")

                st.write("### Customer Data:")
                st.write(customer_data.head())
                st.write(f"Shape: {customer_data.shape}")

                if not customer_data.empty:
                    # Define features consistently with the training process
                    lag_features = [f'precio_total_lag_{lag}' for lag in range(1, 25)]
                    features = lag_features + ['mes', 'marca_id_encoded', 'año', 'cluster_id']
                    
                    # Prepare data for prediction
                    X_predict = customer_data[features]

                    # Convert categorical features to 'category' dtype
                    categorical_features = ['mes', 'marca_id_encoded', 'cluster_id']
                    for feature in categorical_features:
                        X_predict[feature] = X_predict[feature].astype('category')
                    
                    st.write("### Features for Prediction:")
                    st.write(X_predict.head())
                    st.write(f"Shape: {X_predict.shape}")
                    st.write("Data types:")
                    st.write(X_predict.dtypes)
                    
                    # Make Prediction for the selected customer
                    y_pred = gbm.predict(X_predict, num_iteration=gbm.best_iteration)
                    st.write("### Prediction Results:")
                    st.write(f"Type of y_pred: {type(y_pred)}")
                    st.write(f"Shape of y_pred: {y_pred.shape}")
                    st.write("First few predictions:")
                    st.write(y_pred[:5])
                    
                    # Reassemble the results
                    results = customer_data[['cliente_id', 'marca_id_encoded', 'fecha_mes']].copy()
                    results['ventas_predichas'] = y_pred
                    st.write("### Results DataFrame:")
                    st.write(results.head())
                    st.write(f"Shape: {results.shape}")
                    
                    st.write(f"Predicted total sales for Customer {customer_code}: {results['ventas_predichas'].sum():.2f}")

                    # Load actual data
                    actual_sales = df_agg_2024[df_agg_2024['cliente_id'] == customer_code_str]
                    st.write("### Actual Sales DataFrame:")
                    st.write(actual_sales.head())
                    st.write(f"Shape: {actual_sales.shape}")
                    
                    if not actual_sales.empty:
                        results = results.merge(actual_sales[['cliente_id', 'marca_id_encoded', 'fecha_mes', 'precio_total']], 
                                                on=['cliente_id', 'marca_id_encoded', 'fecha_mes'], 
                                                how='left')
                        results.rename(columns={'precio_total': 'ventas_reales'}, inplace=True)
                        results['ventas_reales'].fillna(0, inplace=True)
                        st.write("### Final Results DataFrame:")
                        st.write(results.head())
                        st.write(f"Shape: {results.shape}")
                        
                        # Calculate metrics only for non-null actual sales
                        valid_results = results.dropna(subset=['ventas_reales'])
                        if not valid_results.empty:
                            mae = mean_absolute_error(valid_results['ventas_reales'], valid_results['ventas_predichas'])
                            mape = np.mean(np.abs((valid_results['ventas_reales'] - valid_results['ventas_predichas']) / valid_results['ventas_reales'])) * 100
                            rmse = np.sqrt(mean_squared_error(valid_results['ventas_reales'], valid_results['ventas_predichas']))

                            st.write(f"Actual total sales for Customer {customer_code}: {valid_results['ventas_reales'].sum():.2f}")
                            st.write(f"MAE: {mae:.2f}")
                            st.write(f"MAPE: {mape:.2f}%")
                            st.write(f"RMSE: {rmse:.2f}")

                        # Analysis of results
                        threshold_good = 100  # You may want to adjust this threshold
                        if mae < threshold_good:
                            st.success(f"Customer {customer_code} is performing well based on the predictions.")
                        else:
                            st.warning(f"Customer {customer_code} is not performing well based on the predictions.")
                    else:
                        st.warning(f"No actual sales data found for customer {customer_code} in df_agg_2024.")

                    st.write("### Debug Information for Radar Chart:")
                    st.write(f"Shape of customer_data: {customer_data.shape}")
                    st.write(f"Shape of euros_proveedor: {euros_proveedor.shape}")

                    # Get percentage of units sold for each manufacturer
                    customer_df = df[df["CLIENTE"] == str(customer_code)]  # Get the customer data
                    all_manufacturers = customer_df.iloc[:, 1:].T  # Exclude CLIENTE column (manufacturers are in columns)
                    all_manufacturers.index = all_manufacturers.index.astype(str)

                    # Get total sales for each manufacturer from euros_proveedor
                    customer_euros = euros_proveedor[euros_proveedor["CLIENTE"] == str(customer_code)]
                    sales_data = customer_euros.iloc[:, 1:].T  # Exclude CLIENTE column
                    sales_data.index = sales_data.index.astype(str)

                    # Remove the 'CLIENTE' row from sales_data to avoid issues with mixed types
                    sales_data_filtered = sales_data.drop(index='CLIENTE', errors='ignore')

                    # Ensure all values are numeric
                    sales_data_filtered = sales_data_filtered.apply(pd.to_numeric, errors='coerce')
                    all_manufacturers = all_manufacturers.apply(pd.to_numeric, errors='coerce')

                    # Sort manufacturers by percentage of units and get top 10
                    top_units = all_manufacturers.sort_values(by=all_manufacturers.columns[0], ascending=False).head(10)

                    # Sort manufacturers by total sales and get top 10
                    top_sales = sales_data_filtered.sort_values(by=sales_data_filtered.columns[0], ascending=False).head(10)

                    # Combine top manufacturers from both lists and get up to 20 unique manufacturers
                    combined_top = pd.concat([top_units, top_sales]).index.unique()[:20]

                    # Filter out manufacturers that are not present in both datasets
                    combined_top = [m for m in combined_top if m in all_manufacturers.index and m in sales_data_filtered.index]

                    st.write(f"Number of combined top manufacturers: {len(combined_top)}")

                    if combined_top:
                        # Create a DataFrame with combined data for these top manufacturers
                        combined_data = pd.DataFrame({
                            'units': all_manufacturers.loc[combined_top, all_manufacturers.columns[0]],
                            'sales': sales_data_filtered.loc[combined_top, sales_data_filtered.columns[0]]
                        }).fillna(0)

                        # Sort by units, then by sales
                        combined_data_sorted = combined_data.sort_values(by=['units', 'sales'], ascending=False)

                        # Filter out manufacturers with 0 units
                        non_zero_manufacturers = combined_data_sorted[combined_data_sorted['units'] > 0]

                        # If we have less than 3 non-zero manufacturers, add some zero-value ones
                        if len(non_zero_manufacturers) < 3:
                            zero_manufacturers = combined_data_sorted[combined_data_sorted['units'] == 0].head(3 - len(non_zero_manufacturers))
                            manufacturers_to_show = pd.concat([non_zero_manufacturers, zero_manufacturers])
                        else:
                            manufacturers_to_show = non_zero_manufacturers

                        values = manufacturers_to_show['units'].tolist()
                        amounts = manufacturers_to_show['sales'].tolist()
                        manufacturers = [get_supplier_name(m) for m in manufacturers_to_show.index]

                        st.write(f"### Results for top {len(manufacturers)} manufacturers:")
                        for manufacturer, value, amount in zip(manufacturers, values, amounts):
                            st.write(f"{manufacturer} = {value:.2f}% of units, €{amount:.2f} total sales")

                        if manufacturers:  # Only create the chart if we have data
                            fig = radar_chart(manufacturers, values, amounts, f'Radar Chart for Top {len(manufacturers)} Manufacturers of Customer {customer_code}')
                            st.pyplot(fig)
                        else:
                            st.warning("No data available to create the radar chart.")

                    else:
                        st.warning("No combined top manufacturers found.")

                    # Ensure codigo_cliente in ventas_clientes is a string
                    ventas_clientes['codigo_cliente'] = ventas_clientes['codigo_cliente'].astype(str).str.strip()

                    # Ensure customer_code is a string and strip any spaces
                    customer_code = str(customer_code).strip()

                    if customer_code in ventas_clientes['codigo_cliente'].unique():
                        st.write(f"Customer {customer_code} found in ventas_clientes")  
                    else:
                        st.write(f"Customer {customer_code} not found in ventas_clientes")

                    # Customer sales 2021-2024 (if data exists)
                    sales_columns = ['VENTA_2021', 'VENTA_2022', 'VENTA_2023']
                    if all(col in ventas_clientes.columns for col in sales_columns):
                        customer_sales_data = ventas_clientes[ventas_clientes['codigo_cliente'] == customer_code]
                        
                        if not customer_sales_data.empty:
                            customer_sales = customer_sales_data[sales_columns].values[0]
                            years = ['2021', '2022', '2023']
                            
                            fig_sales = px.line(x=years, y=customer_sales, markers=True, title=f'Sales Over the Years for Customer {customer_code}')
                            fig_sales.update_layout(xaxis_title="Year", yaxis_title="Sales")
                            st.plotly_chart(fig_sales)
                        else:
                            st.warning(f"No historical sales data found for customer {customer_code}")
                    else:
                        st.warning("Sales data for 2021-2023 not available in the dataset.")
                else:
                    st.warning(f"No data found for customer {customer_code}. Please check the code.")
        else:
            st.warning("Please select a customer.")


# Customer Recommendations Page
elif page == "Articles Recommendations":
    st.title("Articles Recommendations")

    st.markdown("""
        Get tailored recommendations for your customers based on their basket.
    """)

    st.write("Select items and assign quantities for the basket:")

    # Mostrar lista de artículos disponibles
    available_articles = productos['ARTICULO'].unique()
    selected_articles = st.multiselect("Select Articles", available_articles)

    # Crear inputs para ingresar las cantidades de cada artículo seleccionado
    quantities = {}
    for article in selected_articles:
        quantities[article] = st.number_input(f"Quantity for {article}", min_value=0, step=1)

    if st.button("Calcular"):  # Añadimos el botón "Calcular"
                # Crear una lista de artículos basada en la selección
        new_basket = [f"{article} x{quantities[article]}" for article in selected_articles if quantities[article] > 0]

        if new_basket:
            # Procesar la lista para recomendar
            recommendations_df = recomienda_tfid(new_basket)

            if not recommendations_df.empty:
                st.write("### Recommendations based on the current basket:")
                st.dataframe(recommendations_df)
            else:
                st.warning("No recommendations found for the provided basket.")
        else:
            st.warning("Please select at least one article and set its quantity.")