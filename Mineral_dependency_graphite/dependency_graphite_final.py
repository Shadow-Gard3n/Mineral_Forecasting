import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. SHARED UTILITY FUNCTIONS
# ==========================================
def get_financial_year(row):
    """Converts Calendar Date to Indian Financial Year (Apr-Mar)."""
    try:
        y = int(row['Year'])
        m_num = {'Jan':1, 'Feb':2, 'Mar':3, 'Apr':4, 'May':5, 'Jun':6, 
                 'Jul':7, 'Aug':8, 'Sep':9, 'Oct':10, 'Nov':11, 'Dec':12}.get(row['Month'], 0)
        return f"{y}-{str(y+1)[-2:]}" if m_num >= 4 else f"{y-1}-{str(y)[-2:]}"
    except:
        return None

def calculate_hhi(df_trade, fy_list):
    """Calculates HHI (Concentration Risk) for a list of FYs."""
    hhi_list = []
    for fy in fy_list:
        sub = df_trade[df_trade['FY'] == fy]
        if sub.empty:
            hhi_list.append(0)
        else:
            total = sub['VALUE'].sum()
            if total == 0:
                hhi_list.append(0)
            else:
                shares = (sub.groupby('CTY')['VALUE'].sum() / total) * 100
                hhi_list.append((shares ** 2).sum())
    return hhi_list

def process_mineral_data(mineral_name, imp_file, exp_file, prod_dict):
    """Loads and processes data for a single mineral."""
    print(f"--- Processing {mineral_name} ---")
    
    # Load Data
    try:
        df_imp = pd.read_csv(imp_file)
        df_exp = pd.read_csv(exp_file)
    except FileNotFoundError:
        print(f"Error: Files for {mineral_name} not found.")
        return None, None, None

    # Filter Valid Trades
    df_imp = df_imp[df_imp['VALUE'] > 0].copy()
    df_exp = df_exp[df_exp['VALUE'] > 0].copy()
    
    # Apply Financial Year
    df_imp['FY'] = df_imp.apply(get_financial_year, axis=1)
    df_exp['FY'] = df_exp.apply(get_financial_year, axis=1)

    # Create Production DataFrame
    df_prod = pd.DataFrame(list(prod_dict.items()), columns=['FY', 'Production_Tonnes'])

    # Aggregate Trade (Convert KGS -> Tonnes)
    imp_fy = df_imp.groupby('FY')['QTY'].sum() / 1000
    exp_fy = df_exp.groupby('FY')['QTY'].sum() / 1000

    # Merge Production + Trade
    df = df_prod.merge(imp_fy.rename("Import"), left_on='FY', right_index=True, how='left')
    df = df.merge(exp_fy.rename("Export"), left_on='FY', right_index=True, how='left').fillna(0)

    # Calculate Strategic Metrics
    df['Net_Imports'] = df['Import'] - df['Export']
    df['Consumption'] = df['Production_Tonnes'] + df['Net_Imports']
    # Handle zero consumption edge case
    df['NIR_Percent'] = df.apply(lambda x: (x['Net_Imports']/x['Consumption']*100) if x['Consumption']!=0 else 0, axis=1)

    # Determine Status
    def get_status(nir):
        if nir < 0: return "Net Exporter"
        if nir > 50: return "High Dependency"
        if nir >= 30: return "Moderate Dependency"
        return "Resilient"
    df['Status'] = df['NIR_Percent'].apply(get_status)

    # Calculate Risk Scores (HHI)
    fys = df['FY'].tolist()
    df['Import_HHI'] = calculate_hhi(df_imp, fys)
    df['Export_HHI'] = calculate_hhi(df_exp, fys)

    return df, df_imp, df_exp

# ==========================================
# 2. VISUALIZATION FUNCTIONS
# ==========================================
def generate_dashboard(mineral_name, df, df_imp):
    """Creates the 3-panel dashboard image."""
    fig = plt.figure(figsize=(14, 10))
    
    # Panel 1: Stacked Bar (Dependency)
    ax1 = plt.subplot(2, 2, 1)
    ax1.bar(df['FY'], df['Production_Tonnes'], label='Domestic Production', color='green', alpha=0.6)
    ax1.bar(df['FY'], df['Net_Imports'], bottom=df['Production_Tonnes'], label='Net Imports', color='firebrick', alpha=0.6)
    ax2 = ax1.twinx()
    ax2.plot(df['FY'], df['NIR_Percent'], 'ko--', linewidth=2, label='Dependency (NIR) %')
    ax2.set_ylabel('Dependency %')
    ax1.axhline(0, color='gray', linewidth=0.8)
    ax1.set_title(f'{mineral_name}: Strategic Dependency Trend', fontweight='bold')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    # Panel 2: HHI Trend
    ax3 = plt.subplot(2, 2, 2)
    ax3.plot(df['FY'], df['Import_HHI'], 'bo-', label='Import Risk (Supply)')
    ax3.plot(df['FY'], df['Export_HHI'], 'o-', color='orange', label='Export Risk (Demand)')
    ax3.axhline(2500, color='red', linestyle=':', label='Critical Risk (>2500)')
    ax3.axhline(1500, color='green', linestyle=':', label='Safe Zone (<1500)')
    ax3.set_title(f'{mineral_name}: Supply Chain Risk Score', fontweight='bold')
    ax3.set_ylabel('HHI Score')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Panel 3: Pie Chart (Latest Available Year)
    ax4 = plt.subplot(2, 2, (3,4))
    target_year = df['FY'].iloc[-1] 
    if df_imp[df_imp['FY'] == target_year].empty: target_year = df['FY'].iloc[-2] # Fallback
        
    pie_data = df_imp[df_imp['FY'] == target_year].groupby('CTY')['VALUE'].sum().reset_index().sort_values('VALUE', ascending=False)
    
    if not pie_data.empty:
        if len(pie_data) > 6:
            top = pie_data.head(6)
            other = pd.DataFrame([{'CTY': 'Rest of World', 'VALUE': pie_data.iloc[6:]['VALUE'].sum()}])
            pie_data = pd.concat([top, other])
        ax4.pie(pie_data['VALUE'], labels=pie_data['CTY'], autopct='%1.1f%%', startangle=140, colors=plt.cm.Set3.colors)
        ax4.set_title(f'Import Sources ({target_year})', fontweight='bold')
    else:
        ax4.text(0.5, 0.5, "No Import Data Available", ha='center')

    plt.tight_layout()
    plt.savefig(f'{mineral_name}_dashboard.png')
    plt.close()

def generate_visual_table(mineral_name, df):
    """Creates the styled table image."""
    # Prepare Data for Display
    df_disp = df[['FY', 'Production_Tonnes', 'Net_Imports', 'NIR_Percent', 'Status']].copy()
    df_disp['Production_Tonnes'] = df_disp['Production_Tonnes'].apply(lambda x: f"{x:,.0f}")
    df_disp['Net_Imports'] = df_disp['Net_Imports'].apply(lambda x: f"{x:+,.0f}")
    df_disp['NIR_Percent'] = df_disp['NIR_Percent'].apply(lambda x: f"{x:.1f}%")
    df_disp.columns = ['FY', 'Production', 'Net Imports', 'Dependency', 'Status']
    
    # Create Plot
    fig, ax = plt.subplots(figsize=(10, len(df)*0.6 + 1))
    ax.axis('off')
    tbl = ax.table(cellText=df_disp.values, colLabels=df_disp.columns, loc='center', cellLoc='center')
    
    # Style Table
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.8)
    
    # Color Coding
    cells = tbl.get_celld()
    # Headers
    for j in range(len(df_disp.columns)):
        cells[(0, j)].set_facecolor('#40466e')
        cells[(0, j)].set_text_props(color='white', weight='bold')
        
    # Status Column Logic
    for i in range(len(df_disp)):
        status = df_disp.iloc[i, 4]
        cell = cells[(i+1, 4)]
        if "Net Exporter" in status:
            cell.set_facecolor('#d9f7be') # Green
            cell.set_text_props(color='darkgreen', weight='bold')
        elif "High" in status:
            cell.set_facecolor('#ffccc7') # Red
            cell.set_text_props(color='darkred', weight='bold')
        elif "Moderate" in status:
            cell.set_facecolor('#fff1b8') # Yellow
            cell.set_text_props(color='#d46b08', weight='bold')
        else:
            cell.set_facecolor('#e6f7ff') # Blue

    plt.title(f"{mineral_name} Strategic Analysis", weight='bold', y=1)
    plt.savefig(f'{mineral_name}_table.png', bbox_inches='tight', dpi=150)
    plt.close()

# ==========================================
# 3. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # --- Define Manual Production Data ---
    copper_prod = {
        '2019-20': 124586, '2020-21': 108718, '2021-22': 115313,
        '2022-23': 112745, '2023-24': 125230, '2024-25': 105012
    }
    graphite_prod = {
        '2019-20': 34674, '2020-21': 35386, '2021-22': 62888,
        '2022-23': 94789, '2023-24': 169080, '2024-25': 85329
    }

    # --- Run for Copper ---
    df_cop, df_cop_imp, _ = process_mineral_data('Copper', 'copper_data_imports.csv', 'copper_data.csv', copper_prod)
    if df_cop is not None:
        generate_dashboard('Copper', df_cop, df_cop_imp)
        generate_visual_table('Copper', df_cop)
        print("Copper Analysis Generated: Copper_dashboard.png, Copper_table.png")

    # --- Run for Graphite ---
    df_gra, df_gra_imp, _ = process_mineral_data('Graphite', 'Graphite_data_imports.csv', 'Graphite_data_exports.csv', graphite_prod)
    if df_gra is not None:
        generate_dashboard('Graphite', df_gra, df_gra_imp)
        generate_visual_table('Graphite', df_gra)
        print("Graphite Analysis Generated: Graphite_dashboard.png, Graphite_table.png")

#Matrices

# ==========================================
# 1. SHARED UTILITY FUNCTIONS
# ==========================================
def get_financial_year(row):
    """Converts Calendar Date to Indian Financial Year (Apr-Mar)."""
    try:
        y = int(row['Year'])
        m_num = {'Jan':1, 'Feb':2, 'Mar':3, 'Apr':4, 'May':5, 'Jun':6, 
                 'Jul':7, 'Aug':8, 'Sep':9, 'Oct':10, 'Nov':11, 'Dec':12}.get(row['Month'], 0)
        return f"{y}-{str(y+1)[-2:]}" if m_num >= 4 else f"{y-1}-{str(y)[-2:]}"
    except:
        return None

def calculate_hhi(df_trade, fy_list):
    """Calculates HHI (Concentration Risk) for a list of FYs."""
    hhi_list = []
    for fy in fy_list:
        sub = df_trade[df_trade['FY'] == fy]
        if sub.empty:
            hhi_list.append(0)
        else:
            total = sub['VALUE'].sum()
            if total == 0:
                hhi_list.append(0)
            else:
                shares = (sub.groupby('CTY')['VALUE'].sum() / total) * 100
                hhi_list.append((shares ** 2).sum())
    return hhi_list

def calculate_shannon(df_trade, fy_list):
    """Calculates Shannon Diversity Index (H') for a list of FYs."""
    shannon_list = []
    for fy in fy_list:
        sub = df_trade[df_trade['FY'] == fy]
        if sub.empty:
            shannon_list.append(0)
        else:
            total = sub['VALUE'].sum()
            if total == 0:
                shannon_list.append(0)
            else:
                shares = sub.groupby('CTY')['VALUE'].sum() / total
                shares = shares[shares > 0] # Filter > 0
                shannon = - (shares * np.log(shares)).sum()
                shannon_list.append(shannon)
    return shannon_list

def process_mineral_data(mineral_name, imp_file, exp_file, prod_dict):
    """Loads and processes data for a single mineral."""
    print(f"--- Processing {mineral_name} ---")
    
    # Load Data
    try:
        df_imp = pd.read_csv(imp_file)
        df_exp = pd.read_csv(exp_file)
    except FileNotFoundError:
        print(f"Error: Files for {mineral_name} not found. Skipping CSV processing.")
        return None, None, None

    # Filter Valid Trades
    df_imp = df_imp[df_imp['VALUE'] > 0].copy()
    df_exp = df_exp[df_exp['VALUE'] > 0].copy()
    
    # Apply Financial Year
    df_imp['FY'] = df_imp.apply(get_financial_year, axis=1)
    df_exp['FY'] = df_exp.apply(get_financial_year, axis=1)

    # Create Production DataFrame
    df_prod = pd.DataFrame(list(prod_dict.items()), columns=['FY', 'Production_Tonnes'])

    # Aggregate Trade (Convert KGS -> Tonnes)
    imp_fy = df_imp.groupby('FY')['QTY'].sum() / 1000
    exp_fy = df_exp.groupby('FY')['QTY'].sum() / 1000

    # Merge Production + Trade
    df = df_prod.merge(imp_fy.rename("Import"), left_on='FY', right_index=True, how='left')
    df = df.merge(exp_fy.rename("Export"), left_on='FY', right_index=True, how='left').fillna(0)

    # Calculate Strategic Metrics
    df['Net_Imports'] = df['Import'] - df['Export']
    df['Consumption'] = df['Production_Tonnes'] + df['Net_Imports']
    df['NIR_Percent'] = df.apply(lambda x: (x['Net_Imports']/x['Consumption']*100) if x['Consumption']!=0 else 0, axis=1)

    # Calculate Risk Scores (HHI & Shannon)
    fys = df['FY'].tolist()
    df['Import_HHI'] = calculate_hhi(df_imp, fys)
    df['Export_HHI'] = calculate_hhi(df_exp, fys)
    df['Import_Shannon'] = calculate_shannon(df_imp, fys)
    df['Export_Shannon'] = calculate_shannon(df_exp, fys)

    return df, df_imp, df_exp

# ==========================================
# 2. VISUALIZATION FUNCTIONS (UPDATED)
# ==========================================
def generate_visual_table(df, title="Strategic Metrics Table"):
    """
    Generates a matplotlib table visualization for the provided dataframe.
    """
    # 1. Prepare Data for Table
    # Select only relevant columns
    cols_to_show = ['FY', 'NIR_Percent', 'Import_HHI', 'Export_HHI', 'Import_Shannon', 'Export_Shannon']
    
    # Check if columns exist (handle manual data vs calculated data)
    table_data = df[cols_to_show].copy()
    
    # 2. Formatting
    # Format NIR as percentage string
    if pd.api.types.is_numeric_dtype(table_data['NIR_Percent']):
        table_data['NIR_Percent'] = table_data['NIR_Percent'].apply(lambda x: f"{x:.1f}%")
    
    # Format HHI with commas (and handle 0 as "No Data" if preferred, or just 0)
    table_data['Import_HHI'] = table_data['Import_HHI'].apply(lambda x: f"{int(x):,}" if pd.notnull(x) and x!=0 else "0 (No data)")
    table_data['Export_HHI'] = table_data['Export_HHI'].apply(lambda x: f"{int(x):,}" if pd.notnull(x) else "0")
    
    # Format Shannon to 2 decimals
    table_data['Import_Shannon'] = table_data['Import_Shannon'].apply(lambda x: f"{x:.2f}")
    table_data['Export_Shannon'] = table_data['Export_Shannon'].apply(lambda x: f"{x:.2f}")

    # Rename columns for display
    col_mapping = {
        'FY': 'Financial Year',
        'NIR_Percent': 'NIR %',
        'Import_HHI': 'Import HHI\n(Supply Risk)',
        'Export_HHI': 'Export HHI\n(Demand Risk)',
        'Import_Shannon': "Import Shannon\n(H')",
        'Export_Shannon': "Export Shannon\n(H')"
    }
    table_data = table_data.rename(columns=col_mapping)

    # 3. Plotting
    fig, ax = plt.subplots(figsize=(10, 4)) # Adjust size as needed
    ax.axis('tight')
    ax.axis('off')
    
    # Create the table
    table = ax.table(cellText=table_data.values,
                     colLabels=table_data.columns,
                     cellLoc='center',
                     loc='center')
    
    # Styling
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2) # Adjust width/height scaling

    # Color header
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#40466e') # Dark Blue Header
        elif row % 2 == 0:
            cell.set_facecolor('#f5f5f5') # Zebra striping
            
    plt.title(title, weight='bold', pad=20)
    plt.show()

# ==========================================
# 3. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    
    # --- A. MANUAL DATA ENTRY (The table you requested) ---
    # Since we might not have the CSV files, we create the dataframe manually
    # based on the data provided in the prompt.
    
    print("--- Generatng Table from Provided Data ---")
    
    manual_data = {
        'FY': ['2019-20', '2020-21', '2021-22', '2022-23', '2023-24', '2024-25'],
        'NIR_Percent': [-21.7, -13.9, 42.6, 35.8, 29.3, 54.9],
        'Import_HHI': [0, 0, 4313, 5911, 6106, 4223],
        'Export_HHI': [3372, 2419, 2173, 2994, 2748, 2123],
        'Import_Shannon': [0.00, 0.00, 1.54, 1.13, 1.10, 1.63],
        'Export_Shannon': [1.68, 2.05, 2.01, 1.77, 1.86, 2.20]
    }
    
    df_manual = pd.DataFrame(manual_data)
    
    # Visualize the Manual Table
    generate_visual_table(df_manual, "Strategic Risk Analysis (Consolidated)")
    
    print("\nTable generated successfully.")

    # --- B. OPTIONAL: CSV PROCESSING ---
    # (This part runs only if you have the CSV files locally)
    
    run_csv_processing = False # Set to True if you have the files
    
    if run_csv_processing:
        # Define Production
        copper_prod = {
            '2019-20': 124586, '2020-21': 108718, '2021-22': 115313,
            '2022-23': 112745, '2023-24': 125230, '2024-25': 105012
        }
        
        # Run Copper
        df_cop, _, _ = process_mineral_data('Copper', 'copper_data_imports.csv', 'copper_data.csv', copper_prod)
        if df_cop is not None:
            generate_visual_table(df_cop, "Copper Strategic Metrics")