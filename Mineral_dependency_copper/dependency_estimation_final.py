import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. DATA PREPARATION
# ==========================================
def load_and_prep_data():
    """Loads CSVs and manually adds production data."""
    # Load Trade Data
    df_exp = pd.read_csv('copper_data_exports.csv')
    df_imp = pd.read_csv('copper_data_imports.csv')
    
    # Filter valid non-zero trades
    df_exp = df_exp[df_exp['VALUE'] > 0].copy()
    df_imp = df_imp[df_imp['VALUE'] > 0].copy()
    
    # Manual Production Data (Copper Concentrate)
    # Source: User provided text input
    prod_data = {
        'FY': ['2019-20', '2020-21', '2021-22', '2022-23', '2023-24', '2024-25'],
        'Production_Tonnes': [124586, 108718, 115313, 112745, 125230, 105012]
    }
    df_prod = pd.DataFrame(prod_data)
    
    return df_exp, df_imp, df_prod

def get_financial_year(row):
    """Converts Calendar Date to Indian Financial Year (Apr-Mar)."""
    try:
        y = int(row['Year'])
        m = row['Month']
        months = {'Jan':1, 'Feb':2, 'Mar':3, 'Apr':4, 'May':5, 'Jun':6, 
                  'Jul':7, 'Aug':8, 'Sep':9, 'Oct':10, 'Nov':11, 'Dec':12}
        m_num = months.get(m, 0)
        # If Apr(4) or later, it belongs to current-next year
        if m_num >= 4:
            return f"{y}-{str(y+1)[-2:]}"
        else:
            return f"{y-1}-{str(y)[-2:]}"
    except:
        return None

# ==========================================
# 2. CALCULATIONS (NIR & HHI)
# ==========================================
def calculate_nir(df_prod, df_exp, df_imp):
    """Calculates Net Import Reliance (Dependency %)."""
    # Aggregate Trade by FY (Convert KGS to TONNES by /1000)
    exp_fy = df_exp.groupby('FY')['QTY'].sum() / 1000 
    imp_fy = df_imp.groupby('FY')['QTY'].sum() / 1000
    
    # Merge Production + Imports + Exports
    df = df_prod.merge(imp_fy.rename("Import_Tonnes"), left_on='FY', right_index=True, how='left')
    df = df.merge(exp_fy.rename("Export_Tonnes"), left_on='FY', right_index=True, how='left')
    df = df.fillna(0)
    
    # NIR Formula: (Net Imports / Apparent Consumption) * 100
    df['Net_Imports'] = df['Import_Tonnes'] - df['Export_Tonnes']
    df['Consumption'] = df['Production_Tonnes'] + df['Net_Imports']
    df['NIR_Percent'] = (df['Net_Imports'] / df['Consumption']) * 100
    
    return df

def calculate_hhi(df_trade, fy_list):
    """Calculates Herfindahl-Hirschman Index for a list of years."""
    hhi_list = []
    for fy in fy_list:
        df_fy = df_trade[df_trade['FY'] == fy]
        if df_fy.empty:
            hhi_list.append(0)
            continue
            
        # Group by Country and calculate shares
        cty_val = df_fy.groupby('CTY')['VALUE'].sum()
        total = cty_val.sum()
        
        if total == 0:
            hhi_list.append(0)
        else:
            shares = (cty_val / total) * 100
            hhi = (shares ** 2).sum() # Sum of squared shares
            hhi_list.append(hhi)
    return hhi_list

# ==========================================
# 3. MAIN EXECUTION FLOW
# ==========================================
# 1. Load Data
df_exp, df_imp, df_prod = load_and_prep_data()

# 2. Process Dates
df_exp['FY'] = df_exp.apply(get_financial_year, axis=1)
df_imp['FY'] = df_imp.apply(get_financial_year, axis=1)

# 3. Calculate Dependency (NIR)
df_metrics = calculate_nir(df_prod, df_exp, df_imp)

# 4. Calculate Risk (HHI)
fys = df_metrics['FY'].tolist()
df_metrics['Import_HHI'] = calculate_hhi(df_imp, fys)
df_metrics['Export_HHI'] = calculate_hhi(df_exp, fys)

# 5. Display Text Report
print("--- STRATEGIC MINERAL REPORT: COPPER CONCENTRATE ---")
print(df_metrics[['FY', 'Production_Tonnes', 'NIR_Percent', 'Import_HHI', 'Export_HHI']].round(1))

# ==========================================
# 4. VISUALIZATION DASHBOARD
# ==========================================
fig = plt.figure(figsize=(14, 10))

# Plot A: Strategic Dependency (Stacked Bar + Line)
# Shows the gap between "What we have" vs "What we need"
ax1 = plt.subplot(2, 2, 1)
ax1.bar(df_metrics['FY'], df_metrics['Production_Tonnes'], label='Domestic Production', color='green', alpha=0.6)
ax1.bar(df_metrics['FY'], df_metrics['Net_Imports'], bottom=df_metrics['Production_Tonnes'], label='Net Imports', color='firebrick', alpha=0.6)
ax2 = ax1.twinx()
ax2.plot(df_metrics['FY'], df_metrics['NIR_Percent'], color='black', marker='o', linestyle='--', linewidth=2, label='Dependency %')
ax2.set_ylim(0, 110)
ax1.set_title('Strategic Dependency (Volume vs Reliance)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Tonnes')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# Plot B: Risk Concentration Trend (HHI)
# Shows if we are relying on fewer countries over time
ax3 = plt.subplot(2, 2, 2)
ax3.plot(df_metrics['FY'], df_metrics['Import_HHI'], marker='o', label='Import Risk (Supply)', color='blue')
ax3.plot(df_metrics['FY'], df_metrics['Export_HHI'], marker='o', label='Export Risk (Demand)', color='orange')
# Add Risk Threshold Zones
ax3.axhline(2500, color='red', linestyle=':', label='Critical Risk (>2500)')
ax3.axhline(1500, color='green', linestyle=':', label='Safe Zone (<1500)')
ax3.set_title('Supply Chain Vulnerability (HHI Score)', fontsize=12, fontweight='bold')
ax3.set_ylabel('HHI Risk Score (0-10,000)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot C: Import Sources Pie (Latest Year)
# Shows WHO we depend on right now
target_year = '2024-25'
ax4 = plt.subplot(2, 2, (3,4)) # Span bottom row
# Prepare Pie Data
df_pie = df_imp[df_imp['FY'] == target_year].groupby('CTY')['VALUE'].sum().reset_index()
df_pie = df_pie.sort_values('VALUE', ascending=False)
if len(df_pie) > 6:
    top = df_pie.head(6)
    other_val = df_pie.iloc[6:]['VALUE'].sum()
    other_row = pd.DataFrame([{'CTY': 'Rest of World', 'VALUE': other_val}])
    df_pie_final = pd.concat([top, other_row])
else:
    df_pie_final = df_pie

ax4.pie(df_pie_final['VALUE'], labels=df_pie_final['CTY'], autopct='%1.1f%%', startangle=140, colors=plt.cm.Set3.colors)
ax4.set_title(f'Current Supply Sources ({target_year})', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('comprehensive_analysis.png')
plt.show()