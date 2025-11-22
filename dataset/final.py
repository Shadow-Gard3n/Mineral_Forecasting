import requests
import csv
import time

# --- CONFIGURATION ---

# 1. The Commodities List you provided
COMMODITIES_LIST = [
    # --- EXISTING CODES (FROM YOUR LIST) ---
    "25309099", # Other Mineral Substances (Lithium/Strontium Ores)
    "28369100", # Lithium Carbonates
    "28252000", # Lithium Oxide and Hydroxide
    "26050010", # Cobalt Ores & Concentrates
    "26050090", # Cobalt Ores (Other)
    "75011000", # Nickel Mattes
    "81052010", # Cobalt Mattes/Intermediate Products
    "81052020", # Cobalt Unwrought
    "25309040", # Ores of Rare Earth Metals
    "28461010", # Cerium Compounds
    "72029921", # Ferro-Alloys (General)
    "25041010", # Natural Graphite (Crystalline)
    "25041020", # Natural Graphite (Amorphous)
    "38011000", # Artificial Graphite
    "26140010", # Ilmenite (Titanium Ore)
    "26140020", # Rutile (Titanium Ore)
    "28230010", # Titanium Dioxide
    "81082000", # Titanium Unwrought

    # --- NEW ADDITIONS (STRATEGIC & CRITICAL MINERALS) ---
    
    # 1. Base Metals (Copper, Nickel, Tin)
    "26030000", # Copper Ores
    "74031100", # Copper Cathodes (Refined)
    "26040000", # Nickel Ores
    "75021000", # Nickel Unwrought
    "26090000", # Tin Ores
    "80011090", # Tin Unwrought

    # 2. Rare Earths & Strategic Metals (REE, Niobium, Tantalum, Vanadium)
    "28469090", # Rare Earth Compounds (Mix)
    "28053000", # Rare Earth Metals (Scandium/Yttrium)
    "26159020", # Niobium Ores
    "26159010", # Tantalum Ores
    "81032010", # Tantalum Unwrought
    "26159090", # Vanadium Ores

    # 3. High Tech Metals (Gallium, Indium, Hafnium, Rhenium)
    # Note: 81129200 is a "Catch-all" for Gallium, Hafnium, Indium, Niobium, Rhenium
    "81129200", # Unwrought Gallium, Hafnium, Indium, Niobium, Rhenium
    "81121200", # Beryllium Unwrought
    "26179090", # Beryllium Ores
    "81092000", # Zirconium Unwrought
    "26151000", # Zirconium Ores

    # 4. Energy & Industrial (Tungsten, Molybdenum, Antimony)
    "26110000", # Tungsten Ores
    "81019400", # Tungsten Unwrought
    "26131000", # Molybdenum Ores
    "81029400", # Molybdenum Unwrought
    "26171000", # Antimony Ores
    "81101000", # Antimony Unwrought

    # 5. Non-Metals (Silicon, Phosphorous, Potash, Tellurium, Selenium)
    "25051000", # Silica Sands (Silicon Ore)
    "28046100", # Silicon (Electronic Grade)
    "25101010", # Rock Phosphate
    "28047020", # Phosphorous (Red)
    "31042000", # Potash (MOP)
    "28045020", # Tellurium
    "28049000", # Selenium
    
    # 6. Precious Group (PGE, Bismuth, Cadmium)
    "26169010", # Platinum Group Ores
    "71101110", # Platinum Unwrought
    "81061090", # Bismuth
    "81072000"  # Cadmium
]

# Convert list to comma-separated string for the API
COMMODITIES_STRING = ",".join(COMMODITIES_LIST)

# 2. The Guest Username (Replace this if the script gives Error 401/405)
GUEST_ID = "22112025100976GUEST"

# 3. Base URL
URL = "https://ftddp.dgciskol.gov.in/dgcis/guestsearch"

# ---------------------

def get_monthly_list():
    """Generates specific monthly periods from Apr-2017 to Sep-2025"""
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    date_list = []
    
    for year in range(2025, 2026):
        start_index = 0
        end_index = 12
        
        if year == 2017: start_index = 3 # Start Apr 2017
        if year == 2025: end_index = 9   # End Sep 2025
            
        for i in range(start_index, end_index):
            date_list.append(f"{months[i]}-{year}")
            
    return date_list

def fetch_data(mode, date_str, filename, is_first_write):
    """
    mode: 'E' for Export, 'I' for Import
    date_str: 'Apr-2017'
    """
    
    # Headers to look like a real user
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
    }

    # Payload matching your Request parameters
    payload = {
        "username": GUEST_ID,
        "eximp": mode,           # 'E' or 'I'
        "datepicker": date_str,  # Start Date
        "datepicker1": date_str, # End Date (Same as start to isolate month)
        "commodities": COMMODITIES_STRING,
        "countries": "A",
        "country_Org": "undefined",
        "regions": "undefined",
        "ports": "A",
        "sorted": "Order By HS_CODE,CTY,Value DESC",
        "currency": "B",
        "digites": "8",
        "reg": "2",
        "offset": "1",
        "noOfRecords": "1", # Check count first
        "description": "1",
        "type": "10",
        "scrollDate": "SEP25"
    }

    try:
        print(f"[{mode}] Checking {date_str}...", end=" ")
        
        # 1. Get Count
        response = requests.post(URL, data=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        # Check if list exists in initial response
        if "searchlists" in data and len(data["searchlists"]) > 0:
            
            # Fetch a larger batch
            payload["noOfRecords"] = "5000" 
            time.sleep(0.5)
            
            resp_all = requests.post(URL, data=payload, headers=headers)
            resp_all.raise_for_status()
            final_data = resp_all.json()
            raw_records = final_data.get("searchlists", [])
            
            if raw_records:
                # --- CLEANING LOGIC START ---
                valid_records = []
                month_part, year_part = date_str.split('-')
                
                for record in raw_records:
                    # CRITICAL FIX: Only process rows that have actual data (ignore empty footers)
                    if "HS_CODE" in record or "COMMODITY" in record:
                        record['Year'] = year_part
                        record['Month'] = month_part
                        record['Date_Full'] = f"01-{month_part}-{year_part}"
                        record['Type'] = "Export" if mode == 'E' else "Import"
                        valid_records.append(record)
                # --- CLEANING LOGIC END ---

                if valid_records:
                    # Save to CSV
                    mode_write = 'w' if is_first_write else 'a'
                    with open(filename, mode_write, newline='', encoding='utf-8') as f:
                        # Use keys from the first VALID record
                        writer = csv.DictWriter(f, fieldnames=valid_records[0].keys())
                        if is_first_write:
                            writer.writeheader()
                        writer.writerows(valid_records)
                    print(f"Found & Saved {len(valid_records)} rows.")
                    return True
                else:
                    print("List contained only empty/invalid rows.")
            else:
                print("No records in full fetch.")
        else:
            print("0 records.")

    except Exception as e:
        print(f"\nError on {date_str}: {e}")
        
    return False

def run_process(mode, filename):
    months = get_monthly_list()
    first_write = True
    print(f"\n--- STARTING {filename} ({len(months)} Months) ---")
    
    for date_str in months:
        # Only set first_write to False if we ACTUALLY wrote valid data
        if fetch_data(mode, date_str, filename, first_write):
            first_write = False
        time.sleep(1) # Be polite to server

if __name__ == "__main__":
    # 1. Run Exports
    run_process("E", "Minerals_Exports_Final_Cleaned_3_2017_2025.csv")
    
    # 2. Run Imports
    run_process("I", "Minerals_Imports_Final_Cleaned_2017_2025.csv")
    
    print("\nAll tasks completed.")