import streamlit as st
import pandas as pd
import numpy as np
import math
from scipy.stats import norm, uniform
import plotly.express as px  # For advanced visualizations
import streamlit.components.v1 as components  # For rendering SVG/PDF if needed

# Optional financial library
try:
    import numpy_financial as nf
    IRR_AVAILABLE = True
except ImportError:
    IRR_AVAILABLE = False

# PDF generation for report output
from fpdf import FPDF

# ---------------------------
# 1. SIMPLE AUTHENTICATION
# ---------------------------
def authenticate():
    st.title("Gasification Feasibility Tool - Secure Login")
    secret_password = "IgudArim2025"  # This password is for Igud Arim only
    password = st.text_input("Enter your client password:", type="password", key="auth_password")
    if st.button("Login", key="login_button"):
        if password == secret_password:
            st.session_state['authenticated'] = True
            st.success("Authentication successful!")
        else:
            st.error("Incorrect password. Please try again.")

if 'authenticated' not in st.session_state or not st.session_state['authenticated']:
    authenticate()
    st.stop()

# ---------------------------
# 2. GLOBAL INPUT DATA
# ---------------------------
ENHANCED_INPUT_DATA = {
    "facility": {
        "annual_capacity_tons": 100_000,
        "daily_capacity_tons": 330,
        "operational_days": 300,
        "gasification_temp_c": 700,
        "energy_consumption_kwh_per_ton": 91.5,
        # Default 550 for incineration-based reference, but we'll override with a dropdown
        "electricity_generation_kwh_per_ton": 550,
        "waste_moisture_pct": 25,
        # Default 5%, but we’ll override with dropdown
        "ash_content_pct": 5,
        # Keep these for scenario analysis
        "max_feedstock_tons_day": 500,
        "nominal_daily_capacity_tons": 300
    },
    "economics": {
        "capex_usd": 100_000_000,
        "base_opex_usd_per_year": 6_000_000,  # covers all maintenance – no separate schedule
        "opex_scaling_factor": 0.10,
        "carbon_cert_one_time_cost_usd": 220_000,
        "carbon_cert_annual_fee_usd": 5_500,
        "carbon_credit_price_usd_per_t_co2": 10.0,
        "electricity_sales_price_usd_per_kwh": 0.11,
        "transport_savings_usd_per_ton": 36.30,
        "municipal_treatment_cost_usd_per_ton": 114.0,
        "base_opex_usd_year1": 6_000_000,
        "inflation_rate_pct": 2.0,
        # Removed the maintenance schedule lines
        # "maintenance_schedule": {5: 1_000_000, 10: 2_000_000},  # <-- Removed
        "tipping_fee_usd_per_ton": 90.0
    },
    "financing": {
        "project_duration_years": 20,
        "tax_incentives_pct": 0.30,
        "discount_rate_pct": 8.0,
        "project_life": 20
    },
    "ghg_baseline": {
        "landfill_emissions_kg_per_ton": 1721,
        "facility_emissions_kg_per_ton": 566.5,
        "methane_factor_kg_co2eq_per_ton": 100
    },
    "trucking": {
        # Default to 15 but we’ll override with dropdown for 15 or 18
        "truck_capacity_tons": 15,
        # Default 400 but we’ll add a dropdown with 400, 450, 500
        "distance_to_landfill_km": 400,
        "emission_factor_kgco2_per_km": 2.26
    }
}

# Multi-feedstock dictionary (unchanged, for scenario usage)
FEEDSTOCKS = {
    "RDF": {
        "fraction": 0.70,
        # Will override with user’s choice of kWh/ton if needed,
        # but let's keep the default 550 here for the code's internal reference
        "electricity_kwh_per_ton": 550,
        "ghg_facility_kg_per_ton": 566.5
    },
    "GreenWaste": {
        "fraction": 0.30,
        "electricity_kwh_per_ton": 550,
        "ghg_facility_kg_per_ton": 566.5
    }
}

# Year-by-year feedstock data
YEARLY_FEEDSTOCK_DATA = {
    1: {"daily_feedstock_tons": 200, "capacity_factor": 0.80},
    2: {"daily_feedstock_tons": 250, "capacity_factor": 0.85},
    3: {"daily_feedstock_tons": 300, "capacity_factor": 0.90},
    4: {"daily_feedstock_tons": 320, "capacity_factor": 0.95},
    5: {"daily_feedstock_tons": 350, "capacity_factor": 1.00},
}
for y in range(6, 21):
    YEARLY_FEEDSTOCK_DATA[y] = {"daily_feedstock_tons": 350, "capacity_factor": 1.0}

# ---------------------------
# 3. HELPER FUNCTIONS
# ---------------------------
def show_footer():
    st.markdown("""
---
**Operated by Oporto-Carbon | Designed & Developed by Dr. Avi Luvchik**  
@ All Rights Reserved 2025
""")

def compute_bau_ghg_tons(daily_capacity, input_data):
    days = input_data["facility"]["operational_days"]
    annual_waste = daily_capacity * days
    ghg_landfill_kg = input_data["ghg_baseline"]["landfill_emissions_kg_per_ton"] * annual_waste
    truck_capacity = input_data["trucking"]["truck_capacity_tons"]
    distance = input_data["trucking"]["distance_to_landfill_km"]
    ef_truck = input_data["trucking"]["emission_factor_kgco2_per_km"]
    trucks_per_year = annual_waste / truck_capacity
    ghg_trucking_kg = trucks_per_year * distance * ef_truck
    return (ghg_landfill_kg + ghg_trucking_kg) / 1000.0

def compute_project_ghg_tons(daily_capacity, feedstocks, input_data):
    days = input_data["facility"]["operational_days"]
    annual_waste = daily_capacity * days
    total_fac_ghg_kg = 0.0
    for key, fstock_data in feedstocks.items():
        frac = fstock_data["fraction"]
        portion = annual_waste * frac
        ghg_fac = fstock_data["ghg_facility_kg_per_ton"]
        total_fac_ghg_kg += ghg_fac * portion
    local_distance_km = 20.0  # fixed local trucking distance
    ef_truck = input_data["trucking"]["emission_factor_kgco2_per_km"]
    truck_capacity = input_data["trucking"]["truck_capacity_tons"]
    trucks_per_year = annual_waste / truck_capacity
    ghg_local_kg = trucks_per_year * local_distance_km * ef_truck
    return (total_fac_ghg_kg + ghg_local_kg) / 1000.0

def compute_revenue_pie(daily_capacity, feedstocks, input_data, fee_mode="Both"):
    """
    fee_mode options:
        "Both" -> use both municipal cost and tipping fee
        "Municipal only" -> skip tipping fee
        "Tipping only" -> skip municipal
    """
    days = input_data["facility"]["operational_days"]
    annual_waste = daily_capacity * days

    # ELECTRICITY
    total_kwh = 0.0
    for key, fstock_data in feedstocks.items():
        frac = fstock_data["fraction"]
        portion = annual_waste * frac
        total_kwh += portion * fstock_data["electricity_kwh_per_ton"]

    elec_price = input_data["economics"]["electricity_sales_price_usd_per_kwh"]
    elec_rev = total_kwh * elec_price

    # CARBON
    ghg_bau = compute_bau_ghg_tons(daily_capacity, input_data)
    ghg_proj = compute_project_ghg_tons(daily_capacity, feedstocks, input_data)
    ghg_reduction_tons = max(0, ghg_bau - ghg_proj)
    carbon_price = input_data["economics"]["carbon_credit_price_usd_per_t_co2"]
    carbon_rev = ghg_reduction_tons * carbon_price

    # TIPPING & MUNICIPAL
    tipping_fee_usd_per_ton = input_data["economics"]["tipping_fee_usd_per_ton"]
    municipal_usd_per_ton = input_data["economics"]["municipal_treatment_cost_usd_per_ton"]

    tipping_rev = 0.0
    municipal_rev = 0.0
    if fee_mode == "Both":
        tipping_rev = annual_waste * tipping_fee_usd_per_ton
        municipal_rev = annual_waste * municipal_usd_per_ton
    elif fee_mode == "Municipal only":
        municipal_rev = annual_waste * municipal_usd_per_ton
    elif fee_mode == "Tipping only":
        tipping_rev = annual_waste * tipping_fee_usd_per_ton

    return {
        "Carbon": carbon_rev,
        "Electricity": elec_rev,
        # Summarize each separately, so user can see them in scenario results:
        "Municipal": municipal_rev,
        "Tipping": tipping_rev
    }

def discounted_cash_flow_series(annual_net, project_life, discount_rate):
    r = discount_rate / 100.0
    return [annual_net / ((1 + r)**y) for y in range(1, project_life+1)]

def is_none_or_empty(df):
    return (df is None) or (df.empty)

# ---------------------------
# 4. MAIN APPLICATION
# ---------------------------
def main():
    # Define tabs
    tabs = st.tabs([
        "Home",
        "Base Input Data",
        "Scenario Analysis",
        "Year-by-Year Analysis",
        "Monte Carlo",
        "Cost Analysis",
        "Conclusions"
    ])
    
    ###########################################################################
    # TAB 0: HOME
    ###########################################################################
    with tabs[0]:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("Gasification Feasibility Tool")
    with col2:
        try:
            st.image("oporto_logo.png", use_column_width=True)
        except Exception as e:
            st.error(f"Error loading logo: {e}")
    st.markdown("""
Welcome to Oporto-Carbon's Gasification Feasibility Tool.  
This tool evaluates the economic and environmental performance of a gasification facility.
""")
        show_footer()
    
    ###########################################################################
    # TAB 1: BASE INPUT DATA
    ###########################################################################
    with tabs[1]:
        st.subheader("Base Input Data")
        
        def flatten_dict(d, parent_key="", sep="."):
            items = {}
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.update(flatten_dict(v, new_key, sep=sep))
                else:
                    items[new_key] = v
            return items
        
        # Display the original dictionary in flattened form
        flat_input = flatten_dict(ENHANCED_INPUT_DATA)
        df_input = pd.DataFrame(list(flat_input.items()), columns=["Parameter", "Value"])
        st.write("### Core Input Parameters (Default)")
        st.dataframe(df_input, use_container_width=True)
        
        # Let the user override certain parameters
        st.write("### Override Key Parameters with Dropdowns")
        distance_choice = st.selectbox(
            "Distance to landfill (km)", [400, 450, 500],
            index=[400,450,500].index(ENHANCED_INPUT_DATA["trucking"]["distance_to_landfill_km"])
        )
        ENHANCED_INPUT_DATA["trucking"]["distance_to_landfill_km"] = distance_choice

        truck_cap_choice = st.selectbox(
            "Truck Capacity (tons)", [15, 18],
            index=[15,18].index(ENHANCED_INPUT_DATA["trucking"]["truck_capacity_tons"])
        )
        ENHANCED_INPUT_DATA["trucking"]["truck_capacity_tons"] = truck_cap_choice
        
        ash_choice = st.selectbox(
            "Ash Content (%)", [5, 7, 10],
            index=[5,7,10].index(ENHANCED_INPUT_DATA["facility"]["ash_content_pct"])
        )
        ENHANCED_INPUT_DATA["facility"]["ash_content_pct"] = ash_choice
        
        # Also let the user choose electricity output for RDF: 550 or 800 (the client requested 800)
        # We'll apply it to facility as well as the FEEDSTOCKS dictionary
        elec_options = [550, 800]
        elec_choice = st.selectbox(
            "Electricity Generation (kWh/ton)", elec_options,
            index=elec_options.index(ENHANCED_INPUT_DATA["facility"]["electricity_generation_kwh_per_ton"])
        )
        ENHANCED_INPUT_DATA["facility"]["electricity_generation_kwh_per_ton"] = elec_choice
        FEEDSTOCKS["RDF"]["electricity_kwh_per_ton"] = elec_choice
        FEEDSTOCKS["GreenWaste"]["electricity_kwh_per_ton"] = elec_choice
        
        st.write("### Municipal vs Tipping Fee Approach")
        fee_mode = st.selectbox(
            "Use both municipal & tipping fees, or only one?",
            ["Both", "Municipal only", "Tipping only"],
            index=0  # default is "Both"
        )
        st.session_state["fee_mode"] = fee_mode
        
        st.write("### Year-by-Year Feedstock Data")
        feedstock_rows = []
        for year in sorted(YEARLY_FEEDSTOCK_DATA.keys()):
            data = YEARLY_FEEDSTOCK_DATA[year]
            feedstock_rows.append({
                "Year": year,
                "Daily Feedstock (t/day)": data["daily_feedstock_tons"],
                "Capacity Factor": data["capacity_factor"]
            })
        st.dataframe(pd.DataFrame(feedstock_rows), use_container_width=True)
        
        st.write("### Multi-Feedstock Approach")
        multi_list = []
        for key, v in FEEDSTOCKS.items():
            multi_list.append({
                "Feedstock": key,
                "Fraction": v["fraction"],
                "Electricity (kWh/ton)": v["electricity_kwh_per_ton"],
                "Facility GHG (kgCO2/ton)": v["ghg_facility_kg_per_ton"]
            })
        st.dataframe(pd.DataFrame(multi_list), use_container_width=True)
        
        show_footer()

    ###########################################################################
    # TAB 2: SCENARIO ANALYSIS (Single & Multi)
    ###########################################################################
    with tabs[2]:
        st.subheader("Scenario Analysis")
        mode = st.radio("Select Scenario Mode", ("Single Scenario", "Multi-Scenario"), key="scenario_mode")
        
        if mode == "Single Scenario":
            st.markdown("#### Single Scenario Settings")
            daily_capacity = st.number_input("Daily Capacity (t/day)", min_value=50, max_value=2000,
                                             value=ENHANCED_INPUT_DATA["facility"]["daily_capacity_tons"],
                                             step=10, key="daily_capacity_single")
            
            new_rdf_frac = st.slider("RDF Fraction", 0.0, 1.0, FEEDSTOCKS["RDF"]["fraction"], 0.05, key="rdf_fraction")
            FEEDSTOCKS["RDF"]["fraction"] = new_rdf_frac
            FEEDSTOCKS["GreenWaste"]["fraction"] = 1.0 - new_rdf_frac
            
            ghg_bau = compute_bau_ghg_tons(daily_capacity, ENHANCED_INPUT_DATA)
            ghg_proj = compute_project_ghg_tons(daily_capacity, FEEDSTOCKS, ENHANCED_INPUT_DATA)
            ghg_reduction = max(0, ghg_bau - ghg_proj)

            # Use the fee_mode from the Base Input tab
            fee_mode = st.session_state.get("fee_mode", "Both")
            revs = compute_revenue_pie(daily_capacity, FEEDSTOCKS, ENHANCED_INPUT_DATA, fee_mode=fee_mode)
            
            # Environmental Metrics (e.g., water usage and particulate emissions)
            water_usage = daily_capacity * ENHANCED_INPUT_DATA["facility"]["operational_days"] * 1.0  # 1 m³ per ton
            particulate_emissions = daily_capacity * ENHANCED_INPUT_DATA["facility"]["operational_days"] * 0.05  # 0.05 kg per ton
            
            st.write("#### Single Scenario Results")
            df_single = pd.DataFrame([{
                "Daily Capacity (t/day)": daily_capacity,
                "GHG BAU (t)": ghg_bau,
                "GHG Project (t)": ghg_proj,
                "GHG Reduction (t)": ghg_reduction,
                "Revenue Carbon (USD)": revs["Carbon"],
                "Revenue Electricity (USD)": revs["Electricity"],
                "Revenue Municipal (USD)": revs["Municipal"],
                "Revenue Tipping (USD)": revs["Tipping"],
                "Water Usage (m³/year)": water_usage,
                "Particulate Emissions (kg/year)": particulate_emissions
            }])
            st.dataframe(df_single)
            
            fig = px.bar(df_single, x=["GHG BAU (t)", "GHG Project (t)"],
                         title="GHG Emissions Comparison",
                         labels={"value": "Tons CO2eq", "variable": "Scenario"})
            st.plotly_chart(fig, use_container_width=True)
            
            rev_df = pd.DataFrame({
                "Revenue Source": list(revs.keys()),
                "USD Amount": list(revs.values())
            })
            fig2 = px.pie(rev_df, names="Revenue Source", values="USD Amount",
                          title="Revenue Breakdown")
            st.plotly_chart(fig2, use_container_width=True)
            
            st.session_state["scenario_df"] = df_single
            
        else:  # Multi-Scenario mode
            st.markdown("#### Multi-Scenario Settings")
            min_cap = st.number_input("Min Capacity (t/day)", 50, 2000, 100, 10, key="min_cap")
            max_cap = st.number_input("Max Capacity (t/day)", 50, 2000, 600, 10, key="max_cap")
            step_cap = st.number_input("Step for Capacity", 1, 500, 50, 10, key="step_cap")
            min_c = st.number_input("Min Carbon Price (USD/t)", 0.0, 300.0, 5.0, 5.0, key="min_c")
            max_c = st.number_input("Max Carbon Price (USD/t)", 0.0, 300.0, 30.0, 5.0, key="max_c")
            step_c = st.number_input("Carbon Price Step", 1.0, 50.0, 5.0, 1.0, key="step_c")
            
            if st.button("Run Multi-Scenario", key="run_multi"):
                scenario_list = []
                fee_mode = st.session_state.get("fee_mode", "Both")
                for cap in range(min_cap, max_cap+1, step_cap):
                    for cp in np.arange(min_c, max_c+0.001, step_c):
                        ENHANCED_INPUT_DATA["economics"]["carbon_credit_price_usd_per_t_co2"] = cp
                        ghg_bau_ = compute_bau_ghg_tons(cap, ENHANCED_INPUT_DATA)
                        ghg_proj_ = compute_project_ghg_tons(cap, FEEDSTOCKS, ENHANCED_INPUT_DATA)
                        ghg_red_ = max(0, ghg_bau_ - ghg_proj_)
                        rev_dict_ = compute_revenue_pie(cap, FEEDSTOCKS, ENHANCED_INPUT_DATA, fee_mode=fee_mode)
                        scenario_list.append({
                            "Capacity (t/day)": cap,
                            "Carbon Price (USD/t)": cp,
                            "GHG BAU (t)": ghg_bau_,
                            "GHG Project (t)": ghg_proj_,
                            "GHG Reduction (t)": ghg_red_,
                            "Revenue Carbon": rev_dict_["Carbon"],
                            "Revenue Elec": rev_dict_["Electricity"],
                            "Revenue Municipal": rev_dict_["Municipal"],
                            "Revenue Tipping": rev_dict_["Tipping"]
                        })
                df_multi = pd.DataFrame(scenario_list)
                st.dataframe(df_multi)
                st.session_state["scenario_df"] = df_multi
                st.success("Multi-scenario run complete!")
        show_footer()
    
    ###########################################################################
    # TAB 3: YEAR-BY-YEAR ANALYSIS
    ###########################################################################
    with tabs[3]:
        st.subheader("Year-by-Year Analysis")
        disc_rate_ = st.number_input("Discount Rate (%)", 0.0, 50.0,
                                     ENHANCED_INPUT_DATA["financing"]["discount_rate_pct"], 0.5, key="disc_rate")
        cprice_ = st.number_input("Base Carbon Price (USD/t)", 0.0, 300.0,
                                  ENHANCED_INPUT_DATA["economics"]["carbon_credit_price_usd_per_t_co2"], 5.0, key="cprice")
        dynamic_trend = st.checkbox("Use Dynamic Market Trends", key="dynamic_trend")
        if dynamic_trend:
            carbon_trend = st.number_input("Annual Carbon Price Change (%)", -10.0, 10.0, 0.0, 0.5, key="carbon_trend")
            elec_trend = st.number_input("Annual Electricity Price Change (%)", -10.0, 10.0, 0.0, 0.5, key="elec_trend")
        else:
            carbon_trend = 0.0
            elec_trend = 0.0
        
        if st.button("Run Year-by-Year Analysis", key="run_yby"):
            ENHANCED_INPUT_DATA["financing"]["discount_rate_pct"] = disc_rate_
            ENHANCED_INPUT_DATA["economics"]["carbon_credit_price_usd_per_t_co2"] = cprice_
            
            fee_mode = st.session_state.get("fee_mode", "Both")
            results_list = []
            for year in sorted(YEARLY_FEEDSTOCK_DATA.keys()):
                daily_tons = YEARLY_FEEDSTOCK_DATA[year]["daily_feedstock_tons"]
                # Adjust prices based on trend: compounded for each year (year-1)
                adjusted_carbon = cprice_ * ((1 + carbon_trend/100)**(year-1))
                adjusted_elec = ENHANCED_INPUT_DATA["economics"]["electricity_sales_price_usd_per_kwh"] * ((1 + elec_trend/100)**(year-1))
                ENHANCED_INPUT_DATA["economics"]["carbon_credit_price_usd_per_t_co2"] = adjusted_carbon
                ENHANCED_INPUT_DATA["economics"]["electricity_sales_price_usd_per_kwh"] = adjusted_elec
                
                ghg_bau_ = compute_bau_ghg_tons(daily_tons, ENHANCED_INPUT_DATA)
                ghg_proj_ = compute_project_ghg_tons(daily_tons, FEEDSTOCKS, ENHANCED_INPUT_DATA)
                red_ = max(0, ghg_bau_ - ghg_proj_)
                revs_ = compute_revenue_pie(daily_tons, FEEDSTOCKS, ENHANCED_INPUT_DATA, fee_mode=fee_mode)
                total_rev = sum(revs_.values())
                opex_ = ENHANCED_INPUT_DATA["economics"]["base_opex_usd_per_year"]
                net_cf = total_rev - opex_
                results_list.append({
                    "Year": year,
                    "Daily Feedstock (t/day)": daily_tons,
                    "GHG BAU (t)": ghg_bau_,
                    "GHG Project (t)": ghg_proj_,
                    "GHG Reduction (t)": red_,
                    "Revenue Carbon": revs_["Carbon"],
                    "Revenue Elec": revs_["Electricity"],
                    "Revenue Municipal": revs_["Municipal"],
                    "Revenue Tipping": revs_["Tipping"],
                    "Net CF (USD)": net_cf
                })
            df_yby = pd.DataFrame(results_list)
            st.dataframe(df_yby)
            st.line_chart(df_yby.set_index("Year")[["GHG BAU (t)", "GHG Project (t)"]])
            st.session_state["yearbyyear_df"] = df_yby
            st.success("Year-by-year analysis complete!")
        show_footer()
    
    ###########################################################################
    # TAB 4: MONTE CARLO SIMULATION
    ###########################################################################
    with tabs[4]:
        st.subheader("Monte Carlo Simulation for Uncertain Parameters")
        runs_ = st.number_input("Number of Monte Carlo Runs", 100, 100000, 1000, 100, key="mc_runs")
        carbon_mean_ = st.number_input("Carbon Price Mean (USD/t)", 0.0, 300.0, 10.0, 1.0, key="carbon_mean")
        carbon_std_  = st.number_input("Carbon Price StdDev", 0.0, 100.0, 2.0, 0.5, key="carbon_std")
        elec_price_mean = st.number_input("Electricity Price Mean (USD/kWh)", 0.0, 1.0, 0.11, 0.01, key="elec_mean")
        elec_price_std = st.number_input("Electricity Price StdDev", 0.0, 0.5, 0.02, 0.005, key="elec_std")
        
        if st.button("Run Monte Carlo", key="run_mc"):
            mc_results = []
            daily_cap = ENHANCED_INPUT_DATA["facility"]["daily_capacity_tons"]
            base_opex = ENHANCED_INPUT_DATA["economics"]["base_opex_usd_per_year"]
            fee_mode = st.session_state.get("fee_mode", "Both")
            for i in range(int(runs_)):
                cp_sample = np.random.normal(carbon_mean_, carbon_std_)
                if cp_sample < 0:
                    cp_sample = 0
                ENHANCED_INPUT_DATA["economics"]["carbon_credit_price_usd_per_t_co2"] = cp_sample
                
                ep_sample = np.random.normal(elec_price_mean, elec_price_std)
                if ep_sample < 0:
                    ep_sample = 0
                ENHANCED_INPUT_DATA["economics"]["electricity_sales_price_usd_per_kwh"] = ep_sample
                
                ghg_bau_ = compute_bau_ghg_tons(daily_cap, ENHANCED_INPUT_DATA)
                ghg_proj_ = compute_project_ghg_tons(daily_cap, FEEDSTOCKS, ENHANCED_INPUT_DATA)
                red_ = max(0, ghg_bau_ - ghg_proj_)
                revs_ = compute_revenue_pie(daily_cap, FEEDSTOCKS, ENHANCED_INPUT_DATA, fee_mode=fee_mode)
                net_annual = sum(revs_.values()) - base_opex
                mc_results.append(net_annual)
            df_mc = pd.DataFrame({"Annual Net Cash Flow (USD)": mc_results})
            st.write("### Monte Carlo Simulation Results Summary")
            st.write(df_mc.describe())
            st.bar_chart(df_mc)
            st.session_state["monte_carlo_df"] = df_mc
        show_footer()
    
    ###########################################################################
    # TAB 5: COST ANALYSIS
    ###########################################################################
    with tabs[5]:
        st.subheader("Cost Analysis (Discounted Cash Flow)")
        colC1, colC2 = st.columns(2)
        with colC1:
            daily_cap_ca = st.number_input("Daily Capacity (t/day)", 50, 2000, 330, 10, key="daily_cap_ca")
            carbon_ca = st.number_input("Carbon Price (USD/t CO2)", 0.0, 300.0, 10.0, 5.0, key="carbon_ca")
            disc_ca = st.number_input("Discount Rate (%)", 0.0, 50.0, 8.0, 1.0, key="disc_ca")
        with colC2:
            # Additional dynamic pricing or cost input options can be added here.
            pass

        ghg_bau_ca = compute_bau_ghg_tons(daily_cap_ca, ENHANCED_INPUT_DATA)
        ghg_proj_ca = compute_project_ghg_tons(daily_cap_ca, FEEDSTOCKS, ENHANCED_INPUT_DATA)
        ghg_red_ca = max(0, ghg_bau_ca - ghg_proj_ca)
        ENHANCED_INPUT_DATA["economics"]["carbon_credit_price_usd_per_t_co2"] = carbon_ca
        fee_mode = st.session_state.get("fee_mode", "Both")
        revs_ca = compute_revenue_pie(daily_cap_ca, FEEDSTOCKS, ENHANCED_INPUT_DATA, fee_mode=fee_mode)
        total_rev_ca = sum(revs_ca.values())
        base_opex = ENHANCED_INPUT_DATA["economics"]["base_opex_usd_per_year"]
        annual_net_ca = total_rev_ca - base_opex
        flows_ca = discounted_cash_flow_series(annual_net_ca,
                                               ENHANCED_INPUT_DATA["financing"]["project_duration_years"],
                                               disc_ca)
        capex = ENHANCED_INPUT_DATA["economics"]["capex_usd"]
        npv_ca = sum(flows_ca) - capex

        if IRR_AVAILABLE:
            irr_ca = nf.irr([-capex] + [annual_net_ca] * ENHANCED_INPUT_DATA["financing"]["project_duration_years"]) * 100.0
        else:
            irr_ca = None

        cost_dict = {
            "Daily Capacity (t/day)": daily_cap_ca,
            "GHG BAU (t)": ghg_bau_ca,
            "GHG Project (t)": ghg_proj_ca,
            "GHG Reduction (t)": ghg_red_ca,
            "Annual Net (USD)": annual_net_ca,
            "NPV (USD)": npv_ca,
            "IRR (%)": irr_ca
        }
        df_cost = pd.DataFrame([cost_dict])
        st.dataframe(df_cost)
        st.session_state["cost_analysis_df"] = df_cost
        show_footer()
    
    ###########################################################################
    # TAB 6: CONCLUSIONS & PDF REPORT
    ###########################################################################
    with tabs[6]:
        st.subheader("Conclusions & Next Steps")
        st.markdown("""
**Summary:**
- Comparison of BAU vs. Project GHG emissions.
- Detailed scenario analysis (single and multi-scenario).
- Year-by-year and Monte Carlo simulations to assess uncertainties.
- Comprehensive cost analysis (NPV, IRR).
- Advanced visualizations using Plotly.
- Dynamic market trends incorporated in the analysis.
- Environmental metrics (water usage, particulate emissions).
- Secure usage exclusively for Igud Arim.

---
""")
        scenario_df = st.session_state.get("scenario_df")
        yearby_df = st.session_state.get("yearbyyear_df")
        cost_df = st.session_state.get("cost_analysis_df")
        mc_df = st.session_state.get("monte_carlo_df")
        
        if is_none_or_empty(scenario_df) and is_none_or_empty(yearby_df) and is_none_or_empty(cost_df) and is_none_or_empty(mc_df):
            st.info("No scenario DataFrames found. Please run the other analyses first.")
        else:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, "Gasification Feasibility Summary", ln=True, align="C")
            pdf.ln(5)
            pdf.set_font("Arial", "", 12)
            
            def add_df_to_pdf(title_str, df):
                if df is None or df.empty:
                    return
                pdf.set_font("Arial", "B", 12)
                pdf.cell(0, 8, title_str, ln=True)
                pdf.set_font("Arial", "", 11)
                pdf.ln(2)
                for line in df.to_string(index=False).split("\n"):
                    pdf.cell(0, 5, line, ln=True)
                pdf.ln(5)
            
            add_df_to_pdf("=== SCENARIO ANALYSIS RESULTS ===", scenario_df)
            add_df_to_pdf("=== YEAR-BY-YEAR ANALYSIS RESULTS ===", yearby_df)
            add_df_to_pdf("=== COST ANALYSIS RESULTS ===", cost_df)
            add_df_to_pdf("=== MONTE CARLO RESULTS ===", mc_df)
            
            pdf_bytes = pdf.output(dest="S").encode("latin-1")
            st.download_button(
                label="Download PDF Report",
                data=pdf_bytes,
                file_name="gasification_report_enhanced.pdf",
                mime="application/pdf",
                key="download_pdf"
            )
        show_footer()

if __name__ == "__main__":
    main()
