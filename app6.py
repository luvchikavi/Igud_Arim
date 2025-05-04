import copy
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from fpdf import FPDF

# Optional financial library
try:
    import numpy_financial as nf
    IRR_AVAILABLE = True
except ImportError:
    IRR_AVAILABLE = False

# ---------------------------
# 1. SIMPLE AUTHENTICATION
# ---------------------------
def authenticate():
    st.title("Gasification Feasibility Tool - Secure Login")
    secret_password = "IgudArim2025"  # For Igud Arim only
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
# 2. BASELINE DATA
# ---------------------------
# Original defaults
BASELINE_DATA = {
    "facility": {
        "daily_capacity_tons": 330,
        "operational_days": 300,
        "ash_content_pct": 5,
        "electricity_generation_kwh_per_ton": 550,  # For revenue generation only
        # We add new parameters for the alternative (syngas-based) approach
        "own_use_kwh_per_ton": 100.0,       # The client mentioned ~100 kWh/ton
        "co2_factor_kg_per_kwh": 0.4        # 0.4 kg CO₂ per kWh for syngas-based power
    },
    "economics": {
        "capex_usd": 100_000_000,
        "base_opex_usd_per_year": 6_000_000,  
        "carbon_credit_price_usd_per_t_co2": 10.0,
        "electricity_sales_price_usd_per_kwh": 0.11,
        "tipping_fee_usd_per_ton": 90.0,
        "discount_rate_pct": 8.0,
    },
    "financing": {
        "project_duration_years": 20,
    },
    "ghg_baseline": {
        # Original defaults:
        "landfill_emissions_kg_per_ton": 1721,  # Old method
        "facility_emissions_kg_per_ton": 566.5, # Old method
        # Also define the "alternative" recommended by client
        "landfill_emissions_alternative_kg_per_ton": 1000
    },
    "trucking": {
        "truck_capacity_tons": 15,          # Original
        "distance_to_landfill_km": 400,
        "emission_factor_kgco2_per_km": 2.26
    },
    "feedstocks": {
        "RDF_fraction": 0.70
    }
}

# A copy of BASELINE_DATA that we'll override in the UI
override_data = copy.deepcopy(BASELINE_DATA)

# FEEDSTOCK definitions for electricity & facility GHG (original approach).
FEEDSTOCKS = {
    "RDF": {
        "electricity_kwh_per_ton": 550,
        "ghg_facility_kg_per_ton": 566.5
    },
    "GreenWaste": {
        "electricity_kwh_per_ton": 550,
        "ghg_facility_kg_per_ton": 566.5
    }
}

# Yearly feedstock data (for the Year-by-Year tab)
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

def flatten_dict(d, parent_key="", sep="."):
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items

def format_numbers(df, fmt="{:,.2f}"):
    if df is None or df.empty:
        return df
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    return df.style.format({col: fmt for col in numeric_cols})

def compute_bau_ghg_tons(daily_capacity, data):
    """
    Computes 'Business As Usual' GHG: 
    1) Landfill emissions from the waste
    2) Trucking to landfill
    """
    days = data["facility"]["operational_days"]
    annual_waste = daily_capacity * days

    # Landfill Emission Factor
    # We will assume the user might have chosen either the original 1721, the alternative 1000,
    # or a custom override in override_data. 
    landfill_ef = data["ghg_baseline"]["landfill_emissions_kg_per_ton"]  # final chosen override

    ghg_landfill_kg = landfill_ef * annual_waste

    # Trucking
    truck_capacity = data["trucking"]["truck_capacity_tons"]
    distance = data["trucking"]["distance_to_landfill_km"]
    ef_truck = data["trucking"]["emission_factor_kgco2_per_km"]
    trucks_per_year = annual_waste / truck_capacity
    ghg_trucking_kg = trucks_per_year * distance * ef_truck

    return (ghg_landfill_kg + ghg_trucking_kg) / 1000.0

def compute_project_ghg_tons(daily_capacity, feedstocks, data):
    """
    Computes GHG from the project scenario:
    1) Facility emissions
    2) Local trucking (20 km assumed in original)
    """
    days = data["facility"]["operational_days"]
    annual_waste = daily_capacity * days

    # Here we handle two possible approaches to facility emissions:
    #  - Original "fixed" approach from feedstocks (566.5 kg/ton) 
    #    for each feedstock fraction.
    #  - A consumption-based approach: 
    #    (100 kWh/ton) * (0.4 kg CO₂/kWh) = 40 kg CO₂/ton
    #    or any user-defined override.
    # 
    # We'll check a new key we store in data["ghg_baseline"]["facility_emission_method"]
    # If "original", we sum feedstocks. Otherwise, we do consumption-based.

    facility_emission_method = data["ghg_baseline"].get("facility_emission_method", "original")

    if facility_emission_method == "original":
        # Original approach: partial for RDF & GreenWaste
        rdf_frac = data["feedstocks"]["RDF_fraction"]
        green_frac = 1.0 - rdf_frac
        total_fac_ghg_kg = 0.0
        for ftype, finfo in feedstocks.items():
            if ftype == "RDF":
                portion = annual_waste * rdf_frac
            else:
                portion = annual_waste * green_frac
            ghg_fac = finfo["ghg_facility_kg_per_ton"]  # e.g. 566.5
            total_fac_ghg_kg += ghg_fac * portion
    elif facility_emission_method == "consumption":
        # Calculation approach
        kwh_per_ton = data["facility"]["own_use_kwh_per_ton"]
        co2_factor = data["facility"]["co2_factor_kg_per_kwh"]
        total_fac_ghg_kg = annual_waste * kwh_per_ton * co2_factor
    else:
        # Potentially a fixed custom user override, stored in "facility_emissions_kg_per_ton" directly.
        # If user picks "custom_fixed", we rely on the single "facility_emissions_kg_per_ton"
        # from override_data
        custom_factor = data["ghg_baseline"]["facility_emissions_kg_per_ton"]
        total_fac_ghg_kg = custom_factor * annual_waste

    # Local trucking (20 km) remains the same logic
    local_distance_km = 20.0
    ef_truck = data["trucking"]["emission_factor_kgco2_per_km"]
    truck_capacity = data["trucking"]["truck_capacity_tons"]
    trucks_per_year = annual_waste / truck_capacity
    ghg_local_kg = trucks_per_year * local_distance_km * ef_truck

    return (total_fac_ghg_kg + ghg_local_kg) / 1000.0

def compute_revenue_pie(daily_capacity, feedstocks, data):
    days = data["facility"]["operational_days"]
    annual_waste = daily_capacity * days
    rdf_frac = data["feedstocks"]["RDF_fraction"]
    green_frac = 1.0 - rdf_frac

    # Electricity revenue uses feedstock electricity_kwh_per_ton from FEEDSTOCKS
    total_kwh = 0.0
    for ftype, finfo in feedstocks.items():
        if ftype == "RDF":
            portion = annual_waste * rdf_frac
        else:
            portion = annual_waste * green_frac
        total_kwh += portion * finfo["electricity_kwh_per_ton"]

    elec_price = data["economics"]["electricity_sales_price_usd_per_kwh"]
    elec_rev = total_kwh * elec_price

    ghg_bau = compute_bau_ghg_tons(daily_capacity, data)
    ghg_proj = compute_project_ghg_tons(daily_capacity, feedstocks, data)
    ghg_reduction_tons = max(0, ghg_bau - ghg_proj)
    carbon_price = data["economics"]["carbon_credit_price_usd_per_t_co2"]
    carbon_rev = ghg_reduction_tons * carbon_price

    tipping_fee_usd_per_ton = data["economics"]["tipping_fee_usd_per_ton"]
    tipping_rev = annual_waste * tipping_fee_usd_per_ton

    return {
        "Carbon": carbon_rev,
        "Electricity": elec_rev,
        "Tipping": tipping_rev
    }

def discounted_cash_flow_series(annual_net, project_life, discount_rate):
    r = discount_rate / 100.0
    return [annual_net / ((1 + r)**y) for y in range(1, project_life+1)]

def compute_npv_irr_roi(daily_capacity, data):
    base_opex = data["economics"]["base_opex_usd_per_year"]
    capex = data["economics"]["capex_usd"]
    discount_rate = data["economics"]["discount_rate_pct"]
    project_life = data["financing"]["project_duration_years"]

    ghg_bau = compute_bau_ghg_tons(daily_capacity, data)
    ghg_proj = compute_project_ghg_tons(daily_capacity, FEEDSTOCKS, data)
    ghg_reduction = max(0, ghg_bau - ghg_proj)

    revs = compute_revenue_pie(daily_capacity, FEEDSTOCKS, data)
    carbon_rev = revs["Carbon"]
    elec_rev = revs["Electricity"]
    tipping_rev = revs["Tipping"]
    total_rev = carbon_rev + elec_rev + tipping_rev
    annual_net = total_rev - base_opex

    flows = discounted_cash_flow_series(annual_net, project_life, discount_rate)
    npv = sum(flows) - capex

    if IRR_AVAILABLE:
        irr = nf.irr([-capex] + [annual_net]*project_life) * 100.0
    else:
        irr = None

    roi = (npv / capex)*100.0 if capex else None

    return {
        "GHG Reduction": ghg_reduction,
        "Carbon Revenue": carbon_rev,
        "Electricity Revenue": elec_rev,
        "Tipping Revenue": tipping_rev,
        "NPV": npv,
        "IRR": irr,
        "ROI": roi
    }

def is_none_or_empty(df):
    return (df is None) or (df.empty)

# ---------------------------
# 4. MAIN APPLICATION
# ---------------------------
def main():
    if "saved_scenarios" not in st.session_state:
        st.session_state["saved_scenarios"] = []

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
                st.image("oporto_logo.png", use_container_width=True)
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
        st.markdown("### Baseline (Original) Input Parameters (Constant)")
        flat_baseline = flatten_dict(BASELINE_DATA)
        df_base = pd.DataFrame(list(flat_baseline.items()), columns=["Parameter", "Value"])
        st.dataframe(df_base)

        st.markdown("---")
        st.markdown("### Override Parameters (Not affecting the baseline)")

        # 1) Daily capacity
        daily_cap_choice = st.number_input(
            "Daily Capacity (t/day)",
            min_value=50,
            max_value=2000,
            value=override_data["facility"]["daily_capacity_tons"],
            step=10
        )
        override_data["facility"]["daily_capacity_tons"] = daily_cap_choice

        # 2) Ash content
        ash_choice = st.number_input(
            "Ash Content (%)",
            min_value=0.0,
            max_value=50.0,
            value=float(override_data["facility"]["ash_content_pct"]),
            step=1.0
        )
        override_data["facility"]["ash_content_pct"] = ash_choice

        # 3) Carbon price
        carbon_choice = st.number_input(
            "Carbon Price (USD/t CO2)",
            min_value=0.0,
            max_value=300.0,
            value=override_data["economics"]["carbon_credit_price_usd_per_t_co2"],
            step=5.0
        )
        override_data["economics"]["carbon_credit_price_usd_per_t_co2"] = carbon_choice

        # 4) Discount rate
        disc_choice = st.number_input(
            "Discount Rate (%)",
            min_value=0.0,
            max_value=50.0,
            value=override_data["economics"]["discount_rate_pct"],
            step=1.0
        )
        override_data["economics"]["discount_rate_pct"] = disc_choice

        # 5) RDF fraction
        rdf_frac_choice = st.slider(
            "RDF Fraction",
            min_value=0.0,
            max_value=1.0,
            value=override_data["feedstocks"]["RDF_fraction"],
            step=0.05
        )
        override_data["feedstocks"]["RDF_fraction"] = rdf_frac_choice

        st.markdown("### Landfill Emission Factor (BAU)")
        landfill_choice = st.radio(
            "Select Landfill Emission Factor",
            ("Original (1721)", "EPA-based (1000)", "Custom"),
            index=0
        )
        if landfill_choice == "Original (1721)":
            override_data["ghg_baseline"]["landfill_emissions_kg_per_ton"] = 1721
        elif landfill_choice == "EPA-based (1000)":
            override_data["ghg_baseline"]["landfill_emissions_kg_per_ton"] = 1000
        else:
            custom_landfill = st.number_input(
                "Custom Landfill Emissions (kg CO2/ton)",
                min_value=0,
                max_value=3000,
                value=int(override_data["ghg_baseline"]["landfill_emissions_kg_per_ton"]),
                step=50
            )
            override_data["ghg_baseline"]["landfill_emissions_kg_per_ton"] = custom_landfill

        st.markdown("### Facility Emissions (Project)")
        facility_emission_choice = st.radio(
            "Select Facility Emission Approach",
            ("Original (fixed 566.5)", "Consumption-based (100 kWh * 0.4)", "Custom Fixed"),
            index=0
        )
        if facility_emission_choice == "Original (fixed 566.5)":
            override_data["ghg_baseline"]["facility_emission_method"] = "original"
            # We'll keep the original 566.5 in the feedstocks or override
            override_data["ghg_baseline"]["facility_emissions_kg_per_ton"] = 566.5
        elif facility_emission_choice == "Consumption-based (100 kWh * 0.4)":
            override_data["ghg_baseline"]["facility_emission_method"] = "consumption"
            # Let the user override the consumption or factor if desired
            own_use_choice = st.number_input(
                "Facility Own-Use Electricity (kWh/ton)",
                min_value=0.0,
                max_value=200.0,
                value=override_data["facility"]["own_use_kwh_per_ton"],
                step=10.0
            )
            override_data["facility"]["own_use_kwh_per_ton"] = own_use_choice

            co2_factor_choice = st.number_input(
                "CO₂ Factor (kg CO2/kWh for syngas)",
                min_value=0.0,
                max_value=1.0,
                value=override_data["facility"]["co2_factor_kg_per_kwh"],
                step=0.05
            )
            override_data["facility"]["co2_factor_kg_per_kwh"] = co2_factor_choice
        else:
            override_data["ghg_baseline"]["facility_emission_method"] = "custom_fixed"
            custom_fac = st.number_input(
                "Custom Facility Emissions (kg CO2/ton)",
                min_value=0.0,
                max_value=2000.0,
                value=override_data["ghg_baseline"]["facility_emissions_kg_per_ton"],
                step=50.0
            )
            override_data["ghg_baseline"]["facility_emissions_kg_per_ton"] = custom_fac

        st.markdown("### Truck Capacity Options")
        truck_choice = st.radio(
            "Select Truck Capacity (tons)",
            ("Original 15", "18", "30", "Custom"),
            index=0
        )
        if truck_choice == "Original 15":
            override_data["trucking"]["truck_capacity_tons"] = 15
        elif truck_choice == "18":
            override_data["trucking"]["truck_capacity_tons"] = 18
        elif truck_choice == "30":
            override_data["trucking"]["truck_capacity_tons"] = 30
        else:
            custom_truck = st.number_input(
                "Custom Truck Capacity (tons)",
                min_value=5,
                max_value=60,
                value=override_data["trucking"]["truck_capacity_tons"],
                step=1
            )
            override_data["trucking"]["truck_capacity_tons"] = custom_truck

        # 8) Electricity generation (kWh/ton) (for revenue)
        elec_gen_choice = st.number_input(
            "Electricity Generation (kWh/ton) for Revenue",
            min_value=100,
            max_value=2000,
            value=override_data["facility"]["electricity_generation_kwh_per_ton"],
            step=50
        )
        override_data["facility"]["electricity_generation_kwh_per_ton"] = elec_gen_choice
        # Update feedstock references if needed for the original approach to revenue
        FEEDSTOCKS["RDF"]["electricity_kwh_per_ton"] = elec_gen_choice
        FEEDSTOCKS["GreenWaste"]["electricity_kwh_per_ton"] = elec_gen_choice

        # 9) Electricity price
        elec_price_choice = st.number_input(
            "Electricity Price (USD/kWh)",
            min_value=0.0,
            max_value=1.0,
            value=override_data["economics"]["electricity_sales_price_usd_per_kwh"],
            step=0.01
        )
        override_data["economics"]["electricity_sales_price_usd_per_kwh"] = elec_price_choice

        # 10) Tipping fee
        tipping_choice = st.number_input(
            "Tipping Fee (USD/ton)",
            min_value=0.0,
            max_value=500.0,
            value=override_data["economics"]["tipping_fee_usd_per_ton"],
            step=10.0
        )
        override_data["economics"]["tipping_fee_usd_per_ton"] = tipping_choice

        st.markdown("---")
        st.markdown("### Current Override Data (for New Scenarios)")

        flat_override = flatten_dict(override_data)
        df_override = pd.DataFrame(list(flat_override.items()), columns=["Parameter", "Value"])
        st.dataframe(df_override)

        st.markdown("---")
        st.markdown("### Save a New Scenario from These Overrides")
        scenario_name = st.text_input("Scenario Name", "")
        if st.button("Save Scenario"):
            daily_capacity = override_data["facility"]["daily_capacity_tons"]
            results = compute_npv_irr_roi(daily_capacity, override_data)
            scenario_dict = {
                "Scenario Name": scenario_name if scenario_name else f"Scenario {len(st.session_state['saved_scenarios']) + 1}",
                "GHG Reduction": results["GHG Reduction"],
                "Carbon Revenue": results["Carbon Revenue"],
                "Electricity Revenue": results["Electricity Revenue"],
                "Tipping Revenue": results["Tipping Revenue"],
                "NPV": results["NPV"],
                "IRR": results["IRR"],
                "ROI": results["ROI"]
            }
            st.session_state["saved_scenarios"].append(scenario_dict)
            st.success(f"Scenario '{scenario_dict['Scenario Name']}' saved successfully!")

        show_footer()

    ###########################################################################
    # TAB 2: SCENARIO ANALYSIS
    ###########################################################################
    with tabs[2]:
        st.subheader("Scenario Analysis")

        # 1) Compute the Baseline Scenario from BASELINE_DATA (unchanged!)
        baseline_capacity = BASELINE_DATA["facility"]["daily_capacity_tons"]
        baseline_results = compute_npv_irr_roi(baseline_capacity, BASELINE_DATA)
        baseline_dict = {
            "Scenario Name": "Baseline Scenario",
            "GHG Reduction": baseline_results["GHG Reduction"],
            "Carbon Revenue": baseline_results["Carbon Revenue"],
            "Electricity Revenue": baseline_results["Electricity Revenue"],
            "Tipping Revenue": baseline_results["Tipping Revenue"],
            "NPV": baseline_results["NPV"],
            "IRR": baseline_results["IRR"],
            "ROI": baseline_results["ROI"]
        }

        # 2) Combine the baseline with user-saved scenarios
        saved_scenarios = st.session_state["saved_scenarios"]
        combined_list = [baseline_dict] + saved_scenarios
        df_combined = pd.DataFrame(combined_list)

        st.markdown("### Comparison of Baseline & Saved Scenarios")
        if df_combined.empty:
            st.info("No scenarios to display.")
        else:
            st.dataframe(format_numbers(df_combined))

            # GHG Reduction bar chart
            fig_ghg = px.bar(
                df_combined,
                x="Scenario Name",
                y="GHG Reduction",
                title="GHG Reduction by Scenario",
                labels={"GHG Reduction": "GHG Reduction (t CO2eq)"},
            )
            st.plotly_chart(fig_ghg, use_container_width=True)

            # Revenue breakdown
            rev_melt = df_combined.melt(
                id_vars=["Scenario Name", "GHG Reduction", "NPV", "IRR", "ROI"],
                value_vars=["Carbon Revenue", "Electricity Revenue", "Tipping Revenue"],
                var_name="Revenue Type",
                value_name="Revenue USD"
            )
            fig_rev = px.bar(
                rev_melt,
                x="Scenario Name",
                y="Revenue USD",
                color="Revenue Type",
                title="Revenue Breakdown by Scenario (Stacked)",
            )
            st.plotly_chart(fig_rev, use_container_width=True)

        show_footer()
    
    ###########################################################################
    # TAB 3: YEAR-BY-YEAR ANALYSIS
    ###########################################################################
    with tabs[3]:
        st.subheader("Year-by-Year Analysis")
        disc_rate_ = st.number_input("Discount Rate (%)", 0.0, 50.0,
                                     override_data["economics"]["discount_rate_pct"], 0.5, key="disc_rate")
        cprice_ = st.number_input("Base Carbon Price (USD/t)", 0.0, 300.0,
                                  override_data["economics"]["carbon_credit_price_usd_per_t_co2"], 5.0, key="cprice")
        dynamic_trend = st.checkbox("Use Dynamic Market Trends", key="dynamic_trend")
        if dynamic_trend:
            carbon_trend = st.number_input("Annual Carbon Price Change (%)", -10.0, 10.0, 0.0, 0.5, key="carbon_trend")
            elec_trend = st.number_input("Annual Electricity Price Change (%)", -10.0, 10.0, 0.0, 0.5, key="elec_trend")
        else:
            carbon_trend = 0.0
            elec_trend = 0.0
        
        if st.button("Run Year-by-Year Analysis", key="run_yby"):
            old_disc = override_data["economics"]["discount_rate_pct"]
            old_cprice = override_data["economics"]["carbon_credit_price_usd_per_t_co2"]
            override_data["economics"]["discount_rate_pct"] = disc_rate_
            override_data["economics"]["carbon_credit_price_usd_per_t_co2"] = cprice_

            results_list = []
            for year in sorted(YEARLY_FEEDSTOCK_DATA.keys()):
                daily_tons = YEARLY_FEEDSTOCK_DATA[year]["daily_feedstock_tons"]
                
                adjusted_carbon = cprice_ * ((1 + carbon_trend/100)**(year-1))
                adjusted_elec = override_data["economics"]["electricity_sales_price_usd_per_kwh"] * ((1 + elec_trend/100)**(year-1))
                
                old_carbon = override_data["economics"]["carbon_credit_price_usd_per_t_co2"]
                old_elec = override_data["economics"]["electricity_sales_price_usd_per_kwh"]
                
                override_data["economics"]["carbon_credit_price_usd_per_t_co2"] = adjusted_carbon
                override_data["economics"]["electricity_sales_price_usd_per_kwh"] = adjusted_elec
                
                ghg_bau_ = compute_bau_ghg_tons(daily_tons, override_data)
                ghg_proj_ = compute_project_ghg_tons(daily_tons, FEEDSTOCKS, override_data)
                red_ = max(0, ghg_bau_ - ghg_proj_)
                revs_ = compute_revenue_pie(daily_tons, FEEDSTOCKS, override_data)
                total_rev = sum(revs_.values())
                base_opex = override_data["economics"]["base_opex_usd_per_year"]
                net_cf = total_rev - base_opex

                override_data["economics"]["carbon_credit_price_usd_per_t_co2"] = old_carbon
                override_data["economics"]["electricity_sales_price_usd_per_kwh"] = old_elec

                results_list.append({
                    "Year": year,
                    "Daily Feedstock (t/day)": daily_tons,
                    "GHG BAU (t)": ghg_bau_,
                    "GHG Project (t)": ghg_proj_,
                    "GHG Reduction (t)": red_,
                    "Revenue Carbon": revs_["Carbon"],
                    "Revenue Elec": revs_["Electricity"],
                    "Revenue Tipping": revs_["Tipping"],
                    "Net CF (USD)": net_cf
                })

            override_data["economics"]["discount_rate_pct"] = old_disc
            override_data["economics"]["carbon_credit_price_usd_per_t_co2"] = old_cprice

            df_yby = pd.DataFrame(results_list)
            st.dataframe(format_numbers(df_yby))
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
            daily_cap = override_data["facility"]["daily_capacity_tons"]
            base_opex = override_data["economics"]["base_opex_usd_per_year"]
            
            old_carbon = override_data["economics"]["carbon_credit_price_usd_per_t_co2"]
            old_elec   = override_data["economics"]["electricity_sales_price_usd_per_kwh"]
            
            for i in range(int(runs_)):
                cp_sample = np.random.normal(carbon_mean_, carbon_std_)
                if cp_sample < 0:
                    cp_sample = 0
                override_data["economics"]["carbon_credit_price_usd_per_t_co2"] = cp_sample
                
                ep_sample = np.random.normal(elec_price_mean, elec_price_std)
                if ep_sample < 0:
                    ep_sample = 0
                override_data["economics"]["electricity_sales_price_usd_per_kwh"] = ep_sample
                
                revs_ = compute_revenue_pie(daily_cap, FEEDSTOCKS, override_data)
                net_annual = sum(revs_.values()) - base_opex
                mc_results.append(net_annual)
            
            override_data["economics"]["carbon_credit_price_usd_per_t_co2"] = old_carbon
            override_data["economics"]["electricity_sales_price_usd_per_kwh"] = old_elec

            df_mc = pd.DataFrame({"Annual Net Cash Flow (USD)": mc_results})
            st.write("### Monte Carlo Simulation Results Summary")
            st.write(format_numbers(df_mc))
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
            pass

        # Temporarily override just for cost analysis
        temp_data = copy.deepcopy(override_data)
        temp_data["facility"]["daily_capacity_tons"] = daily_cap_ca
        temp_data["economics"]["carbon_credit_price_usd_per_t_co2"] = carbon_ca
        temp_data["economics"]["discount_rate_pct"] = disc_ca

        results_dict = compute_npv_irr_roi(daily_cap_ca, temp_data)
        ghg_bau_ca = compute_bau_ghg_tons(daily_cap_ca, temp_data)
        ghg_proj_ca = compute_project_ghg_tons(daily_cap_ca, FEEDSTOCKS, temp_data)
        ghg_red_ca = results_dict["GHG Reduction"]

        cost_dict = {
            "Daily Capacity (t/day)": daily_cap_ca,
            "GHG BAU (t)": ghg_bau_ca,
            "GHG Project (t)": ghg_proj_ca,
            "GHG Reduction (t)": ghg_red_ca,
            "Carbon Revenue (USD)": results_dict["Carbon Revenue"],
            "Electricity Revenue (USD)": results_dict["Electricity Revenue"],
            "Tipping Revenue (USD)": results_dict["Tipping Revenue"],
            "NPV (USD)": results_dict["NPV"],
            "IRR (%)": results_dict["IRR"],
            "ROI (%)": results_dict["ROI"]
        }
        df_cost = pd.DataFrame([cost_dict])
        st.dataframe(format_numbers(df_cost))
        st.session_state["cost_analysis_df"] = df_cost
        show_footer()
    
    ###########################################################################
    # TAB 6: CONCLUSIONS & PDF REPORT
    ###########################################################################
    with tabs[6]:
        st.subheader("Conclusions & Next Steps")
        st.markdown("""
**Summary:**
1. **Baseline Scenario** uses the original, unchanging parameters.
2. **Override Data** is separate and does *not* affect the Baseline.
3. You can save multiple scenarios derived from your overrides. 
4. The **Scenario Analysis** tab compares the constant Baseline to all saved scenarios.
5. Year-by-year, Monte Carlo, and Cost Analysis use your override data.
""")

        scenario_df = None
        if "saved_scenarios" in st.session_state and st.session_state["saved_scenarios"]:
            scenario_df = pd.DataFrame(st.session_state["saved_scenarios"])
        yearby_df = st.session_state.get("yearbyyear_df")
        cost_df = st.session_state.get("cost_analysis_df")
        mc_df = st.session_state.get("monte_carlo_df")
        
        # Make a combined table for PDF if needed
        baseline_results = compute_npv_irr_roi(BASELINE_DATA["facility"]["daily_capacity_tons"], BASELINE_DATA)
        baseline_dict = {
            "Scenario Name": "Baseline Scenario",
            "GHG Reduction": baseline_results["GHG Reduction"],
            "Carbon Revenue": baseline_results["Carbon Revenue"],
            "Electricity Revenue": baseline_results["Electricity Revenue"],
            "Tipping Revenue": baseline_results["Tipping Revenue"],
            "NPV": baseline_results["NPV"],
            "IRR": baseline_results["IRR"],
            "ROI": baseline_results["ROI"]
        }
        df_baseline_pdf = pd.DataFrame([baseline_dict])

        if scenario_df is not None and not scenario_df.empty:
            df_scen_pdf = pd.concat([df_baseline_pdf, scenario_df], ignore_index=True)
        else:
            df_scen_pdf = df_baseline_pdf
        
        if is_none_or_empty(df_scen_pdf) and is_none_or_empty(yearby_df) and is_none_or_empty(cost_df) and is_none_or_empty(mc_df):
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
            
            add_df_to_pdf("=== SCENARIO ANALYSIS RESULTS ===", df_scen_pdf)
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
