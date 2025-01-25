import streamlit as st
import pandas as pd
import numpy as np
import math

# (Optional) If you want to use an IRR function:
# pip install numpy-financial
try:
    import numpy_financial as nf
    IRR_AVAILABLE = True
except ImportError:
    IRR_AVAILABLE = False

###############################################################################
# 1. ENHANCED GLOBAL INPUT DATA
###############################################################################
input_data = {
    "facility": {
        "annual_capacity_tons": 100_000,  # integer
        "daily_capacity_tons": 330,       # default daily capacity
        "operational_days": 300,
        "gasification_temp_c": 700,
        "energy_consumption_kwh_per_ton": 75,
        "electricity_generation_kwh_per_ton": 800,
        "waste_moisture_pct": 25,
        "ash_content_pct": 5,
        "max_feedstock_tons_day": 500  # e.g., we cannot process more than 500 t/day
    },
    "economics": {
        "capex_usd": 100_000_000,
        "base_opex_usd_per_year": 6_000_000,
        "opex_scaling_factor": 0.10, 
        "carbon_cert_one_time_cost_usd": 220_000,
        "carbon_cert_annual_fee_usd": 5_500,
        "carbon_credit_price_usd_per_t_co2": 10.0,
        "electricity_sales_price_usd_per_kwh": 0.11,
        "transport_savings_usd_per_ton": 36.30,
        "municipal_treatment_cost_usd_per_ton": 114.0
    },
    "financing": {
        "project_duration_years": 20,
        "tax_incentives_pct": 0.30,
        "discount_rate_pct": 8.0
    },
    "ghg_baseline": {
        "landfill_emissions_kg_per_ton": 500,
        "facility_emissions_kg_per_ton": 200,
        "methane_factor_kg_co2eq_per_ton": 100  
    }
}

###############################################################################
# 2. ADVANCED HELPER FUNCTIONS
###############################################################################

def calculate_advanced_opex(daily_capacity, input_data):
    base_opex = input_data["economics"]["base_opex_usd_per_year"]
    scaling_factor = input_data["economics"]["opex_scaling_factor"]
    design_capacity = input_data["facility"]["annual_capacity_tons"] / input_data["facility"]["operational_days"]
    if daily_capacity <= design_capacity:
        return base_opex
    else:
        ratio_over = (daily_capacity - design_capacity) / design_capacity
        scaled_opex = base_opex * (1 + ratio_over * scaling_factor)
        return scaled_opex

def calculate_avoided_methane(annual_waste, methane_factor_kg_co2eq_per_ton):
    return annual_waste * methane_factor_kg_co2eq_per_ton

def calculate_ghg_reduction_advanced(
    landfill_emissions_kg_per_ton,
    facility_emissions_kg_per_ton,
    methane_factor_kg_co2eq_per_ton,
    annual_waste
):
    baseline = landfill_emissions_kg_per_ton * annual_waste
    meth_avoided = calculate_avoided_methane(annual_waste, methane_factor_kg_co2eq_per_ton)
    facility = facility_emissions_kg_per_ton * annual_waste
    return (baseline + meth_avoided) - facility

def calculate_carbon_revenue(ghg_reduction_kg_co2, carbon_price_usd_per_t_co2):
    ghg_reduction_t_co2 = ghg_reduction_kg_co2 / 1000.0
    return ghg_reduction_t_co2 * carbon_price_usd_per_t_co2

def calculate_electricity_revenue(annual_waste, electricity_generation_kwh_per_ton, electricity_price):
    total_kwh = annual_waste * electricity_generation_kwh_per_ton
    return total_kwh * electricity_price

def calculate_tipping_revenue(annual_waste, tipping_fee):
    return annual_waste * tipping_fee

def annual_cashflow(annual_waste, daily_capacity, input_data, user_params):
    ghg_reduction_kg_co2 = calculate_ghg_reduction_advanced(
        landfill_emissions_kg_per_ton = user_params["landfill_emissions"],
        facility_emissions_kg_per_ton = user_params["facility_emissions"],
        methane_factor_kg_co2eq_per_ton = input_data["ghg_baseline"]["methane_factor_kg_co2eq_per_ton"],
        annual_waste = annual_waste
    )
    carbon_rev = calculate_carbon_revenue(
        ghg_reduction_kg_co2,
        user_params["carbon_price"]
    )
    elec_rev = calculate_electricity_revenue(
        annual_waste,
        input_data["facility"]["electricity_generation_kwh_per_ton"],
        user_params["electricity_price"]
    )
    tip_rev = calculate_tipping_revenue(annual_waste, user_params["tipping_fee"])
    total_rev = carbon_rev + elec_rev + tip_rev
    adv_opex = calculate_advanced_opex(daily_capacity, input_data)
    net = total_rev - adv_opex

    return {
        "GHG_Reduction_kg": ghg_reduction_kg_co2,
        "Carbon_Revenue": carbon_rev,
        "Electricity_Revenue": elec_rev,
        "Tipping_Revenue": tip_rev,
        "Total_Revenue": total_rev,
        "OPEX": adv_opex,
        "Annual_Net": net
    }

def discounted_cash_flow_series(annual_net, project_life, discount_rate):
    r = discount_rate / 100.0
    flows = []
    for y in range(1, project_life+1):
        discounted = annual_net / ((1 + r)**y)
        flows.append(discounted)
    return flows

def compute_irr(discounted_flows, capex):
    if IRR_AVAILABLE:
        return None
    else:
        return None

###############################################################################
# 3. STREAMLIT APP
###############################################################################

def main():
    try:
        st.image("oporto_logo.png", width=200)
    except:
        st.write("*(Logo not found or could not load.)*")

    st.title("Enhanced Gasification Feasibility with Advanced OPEX & GHG Accounting")

    tabs = st.tabs(["Home", "Input Data", "Single Scenario", "Multi-Scenario", "Cost Analysis", "Conclusions"])

    ############################################################################
    # TAB 1: HOME
    ############################################################################
    with tabs[0]:
        st.subheader("Welcome")
        st.markdown("""
        This **enhanced** tool demonstrates:
        - **Advanced OPEX** modeling that scales with capacity.
        - **Avoided methane** emissions from landfills in GHG calculations.
        - **Constraints** on feedstock (e.g., max capacity).
        - A new **Cost Analysis** tab with ROI, IRR, and more.
        
        **Use the tabs** to explore the data, run single or multi-scenario analyses, 
        then see detailed cost metrics (ROI, IRR, NPV).
        """)

    ############################################################################
    # TAB 2: INPUT DATA
    ############################################################################
    with tabs[1]:
        st.subheader("Base Input Data")

        def flatten_dict(d, parent_key="", sep="."):
            items = []
            for k,v in d.items():
                new_key = parent_key + sep + k if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)

        flat_input = flatten_dict(input_data)
        df_input = pd.DataFrame(list(flat_input.items()), columns=["Parameter", "Value"])
        st.dataframe(df_input, use_container_width=True)
        st.info("These are the default or assumed values for advanced OPEX, GHG, etc.")

    ############################################################################
    # TAB 3: SINGLE SCENARIO
    ############################################################################
    with tabs[2]:
        st.subheader("Single Scenario Analysis")
        st.write("""
        Adjust daily capacity and other parameters to see advanced OPEX & 
        avoided methane GHG calculations in action.
        """)

        col1, col2 = st.columns(2)
        max_feedstock = input_data["facility"]["max_feedstock_tons_day"]

        with col1:
            daily_capacity = st.number_input(
                "Daily Capacity (tons/day)",
                min_value=50,
                max_value=2000,
                value=int(input_data["facility"]["daily_capacity_tons"]),
                step=10,
                format="%d",
                key="single_daily_capacity"  # <-- KEY ADDED
            )
            if daily_capacity > max_feedstock:
                st.warning(f"Exceeding max feedstock constraint of {max_feedstock} t/day. We'll still compute, but it may be infeasible.")

            landfill_emissions = st.number_input(
                "Landfill Emissions (kg CO2/ton)",
                0.0, 2000.0,
                float(input_data["ghg_baseline"]["landfill_emissions_kg_per_ton"]),
                50.0,
                key="single_landfill_emissions"  # <-- KEY ADDED
            )
            facility_emissions = st.number_input(
                "Facility Emissions (kg CO2/ton)",
                0.0, 1000.0,
                float(input_data["ghg_baseline"]["facility_emissions_kg_per_ton"]),
                50.0,
                key="single_facility_emissions"  # <-- KEY ADDED
            )
            methane_factor = st.number_input(
                "Methane Factor (kg CO2e/ton in landfill)",
                0.0, 1000.0,
                float(input_data["ghg_baseline"]["methane_factor_kg_co2eq_per_ton"]),
                25.0,
                key="single_methane_factor"  # <-- KEY ADDED
            )

        with col2:
            carbon_price = st.number_input(
                "Carbon Price (USD/ton CO2)",
                0.0, 300.0,
                float(input_data["economics"]["carbon_credit_price_usd_per_t_co2"]),
                5.0,
                key="single_carbon_price"  # <-- KEY ADDED
            )
            electricity_price = st.number_input(
                "Electricity Price (USD/kWh)",
                0.00, 1.00,
                float(input_data["economics"]["electricity_sales_price_usd_per_kwh"]),
                0.01,
                key="single_electricity_price"  # <-- KEY ADDED
            )
            tipping_fee = st.number_input(
                "Tipping Fee (USD/ton)",
                0.0, 300.0,
                90.0, 5.0,
                key="single_tipping_fee"  # <-- KEY ADDED
            )

        operational_days = input_data["facility"]["operational_days"]
        annual_waste = daily_capacity * operational_days

        user_params = {
            "landfill_emissions": landfill_emissions,
            "facility_emissions": facility_emissions,
            "carbon_price": carbon_price,
            "electricity_price": electricity_price,
            "tipping_fee": tipping_fee
        }

        results = annual_cashflow(annual_waste, daily_capacity, input_data, user_params)
        single_dict = {
            "Annual Waste (tons)": [annual_waste],
            "GHG Reduction (kg CO2eq/year)": [results["GHG_Reduction_kg"]],
            "Carbon Revenue (USD/yr)": [results["Carbon_Revenue"]],
            "Electricity Revenue (USD/yr)": [results["Electricity_Revenue"]],
            "Tipping Revenue (USD/yr)": [results["Tipping_Revenue"]],
            "Total Revenue (USD/yr)": [results["Total_Revenue"]],
            "OPEX (USD/yr)": [results["OPEX"]],
            "Annual Net (USD/yr)": [results["Annual_Net"]]
        }
        st.table(pd.DataFrame(single_dict).style.format("{:,.2f}"))

        if results["Annual_Net"] >= 0:
            st.success("Project is profitable on an annual basis!")
        else:
            st.warning("Project is not profitable on an annual basis with these inputs.")

    ############################################################################
    # TAB 4: MULTI-SCENARIO (What-If)
    ############################################################################
    with tabs[3]:
        st.subheader("Multi-Scenario Analysis")
        st.write("Explore a range of daily capacities (and optionally carbon prices) with advanced OPEX + GHG logic.")

        col1, col2 = st.columns(2)
        with col1:
            min_cap = st.number_input("Min Capacity (t/day)", 50, 2000, 100, 10, key="multi_min_cap")  # <-- KEY ADDED
            max_cap = st.number_input("Max Capacity (t/day)", 50, 2000, 600, 10, key="multi_max_cap") # <-- KEY ADDED
            step_cap = st.number_input("Step Capacity", 1, 500, 50, 10, key="multi_step_cap")         # <-- KEY ADDED
        with col2:
            min_c = st.number_input("Min Carbon (USD/t CO2)", 0.0, 300.0, 5.0, 5.0, key="multi_min_c") # <-- KEY ADDED
            max_c = st.number_input("Max Carbon (USD/t CO2)", 0.0, 300.0, 30.0, 5.0, key="multi_max_c")# <-- KEY ADDED
            step_c = st.number_input("Step Carbon", 1.0, 50.0, 5.0, 1.0, key="multi_step_c")           # <-- KEY ADDED

        st.write("Other fixed parameters (landfill/facility emissions, methane factor, etc.)")
        col3, col4 = st.columns(2)
        with col3:
            landfill_ms = st.number_input("Landfill Emissions Ms (kgCO2/ton)", 0.0, 2000.0, 500.0, 50.0, key="multi_landfill")  # <-- KEY ADDED
            facility_ms = st.number_input("Facility Emissions Ms (kgCO2/ton)", 0.0, 1000.0, 200.0, 50.0, key="multi_facility")  # <-- KEY ADDED
        with col4:
            methane_ms = st.number_input("Methane Factor Ms (kgCO2eq/ton)", 0.0, 1000.0, 100.0, 10.0, key="multi_methane")      # <-- KEY ADDED
            electricity_ms = st.number_input("Electricity Price Ms (USD/kWh)", 0.0, 1.0, 0.11, 0.01, key="multi_elec")          # <-- KEY ADDED

        tipping_ms = st.number_input("Tipping Fee Ms (USD/ton)", 0.0, 500.0, 90.0, 5.0, key="multi_tipping")                     # <-- KEY ADDED

        run_multi = st.button("Run Multi-Scenario", key="multi_run_button")  # <-- KEY ADDED
        if run_multi:
            capacities = range(min_cap, max_cap+1, step_cap)
            carbon_prices = np.arange(min_c, max_c+0.0001, step_c)
            op_days = input_data["facility"]["operational_days"]
            scenario_list = []
            for c in capacities:
                for cp in carbon_prices:
                    if c > input_data["facility"]["max_feedstock_tons_day"]:
                        feasible = False
                    else:
                        feasible = True
                    annual_waste = c * op_days
                    user_params = {
                        "landfill_emissions": landfill_ms,
                        "facility_emissions": facility_ms,
                        "carbon_price": cp,
                        "electricity_price": electricity_ms,
                        "tipping_fee": tipping_ms
                    }
                    result = annual_cashflow(annual_waste, c, input_data, user_params)
                    scenario_list.append({
                        "Daily Capacity (t/day)": c,
                        "Carbon Price (USD/t)": cp,
                        "Feasible?": "Yes" if feasible else "No",
                        "Annual Waste (t)": annual_waste,
                        "GHG Red. (kg/yr)": result["GHG_Reduction_kg"],
                        "Carbon Rev. (USD/yr)": result["Carbon_Revenue"],
                        "Elec Rev. (USD/yr)": result["Electricity_Revenue"],
                        "Tipping Rev. (USD/yr)": result["Tipping_Revenue"],
                        "Total Rev. (USD/yr)": result["Total_Revenue"],
                        "OPEX (USD/yr)": result["OPEX"],
                        "Annual Net (USD/yr)": result["Annual_Net"]
                    })

            df_scen = pd.DataFrame(scenario_list)
            st.write("### Multi-Scenario Results")
            st.dataframe(df_scen.style.format("{:,.2f}"), use_container_width=True)

            st.success("Multi-scenario run complete!")

    ############################################################################
    # TAB 5: COST ANALYSIS (ROI, IRR, etc.)
    ############################################################################
    with tabs[4]:
        st.subheader("Cost Analysis: ROI, IRR, and NPV")
        st.write("""
        This tab shows more advanced financial metrics like **ROI**, 
        **IRR**, and **NPV** for a single scenario or a small range of scenarios.
        """)

        colA, colB = st.columns(2)
        with colA:
            daily_capacity_ca = st.number_input("Daily Capacity (t/day, for cost analysis)",
                50, 2000, 330, 10, key="ca_daily_capacity")  # <-- KEY ADDED
            carbon_price_ca = st.number_input("Carbon Price (USD/ton CO2, cost analysis)",
                0.0, 300.0, 10.0, 5.0, key="ca_carbon_price")  # <-- KEY ADDED
            discount_ca = st.number_input("Discount Rate (%)", 
                0.0, 50.0, 8.0, 1.0, key="ca_discount")        # <-- KEY ADDED

        with colB:
            facility_ca = st.number_input("Facility Emissions (kgCO2/t)", 
                0.0, 1000.0, 200.0, 50.0, key="ca_facility")   # <-- KEY ADDED
            landfill_ca = st.number_input("Landfill Emissions (kgCO2/t)", 
                0.0, 2000.0, 500.0, 50.0, key="ca_landfill")   # <-- KEY ADDED
            methane_ca = st.number_input("Methane Factor (kgCO2eq/t)", 
                0.0, 1000.0, 100.0, 10.0, key="ca_methane")    # <-- KEY ADDED

        electricity_ca = st.number_input("Electricity Price (USD/kWh)", 
            0.0, 1.0, 0.11, 0.01, key="ca_elec")               # <-- KEY ADDED
        tipping_ca = st.number_input("Tipping Fee (USD/ton)", 
            0.0, 500.0, 90.0, 5.0, key="ca_tipping")           # <-- KEY ADDED

        operational_days = input_data["facility"]["operational_days"]
        annual_waste_ca = daily_capacity_ca * operational_days

        user_params_ca = {
            "landfill_emissions": landfill_ca,
            "facility_emissions": facility_ca,
            "carbon_price": carbon_price_ca,
            "electricity_price": electricity_ca,
            "tipping_fee": tipping_ca
        }

        result_ca = annual_cashflow(annual_waste_ca, daily_capacity_ca, input_data, user_params_ca)
        annual_net_ca = result_ca["Annual_Net"]

        project_life_ca = input_data["financing"]["project_duration_years"]
        flows = discounted_cash_flow_series(annual_net_ca, project_life_ca, discount_ca)
        capex = input_data["economics"]["capex_usd"]
        npv_project = sum(flows) - capex

        total_net_no_discount = annual_net_ca * project_life_ca
        roi = total_net_no_discount / capex * 100.0

        cash_flows = [-capex] + [annual_net_ca]*(project_life_ca)
        if IRR_AVAILABLE:
            irr_value = nf.irr(cash_flows) * 100.0
        else:
            irr_value = None

        cost_dict = {
            "Annual Waste (t)": annual_waste_ca,
            "GHG Reduction (kg CO2/yr)": result_ca["GHG_Reduction_kg"],
            "Annual Net (USD/yr)": annual_net_ca,
            "NPV (USD)": npv_project,
            "ROI (%)": roi,
            "IRR (%)": irr_value if irr_value is not None else "Install `numpy-financial` for IRR"
        }
        df_cost = pd.DataFrame([cost_dict])
        st.table(df_cost.style.format("{:,.2f}"))

        if npv_project > 0:
            st.success(f"Positive NPV => project is financially attractive. IRR={irr_value if irr_value else 'N/A'}")
        else:
            st.warning(f"Negative NPV => project might not be viable at these parameters. IRR={irr_value if irr_value else 'N/A'}")

    ############################################################################
    # TAB 6: CONCLUSIONS
    ############################################################################
    with tabs[5]:
        st.subheader("Conclusions & Next Steps")
        st.markdown("""
        **Summary**:
        - We introduced **advanced OPEX** scaling with capacity, 
          **avoided methane** in GHG accounting, 
          and a feedstock **constraint** on maximum capacity.
        - A new **Cost Analysis** tab shows ROI, IRR, and NPV for 
          deeper economic insights.

        **Future Enhancements**:
        1. More detailed year-by-year OPEX, maintenance, feedstock mix changes.
        2. Integration with **real** LCA (Life Cycle Assessment) tools or Aspen Plus modeling.
        3. More sophisticated IRR if different cash flows per year.

        **References**:
        - IPCC Guidelines for landfill methane factors.
        - Local regulations for emission limits or carbon offset calculations.
        - [NREL Gasification Publications](https://www.nrel.gov).

        ***Thank you for using the Enhanced Gasification Feasibility Tool!***
        """)

if __name__ == "__main__":
    main()
