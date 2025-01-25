import streamlit as st
import pandas as pd
import numpy as np
import math

# If you want advanced financial calculations (IRR, NPV), install numpy-financial:
# pip install numpy-financial
try:
    import numpy_financial as nf
    IRR_AVAILABLE = True
except ImportError:
    IRR_AVAILABLE = False

###############################################################################
# 1. COMBINED GLOBAL INPUT DATA
###############################################################################
# A single dictionary that holds both the "year-by-year" approach data
# and the "single-scenario" approach data.

ENHANCED_INPUT_DATA = {
    "facility": {
        # Single-year approach:
        "annual_capacity_tons": 100_000,    # For single scenario or multi-scenario
        "daily_capacity_tons": 330,
        "operational_days": 300,
        "gasification_temp_c": 700,
        "energy_consumption_kwh_per_ton": 75,
        "electricity_generation_kwh_per_ton": 800,
        "waste_moisture_pct": 25,
        "ash_content_pct": 5,
        "max_feedstock_tons_day": 500,      # max feasible daily feedstock
        # Year-by-year approach:
        "nominal_daily_capacity_tons": 300,  # "design" capacity for multi-year
    },
    "economics": {
        # Single-year approach:
        "capex_usd": 100_000_000,
        "base_opex_usd_per_year": 6_000_000,
        "opex_scaling_factor": 0.10,
        "carbon_cert_one_time_cost_usd": 220_000,
        "carbon_cert_annual_fee_usd": 5_500,
        "carbon_credit_price_usd_per_t_co2": 10.0,
        "electricity_sales_price_usd_per_kwh": 0.11,
        "transport_savings_usd_per_ton": 36.30,
        "municipal_treatment_cost_usd_per_ton": 114.0,
        # Year-by-year approach:
        "base_opex_usd_year1": 6_000_000,   # starting OPEX for year-by-year
        "inflation_rate_pct": 2.0,          # OPEX grows by 2% each year
        "maintenance_schedule": {
            5: 1_000_000,  # additional lumps at year 5
            10: 2_000_000  # additional lumps at year 10
        },
        "tipping_fee_usd_per_ton": 90.0,    # used in year-by-year approach
    },
    "financing": {
        # Single-year approach:
        "project_duration_years": 20,
        "tax_incentives_pct": 0.30,
        "discount_rate_pct": 8.0,
        # Year-by-year approach:
        "project_life": 20,  # same number, but used in the year-by-year
    },
    "ghg_baseline": {
        "landfill_emissions_kg_per_ton": 500,
        "facility_emissions_kg_per_ton": 200,
        "methane_factor_kg_co2eq_per_ton": 100
    }
}

###############################################################################
# 2. YEAR-BY-YEAR FEEDSTOCK DATA (for multi-year approach)
###############################################################################
# You might load this from CSV or Aspen results. For demonstration, we define it:

YEARLY_FEEDSTOCK_DATA = {
    1:  {"daily_feedstock_tons": 200, "capacity_factor": 0.80},
    2:  {"daily_feedstock_tons": 250, "capacity_factor": 0.85},
    3:  {"daily_feedstock_tons": 300, "capacity_factor": 0.90},
    4:  {"daily_feedstock_tons": 320, "capacity_factor": 0.95},
    5:  {"daily_feedstock_tons": 350, "capacity_factor": 1.00},
    # For years 6-20, assume stable
}
for y in range(6, 21):
    YEARLY_FEEDSTOCK_DATA[y] = {"daily_feedstock_tons": 350, "capacity_factor": 1.0}

###############################################################################
# 3. HELPER FUNCTIONS (Year-by-year approach)
###############################################################################
def integrate_lca_or_aspen_sim(year):
    """Placeholder: You'd query LCA/Aspen data for year-based emissions, etc."""
    pass

def compute_yearly_opex(year, base_opex_year1, inflation_rate_pct, maintenance_schedule):
    """OPEX in year n = base_opex_year1 * (1+inflation)^(n-1) + any maintenance lumps."""
    inflated_opex = base_opex_year1 * ((1 + inflation_rate_pct/100)**(year - 1))
    additional_maintenance = maintenance_schedule.get(year, 0.0)
    return inflated_opex + additional_maintenance

def compute_annual_results_for_year(year, base_input, feedstock_info, ghg_baseline):
    """Calculate net cash flow for a single year with variable feedstock/capacity."""
    daily_tons = feedstock_info["daily_feedstock_tons"]
    capacity_factor = feedstock_info["capacity_factor"]
    design_operational_days = 330
    actual_operational_days = design_operational_days * capacity_factor
    annual_waste = daily_tons * actual_operational_days

    # Baseline landfill + methane vs. facility emissions
    baseline_co2 = ghg_baseline["landfill_emissions_kg_per_ton"] * annual_waste
    methane_co2 = ghg_baseline["methane_factor_kg_co2eq_per_ton"] * annual_waste
    facility_co2 = ghg_baseline["facility_emissions_kg_per_ton"] * annual_waste
    ghg_reduction_kg = (baseline_co2 + methane_co2) - facility_co2

    # Revenues
    econ = base_input["economics"]
    carbon_revenue = (ghg_reduction_kg/1000.0) * econ["carbon_credit_price_usd_per_t_co2"]
    electricity_revenue = annual_waste * econ["electricity_sales_price_usd_per_kwh"] \
        * base_input["facility"]["electricity_generation_kwh_per_ton"]
    tipping_revenue = annual_waste * econ["tipping_fee_usd_per_ton"]
    total_revenue = carbon_revenue + electricity_revenue + tipping_revenue

    # OPEX with inflation & maintenance
    opex_this_year = compute_yearly_opex(
        year,
        econ["base_opex_usd_year1"],
        econ["inflation_rate_pct"],
        econ["maintenance_schedule"]
    )
    net_cash_flow = total_revenue - opex_this_year

    return {
        "Year": year,
        "DailyTons": daily_tons,
        "CapacityFactor": capacity_factor,
        "AnnualWaste": annual_waste,
        "GHG_Reduction_kg": ghg_reduction_kg,
        "Carbon_Revenue": carbon_revenue,
        "Electricity_Revenue": electricity_revenue,
        "Tipping_Revenue": tipping_revenue,
        "Total_Revenue": total_revenue,
        "OPEX": opex_this_year,
        "NetCashFlow": net_cash_flow
    }

###############################################################################
# 4. HELPER FUNCTIONS (Single-year approach, multi-scenario)
###############################################################################
def calculate_advanced_opex(daily_capacity, input_data):
    """Scale OPEX if daily capacity > design capacity."""
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
    carbon_rev = calculate_carbon_revenue(ghg_reduction_kg_co2, user_params["carbon_price"])
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

###############################################################################
# 5. STREAMLIT APP
###############################################################################
def main():
    st.title("Unified Gasification Feasibility Tool (Single/Multi-Year Approaches)")

    # We create multiple tabs to handle each approach
    tabs = st.tabs([
        "Home",
        "Base Input Data",
        "Single Scenario",
        "Multi-Scenario",
        "Year-by-Year Model",
        "Year-by-Year Results",
        "Cost Analysis",
        "Conclusions"
    ])

    ############################################################################
    # TAB: HOME
    ############################################################################
    with tabs[0]:
        st.subheader("Welcome")
        st.markdown("""
        This **unified** tool combines:
        - **Single-year** advanced OPEX and GHG calculations,
        - **Multi-scenario** "what-if" daily capacity & carbon price,
        - **Year-by-year** approach (feedstock ramp-up, OPEX inflation, maintenance),
        - **Graphs & Cost Analysis** (NPV, IRR).

        You can explore:
        1. The **Base Input Data** (Tab 2)  
        2. **Single Scenario** (Tab 3) for a one-year feasibility snapshot  
        3. **Multi-Scenario** (Tab 4) for daily capacity & carbon price loops  
        4. A **Year-by-Year Model** (Tab 5) for multi-year feedstock ramp  
        5. **Year-by-Year Results** (Tab 6) with charts  
        6. **Cost Analysis** (Tab 7)  
        7. **Conclusions** (Tab 8)

        ---
        """)

    ############################################################################
    # TAB: BASE INPUT DATA
    ############################################################################
    with tabs[1]:
        st.subheader("Base Input Data")

        def flatten_dict(d, parent_key="", sep="."):
            items = []
            for k, v in d.items():
                k_str = str(k)
                new_key = parent_key + sep + k_str if parent_key else k_str
                
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)

        flat_input = flatten_dict(ENHANCED_INPUT_DATA)
        df_input = pd.DataFrame(list(flat_input.items()), columns=["Parameter", "Value"])
        st.dataframe(df_input, use_container_width=True)

        st.markdown("#### Year-by-Year Feedstock Data (Demo)")
        feedstock_df = pd.DataFrame([
            {"Year": y,
             "Daily Feedstock (tons)": YEARLY_FEEDSTOCK_DATA[y]["daily_feedstock_tons"],
             "Capacity Factor": YEARLY_FEEDSTOCK_DATA[y]["capacity_factor"]}
            for y in range(1, 1 + ENHANCED_INPUT_DATA["financing"]["project_life"])
        ])
        st.dataframe(feedstock_df, use_container_width=True)

    ############################################################################
    # TAB: SINGLE SCENARIO (One-year approach)
    ############################################################################
    with tabs[2]:
        st.subheader("Single-Year Scenario Analysis")
        st.write("""
        Adjust daily capacity, carbon price, etc., and see advanced OPEX & GHG calculations
        for **one** year.
        """)

        col1, col2 = st.columns(2)
        max_feedstock = ENHANCED_INPUT_DATA["facility"]["max_feedstock_tons_day"]

        with col1:
            daily_capacity = st.number_input(
                "Daily Capacity (tons/day)",
                min_value=50,
                max_value=2000,
                value=int(ENHANCED_INPUT_DATA["facility"]["daily_capacity_tons"]),
                step=10,
                format="%d",
                key="single_daily_capacity"
            )
            if daily_capacity > max_feedstock:
                st.warning(f"Exceeding max feedstock limit ({max_feedstock} t/day). Still computing, but may be infeasible.")

            landfill_emissions = st.number_input(
                "Landfill Emissions (kg CO2/ton)",
                0.0, 2000.0,
                float(ENHANCED_INPUT_DATA["ghg_baseline"]["landfill_emissions_kg_per_ton"]),
                50.0,
                key="single_landfill_emissions"
            )
            facility_emissions = st.number_input(
                "Facility Emissions (kg CO2/ton)",
                0.0, 1000.0,
                float(ENHANCED_INPUT_DATA["ghg_baseline"]["facility_emissions_kg_per_ton"]),
                50.0,
                key="single_facility_emissions"
            )
            methane_factor = st.number_input(
                "Methane Factor (kg CO2e/ton in landfill)",
                0.0, 1000.0,
                float(ENHANCED_INPUT_DATA["ghg_baseline"]["methane_factor_kg_co2eq_per_ton"]),
                25.0,
                key="single_methane_factor"
            )

        with col2:
            carbon_price = st.number_input(
                "Carbon Price (USD/ton CO2)",
                0.0, 300.0,
                float(ENHANCED_INPUT_DATA["economics"]["carbon_credit_price_usd_per_t_co2"]),
                5.0,
                key="single_carbon_price"
            )
            electricity_price = st.number_input(
                "Electricity Price (USD/kWh)",
                0.00, 1.00,
                float(ENHANCED_INPUT_DATA["economics"]["electricity_sales_price_usd_per_kwh"]),
                0.01,
                key="single_electricity_price"
            )
            tipping_fee = st.number_input(
                "Tipping Fee (USD/ton)",
                0.0, 300.0,
                90.0, 5.0,
                key="single_tipping_fee"
            )

        # Calculate
        operational_days = ENHANCED_INPUT_DATA["facility"]["operational_days"]
        annual_waste = daily_capacity * operational_days

        user_params = {
            "landfill_emissions": landfill_emissions,
            "facility_emissions": facility_emissions,
            "carbon_price": carbon_price,
            "electricity_price": electricity_price,
            "tipping_fee": tipping_fee
        }

        single_results = annual_cashflow(
            annual_waste, daily_capacity, ENHANCED_INPUT_DATA, user_params
        )
        single_dict = {
            "Annual Waste (tons)": [annual_waste],
            "GHG Reduction (kg CO2eq/year)": [single_results["GHG_Reduction_kg"]],
            "Carbon Revenue (USD/yr)": [single_results["Carbon_Revenue"]],
            "Electricity Revenue (USD/yr)": [single_results["Electricity_Revenue"]],
            "Tipping Revenue (USD/yr)": [single_results["Tipping_Revenue"]],
            "Total Revenue (USD/yr)": [single_results["Total_Revenue"]],
            "OPEX (USD/yr)": [single_results["OPEX"]],
            "Annual Net (USD/yr)": [single_results["Annual_Net"]]
        }

        st.table(pd.DataFrame(single_dict).style.format("{:,.2f}"))

        if single_results["Annual_Net"] >= 0:
            st.success("Project is profitable on an annual basis!")
        else:
            st.warning("Project is not profitable on an annual basis with these inputs.")

    ############################################################################
    # TAB: MULTI-SCENARIO (One-year approach)
    ############################################################################
    with tabs[3]:
        st.subheader("Multi-Scenario Analysis (One-Year What-If)")
        st.write("""
        Loop over daily capacities and carbon prices (or other parameters)
        to see how the annual results vary.
        """)

        col1, col2 = st.columns(2)
        with col1:
            min_cap = st.number_input("Min Capacity (t/day)", 50, 2000, 100, 10, key="multi_min_cap")
            max_cap = st.number_input("Max Capacity (t/day)", 50, 2000, 600, 10, key="multi_max_cap")
            step_cap = st.number_input("Step Capacity", 1, 500, 50, 10, key="multi_step_cap")
        with col2:
            min_c = st.number_input("Min Carbon (USD/t CO2)", 0.0, 300.0, 5.0, 5.0, key="multi_min_c")
            max_c = st.number_input("Max Carbon (USD/t CO2)", 0.0, 300.0, 30.0, 5.0, key="multi_max_c")
            step_c = st.number_input("Step Carbon", 1.0, 50.0, 5.0, 1.0, key="multi_step_c")

        st.write("Other fixed parameters for the multi-scenario run:")
        col3, col4 = st.columns(2)
        with col3:
            landfill_ms = st.number_input("Landfill Emissions Ms (kgCO2/ton)", 0.0, 2000.0, 500.0, 50.0, key="multi_landfill")
            facility_ms = st.number_input("Facility Emissions Ms (kgCO2/ton)", 0.0, 1000.0, 200.0, 50.0, key="multi_facility")
        with col4:
            methane_ms = st.number_input("Methane Factor Ms (kgCO2eq/ton)", 0.0, 1000.0, 100.0, 10.0, key="multi_methane")
            electricity_ms = st.number_input("Electricity Price Ms (USD/kWh)", 0.0, 1.0, 0.11, 0.01, key="multi_elec")

        tipping_ms = st.number_input("Tipping Fee Ms (USD/ton)", 0.0, 500.0, 90.0, 5.0, key="multi_tipping")

        run_multi = st.button("Run Multi-Scenario", key="multi_run_button")
        if run_multi:
            capacities = range(min_cap, max_cap+1, step_cap)
            carbon_prices = np.arange(min_c, max_c+0.0001, step_c)
            op_days = ENHANCED_INPUT_DATA["facility"]["operational_days"]
            scenario_list = []

            for c in capacities:
                feasible = True
                if c > ENHANCED_INPUT_DATA["facility"]["max_feedstock_tons_day"]:
                    feasible = False
                for cp in carbon_prices:
                    annual_waste = c * op_days
                    user_params = {
                        "landfill_emissions": landfill_ms,
                        "facility_emissions": facility_ms,
                        "carbon_price": cp,
                        "electricity_price": electricity_ms,
                        "tipping_fee": tipping_ms
                    }
                    result = annual_cashflow(annual_waste, c, ENHANCED_INPUT_DATA, user_params)
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
            st.write("### Multi-Scenario Results (One Year)")
            st.dataframe(df_scen.style.format("{:,.2f}"), use_container_width=True)

            st.success("Multi-scenario run complete!")

    ############################################################################
    # TAB: YEAR-BY-YEAR MODEL
    ############################################################################
    with tabs[4]:
        st.subheader("Year-by-Year Model Inputs & Calculation")
        st.write("""
        In this approach, feedstock capacity ramps up annually,
        OPEX inflates each year, and we can add maintenance lumps.
        """)

        discount_rate = st.number_input("Discount Rate (%)", 0.0, 50.0,
            ENHANCED_INPUT_DATA["financing"]["discount_rate_pct"], 0.5,
            key="yearly_discount")
        carbon_price_yb = st.number_input("Carbon Price (USD/ton CO2)", 0.0, 300.0,
            ENHANCED_INPUT_DATA["economics"]["carbon_credit_price_usd_per_t_co2"], 5.0,
            key="yearly_carbon")
        electricity_yb = st.number_input("Electricity Price (USD/kWh)", 0.0, 1.0,
            ENHANCED_INPUT_DATA["economics"]["electricity_sales_price_usd_per_kwh"], 0.01,
            key="yearly_electricity")
        tipping_yb = st.number_input("Tipping Fee (USD/ton)", 0.0, 300.0,
            ENHANCED_INPUT_DATA["economics"]["tipping_fee_usd_per_ton"], 5.0,
            key="yearly_tipping")

        run_yearly = st.button("Run Year-by-Year Simulation")
        if run_yearly:
            # Update the dictionary with new user inputs
            ENHANCED_INPUT_DATA["financing"]["discount_rate_pct"] = discount_rate
            ENHANCED_INPUT_DATA["economics"]["carbon_credit_price_usd_per_t_co2"] = carbon_price_yb
            ENHANCED_INPUT_DATA["economics"]["electricity_sales_price_usd_per_kwh"] = electricity_yb
            ENHANCED_INPUT_DATA["economics"]["tipping_fee_usd_per_ton"] = tipping_yb

            # Compute each year
            results_list = []
            proj_life = ENHANCED_INPUT_DATA["financing"]["project_life"]
            for y in range(1, proj_life+1):
                integrate_lca_or_aspen_sim(y)
                year_data = compute_annual_results_for_year(
                    y,
                    ENHANCED_INPUT_DATA,
                    YEARLY_FEEDSTOCK_DATA[y],
                    ENHANCED_INPUT_DATA["ghg_baseline"]
                )
                results_list.append(year_data)

            # store results
            st.session_state["yearly_results"] = results_list
            st.success("Year-by-year simulation complete. See 'Year-by-Year Results' tab.")

    ############################################################################
    # TAB: YEAR-BY-YEAR RESULTS
    ############################################################################
    with tabs[5]:
        st.subheader("Year-by-Year Results & Visualization")
        if "yearly_results" not in st.session_state:
            st.warning("No results. Run the year-by-year simulation first.")
        else:
            df_yb = pd.DataFrame(st.session_state["yearly_results"])
            st.write("### Year-by-Year Table")
            st.dataframe(df_yb.style.format("{:,.2f}"), use_container_width=True)

            # Quick charts
            st.write("#### Net Cash Flow by Year (Line Chart)")
            st.line_chart(df_yb, x="Year", y="NetCashFlow")

            st.write("#### GHG Reduction by Year (Bar Chart)")
            st.bar_chart(df_yb.set_index("Year")["GHG_Reduction_kg"])

            # Pie chart for final-year revenue breakdown
            final_year = df_yb.loc[df_yb["Year"] == df_yb["Year"].max()]
            if not final_year.empty:
                fy = final_year.iloc[0]
                pie_data = pd.DataFrame({
                    "Revenue Source": ["Carbon", "Electricity", "Tipping"],
                    "Amount": [fy["Carbon_Revenue"], fy["Electricity_Revenue"], fy["Tipping_Revenue"]]
                })
                st.write("#### Pie Chart - Final Year Revenue Breakdown")
                import altair as alt
                pie_chart = alt.Chart(pie_data).mark_arc().encode(
                    theta=alt.Theta(field="Amount", type="quantitative"),
                    color=alt.Color(field="Revenue Source", type="nominal"),
                    tooltip=["Revenue Source", "Amount"]
                )
                st.altair_chart(pie_chart, use_container_width=True)

    ############################################################################
    # TAB: COST ANALYSIS (ROI, IRR, NPV) for Single-Year Approach
    ############################################################################
    with tabs[6]:
        st.subheader("Cost Analysis: ROI, IRR, and NPV (Single-Year)")

        st.write("""
        Evaluate ROI, IRR, and NPV for a single-scenario approach, 
        assuming the same net cash flow each year for the project life.
        """)

        colA, colB = st.columns(2)
        with colA:
            daily_capacity_ca = st.number_input("Daily Capacity (t/day, for cost analysis)", 50, 2000, 330, 10, key="ca_daily_capacity")
            carbon_price_ca = st.number_input("Carbon Price (USD/ton CO2, cost analysis)", 0.0, 300.0, 10.0, 5.0, key="ca_carbon_price")
            discount_ca = st.number_input("Discount Rate (%)", 0.0, 50.0, 8.0, 1.0, key="ca_discount")
        with colB:
            facility_ca = st.number_input("Facility Emissions (kgCO2/t)", 0.0, 1000.0, 200.0, 50.0, key="ca_facility")
            landfill_ca = st.number_input("Landfill Emissions (kgCO2/t)", 0.0, 2000.0, 500.0, 50.0, key="ca_landfill")
            methane_ca = st.number_input("Methane Factor (kgCO2eq/t)", 0.0, 1000.0, 100.0, 10.0, key="ca_methane")

        electricity_ca = st.number_input("Electricity Price (USD/kWh)", 0.0, 1.0, 0.11, 0.01, key="ca_elec")
        tipping_ca = st.number_input("Tipping Fee (USD/ton)", 0.0, 500.0, 90.0, 5.0, key="ca_tipping")

        operational_days = ENHANCED_INPUT_DATA["facility"]["operational_days"]
        annual_waste_ca = daily_capacity_ca * operational_days

        # Single-year net
        user_params_ca = {
            "landfill_emissions": landfill_ca,
            "facility_emissions": facility_ca,
            "carbon_price": carbon_price_ca,
            "electricity_price": electricity_ca,
            "tipping_fee": tipping_ca
        }

        result_ca = annual_cashflow(annual_waste_ca, daily_capacity_ca, ENHANCED_INPUT_DATA, user_params_ca)
        annual_net_ca = result_ca["Annual_Net"]

        project_life_ca = ENHANCED_INPUT_DATA["financing"]["project_duration_years"]
        flows_ca = discounted_cash_flow_series(annual_net_ca, project_life_ca, discount_ca)
        capex = ENHANCED_INPUT_DATA["economics"]["capex_usd"]
        npv_project = sum(flows_ca) - capex

        total_net_no_discount = annual_net_ca * project_life_ca
        roi = (total_net_no_discount / capex) * 100.0

        # IRR: single-year repeated flow approach
        cash_flows_ca = [-capex] + [annual_net_ca]*project_life_ca
        if IRR_AVAILABLE:
            irr_value = nf.irr(cash_flows_ca)*100.0
        else:
            irr_value = None

        if isinstance(irr_value, float):
            irr_for_df = irr_value
        else:
            irr_for_df = np.nan

        cost_dict = {
            "Annual Waste (t)": annual_waste_ca,
            "GHG Reduction (kg CO2/yr)": result_ca["GHG_Reduction_kg"],
            "Annual Net (USD/yr)": annual_net_ca,
            "NPV (USD)": npv_project,
            "ROI (%)": roi,
            "IRR (%)": irr_for_df
        }
        df_cost = pd.DataFrame([cost_dict])
        st.table(df_cost.style.format("{:,.2f}"))

        if IRR_AVAILABLE:
            st.info(f"IRR = {irr_for_df:.2f}%")
        else:
            st.info("Install `numpy-financial` to enable IRR calculation.")

        if npv_project > 0:
            st.success(f"Positive NPV => project is financially attractive (IRR={irr_for_df if not np.isnan(irr_for_df) else 'N/A'}).")
        else:
            st.warning(f"Negative NPV => project may not be viable (IRR={irr_for_df if not np.isnan(irr_for_df) else 'N/A'}).")

    ############################################################################
    # TAB: CONCLUSIONS
    ############################################################################
    with tabs[7]:
        st.subheader("Conclusions & Next Steps")
        st.markdown("""
        ### Summary
        This **unified** script shows how to:
        - Perform **single-year** advanced OPEX & GHG calculations,
        - Run a **multi-scenario** approach for daily capacity & carbon price,
        - Model a **year-by-year** feedstock ramp-up, OPEX inflation, and maintenance,
        - Calculate **NPV**, **ROI**, and **IRR** for cost analysis,
        - Present results in **tables** and **visual** charts (line, bar, pie).

        ### Future Enhancements
        1. Integrate real **Aspen/LCA** data or CSV inputs for feedstock & emission factors.
        2. Model complex multi-feedstock blends or partial load inefficiencies.
        3. Add year-by-year changes to carbon price, electricity tariff, or policy incentives.
        4. Expand cost analysis with real financing structures (debt, equity, interest schedules).

        **Thank you for using the Unified Gasification Feasibility Tool!**
        ---
        """)
        

if __name__ == "__main__":
    main()
