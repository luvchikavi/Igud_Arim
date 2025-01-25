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
# 1. GLOBAL INPUT DATA
###############################################################################
ENHANCED_INPUT_DATA = {
    "facility": {
        "annual_capacity_tons": 100_000,
        "daily_capacity_tons": 330,
        "operational_days": 300,
        "gasification_temp_c": 700,
        "energy_consumption_kwh_per_ton": 75,
        "electricity_generation_kwh_per_ton": 800,
        "waste_moisture_pct": 25,
        "ash_content_pct": 5,
        "max_feedstock_tons_day": 500,
        "nominal_daily_capacity_tons": 300
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
        "municipal_treatment_cost_usd_per_ton": 114.0,
        # Additional year-by-year parameters:
        "base_opex_usd_year1": 6_000_000,
        "inflation_rate_pct": 2.0,
        "maintenance_schedule": {
            5: 1_000_000,
            10: 2_000_000
        },
        "tipping_fee_usd_per_ton": 90.0
    },
    "financing": {
        "project_duration_years": 20,
        "tax_incentives_pct": 0.30,
        "discount_rate_pct": 8.0,
        "project_life": 20
    },
    "ghg_baseline": {
        "landfill_emissions_kg_per_ton": 500,
        "facility_emissions_kg_per_ton": 200,
        "methane_factor_kg_co2eq_per_ton": 100
    }
}

###############################################################################
# 2. YEAR-BY-YEAR FEEDSTOCK DATA
###############################################################################
YEARLY_FEEDSTOCK_DATA = {
    1: {"daily_feedstock_tons": 200, "capacity_factor": 0.80},
    2: {"daily_feedstock_tons": 250, "capacity_factor": 0.85},
    3: {"daily_feedstock_tons": 300, "capacity_factor": 0.90},
    4: {"daily_feedstock_tons": 320, "capacity_factor": 0.95},
    5: {"daily_feedstock_tons": 350, "capacity_factor": 1.00},
}
for y in range(6, 21):
    YEARLY_FEEDSTOCK_DATA[y] = {"daily_feedstock_tons": 350, "capacity_factor": 1.0}

###############################################################################
# 3. YEAR-BY-YEAR FUNCTIONS
###############################################################################
def integrate_lca_or_aspen_sim(year):
    """Placeholder for LCA/Aspen integration."""
    pass

def compute_yearly_opex(year, base_opex_year1, inflation_rate_pct, maintenance_schedule):
    inflated_opex = base_opex_year1 * ((1 + inflation_rate_pct/100)**(year - 1))
    additional_maintenance = maintenance_schedule.get(year, 0.0)
    return inflated_opex + additional_maintenance

def compute_annual_results_for_year(year, base_input, feedstock_info, ghg_baseline):
    daily_tons = feedstock_info["daily_feedstock_tons"]
    capacity_factor = feedstock_info["capacity_factor"]
    design_operational_days = 330
    actual_operational_days = design_operational_days * capacity_factor
    annual_waste = daily_tons * actual_operational_days

    baseline_co2 = ghg_baseline["landfill_emissions_kg_per_ton"] * annual_waste
    methane_co2 = ghg_baseline["methane_factor_kg_co2eq_per_ton"] * annual_waste
    facility_co2 = ghg_baseline["facility_emissions_kg_per_ton"] * annual_waste
    ghg_reduction_kg = (baseline_co2 + methane_co2) - facility_co2

    econ = base_input["economics"]
    carbon_revenue = (ghg_reduction_kg/1000.0) * econ["carbon_credit_price_usd_per_t_co2"]
    electricity_revenue = annual_waste * econ["electricity_sales_price_usd_per_kwh"] \
        * base_input["facility"]["electricity_generation_kwh_per_ton"]
    tipping_revenue = annual_waste * econ["tipping_fee_usd_per_ton"]
    total_revenue = carbon_revenue + electricity_revenue + tipping_revenue

    opex_this_year = compute_yearly_opex(
        year, econ["base_opex_usd_year1"], econ["inflation_rate_pct"], econ["maintenance_schedule"]
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
# 4. SINGLE-YEAR FUNCTIONS
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

def calculate_ghg_reduction_advanced(landfill_emissions, facility_emissions, methane_factor, annual_waste):
    baseline = landfill_emissions * annual_waste
    meth_avoided = methane_factor * annual_waste
    facility = facility_emissions * annual_waste
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
        user_params["landfill_emissions"],
        user_params["facility_emissions"],
        input_data["ghg_baseline"]["methane_factor_kg_co2eq_per_ton"],
        annual_waste
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
        flows.append(annual_net / ((1 + r)**y))
    return flows

###############################################################################
# 5. STREAMLIT APP
###############################################################################
def main():
    st.title("Unified Gasification Feasibility: Single/Multi-Year + Visuals")

    # Create the tabs ONCE with unique references
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

    ###########################################################################
    # TAB 0: HOME
    ###########################################################################
    with tabs[0]:
        st.subheader("Welcome")
        st.markdown("""
        This app merges:
        - Single-year advanced OPEX/ghg & cost analysis
        - Multi-scenario "what-if" for daily capacity & carbon price
        - Year-by-year ramp (OPEX inflation, maintenance)
        - Visual charts & cost metrics (ROI, IRR, NPV)
        """)

    ###########################################################################
    # TAB 1: BASE INPUT DATA
    ###########################################################################
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

        st.write("### Year-by-Year Feedstock Data")
        feedstock_rows = []
        for y in sorted(YEARLY_FEEDSTOCK_DATA.keys()):
            feedstock_rows.append({
                "Year": y,
                "Daily Feedstock (tons)": YEARLY_FEEDSTOCK_DATA[y]["daily_feedstock_tons"],
                "Capacity Factor": YEARLY_FEEDSTOCK_DATA[y]["capacity_factor"]
            })
        df_feedstock = pd.DataFrame(feedstock_rows)
        st.dataframe(df_feedstock, use_container_width=True)

    ###########################################################################
    # TAB 2: SINGLE SCENARIO
    ###########################################################################
    with tabs[2]:
        st.subheader("Single-Year Scenario")
        col1, col2 = st.columns(2)

        with col1:
            # We add unique keys here
            daily_capacity = st.number_input(
                "Daily Capacity (t/day)",
                min_value=50, max_value=2000,
                value=int(ENHANCED_INPUT_DATA["facility"]["daily_capacity_tons"]),
                step=10, format="%d",
                key="single_daily_capacity"
            )
            max_feed = ENHANCED_INPUT_DATA["facility"]["max_feedstock_tons_day"]
            if daily_capacity > max_feed:
                st.warning(f"Exceeding max feedstock of {max_feed} t/day...")

            landfill_emissions = st.number_input(
                "Landfill Emissions (kg CO2/ton)",
                0.0, 2000.0,
                float(ENHANCED_INPUT_DATA["ghg_baseline"]["landfill_emissions_kg_per_ton"]),
                50.0,
                key="single_landfill"
            )
            facility_emissions = st.number_input(
                "Facility Emissions (kg CO2/ton)",
                0.0, 1000.0,
                float(ENHANCED_INPUT_DATA["ghg_baseline"]["facility_emissions_kg_per_ton"]),
                50.0,
                key="single_facility"
            )
            methane_factor = st.number_input(
                "Methane Factor (kg CO2e/ton)",
                0.0, 1000.0,
                float(ENHANCED_INPUT_DATA["ghg_baseline"]["methane_factor_kg_co2eq_per_ton"]),
                25.0,
                key="single_methane"
            )

        with col2:
            carbon_price = st.number_input(
                "Carbon Price (USD/ton CO2)",
                0.0, 300.0,
                float(ENHANCED_INPUT_DATA["economics"]["carbon_credit_price_usd_per_t_co2"]),
                5.0,
                key="single_carbon"
            )
            electricity_price = st.number_input(
                "Electricity Price (USD/kWh)",
                0.0, 1.0,
                float(ENHANCED_INPUT_DATA["economics"]["electricity_sales_price_usd_per_kwh"]),
                0.01,
                key="single_elec"
            )
            tipping_fee = st.number_input(
                "Tipping Fee (USD/ton)",
                0.0, 300.0,
                90.0, 5.0,
                key="single_tipping"
            )

        op_days = ENHANCED_INPUT_DATA["facility"]["operational_days"]
        annual_waste = daily_capacity * op_days

        user_params = {
            "landfill_emissions": landfill_emissions,
            "facility_emissions": facility_emissions,
            "carbon_price": carbon_price,
            "electricity_price": electricity_price,
            "tipping_fee": tipping_fee
        }

        single_res = annual_cashflow(annual_waste, daily_capacity, ENHANCED_INPUT_DATA, user_params)
        single_dict = {
            "Annual Waste (tons)": [annual_waste],
            "GHG Reduction (kg CO2eq/year)": [single_res["GHG_Reduction_kg"]],
            "Carbon Revenue (USD/yr)": [single_res["Carbon_Revenue"]],
            "Electricity Revenue (USD/yr)": [single_res["Electricity_Revenue"]],
            "Tipping Revenue (USD/yr)": [single_res["Tipping_Revenue"]],
            "Total Revenue (USD/yr)": [single_res["Total_Revenue"]],
            "OPEX (USD/yr)": [single_res["OPEX"]],
            "Annual Net (USD/yr)": [single_res["Annual_Net"]]
        }
        st.table(pd.DataFrame(single_dict).style.format("{:,.2f}"))

        if single_res["Annual_Net"] >= 0:
            st.success("Project is profitable on an annual basis!")
        else:
            st.warning("Project is not profitable on an annual basis with these inputs.")

    ###########################################################################
    # TAB 3: MULTI-SCENARIO (One-year approach)
    ###########################################################################
    with tabs[3]:
        st.subheader("Multi-Scenario Analysis (One-Year)")

        col1, col2 = st.columns(2)
        with col1:
            min_cap = st.number_input("Min Capacity (t/day)", 50, 2000, 100, 10, key="ms_min_cap")
            max_cap = st.number_input("Max Capacity (t/day)", 50, 2000, 600, 10, key="ms_max_cap")
            step_cap = st.number_input("Step Capacity", 1, 500, 50, 10, key="ms_step_cap")
        with col2:
            min_c = st.number_input("Min Carbon (USD/t CO2)", 0.0, 300.0, 5.0, 5.0, key="ms_min_c")
            max_c = st.number_input("Max Carbon (USD/t CO2)", 0.0, 300.0, 30.0, 5.0, key="ms_max_c")
            step_c = st.number_input("Step Carbon", 1.0, 50.0, 5.0, 1.0, key="ms_step_c")

        st.write("Other fixed parameters:")
        col3, col4 = st.columns(2)
        with col3:
            landfill_ms = st.number_input("Landfill Emissions Ms (kgCO2/ton)", 0.0, 2000.0, 500.0, 50.0, key="ms_landfill")
            facility_ms = st.number_input("Facility Emissions Ms (kgCO2/ton)", 0.0, 1000.0, 200.0, 50.0, key="ms_facility")
        with col4:
            methane_ms = st.number_input("Methane Factor Ms (kgCO2eq/ton)", 0.0, 1000.0, 100.0, 10.0, key="ms_methane")
            electricity_ms = st.number_input("Electricity Price Ms (USD/kWh)", 0.0, 1.0, 0.11, 0.01, key="ms_elec")

        tipping_ms = st.number_input("Tipping Fee Ms (USD/ton)", 0.0, 500.0, 90.0, 5.0, key="ms_tipping")

        run_multi = st.button("Run Multi-Scenario (One-Year)", key="ms_run")
        if run_multi:
            capacities = range(min_cap, max_cap + 1, step_cap)
            carbon_prices = np.arange(min_c, max_c + 0.0001, step_c)
            op_days = ENHANCED_INPUT_DATA["facility"]["operational_days"]
            scenario_list = []

            max_feed = ENHANCED_INPUT_DATA["facility"]["max_feedstock_tons_day"]
            for c in capacities:
                feasible = "Yes" if c <= max_feed else "No"
                for cp in carbon_prices:
                    annual_waste = c * op_days
                    user_ms = {
                        "landfill_emissions": landfill_ms,
                        "facility_emissions": facility_ms,
                        "carbon_price": cp,
                        "electricity_price": electricity_ms,
                        "tipping_fee": tipping_ms
                    }
                    ms_res = annual_cashflow(annual_waste, c, ENHANCED_INPUT_DATA, user_ms)
                    scenario_list.append({
                        "Daily Capacity (t/day)": c,
                        "Carbon Price (USD/t)": cp,
                        "Feasible?": feasible,
                        "Annual Waste (t)": annual_waste,
                        "GHG Red. (kg/yr)": ms_res["GHG_Reduction_kg"],
                        "Carbon Rev. (USD/yr)": ms_res["Carbon_Revenue"],
                        "Elec Rev. (USD/yr)": ms_res["Electricity_Revenue"],
                        "Tipping Rev. (USD/yr)": ms_res["Tipping_Revenue"],
                        "Total Rev. (USD/yr)": ms_res["Total_Revenue"],
                        "OPEX (USD/yr)": ms_res["OPEX"],
                        "Annual Net (USD/yr)": ms_res["Annual_Net"]
                    })

            df_scen = pd.DataFrame(scenario_list).replace({None: np.nan})
            numeric_cols = [
                "Daily Capacity (t/day)",
                "Carbon Price (USD/t)",
                "Annual Waste (t)",
                "GHG Red. (kg/yr)",
                "Carbon Rev. (USD/yr)",
                "Elec Rev. (USD/yr)",
                "Tipping Rev. (USD/yr)",
                "Total Rev. (USD/yr)",
                "OPEX (USD/yr)",
                "Annual Net (USD/yr)"
            ]
            format_dict = {col: "{:,.2f}" for col in numeric_cols if col in df_scen.columns}

            st.write("### Multi-Scenario Results (One Year)")
            st.dataframe(df_scen.style.format(format_dict), use_container_width=True)
            st.success("Multi-scenario run complete!")

    ###########################################################################
    # TAB 4: YEAR-BY-YEAR MODEL
    ###########################################################################
    with tabs[4]:
        st.subheader("Year-by-Year Model (Ramp & Maintenance)")
        disc_yb = st.number_input("Discount Rate (%)", 0.0, 50.0,
                                  ENHANCED_INPUT_DATA["financing"]["discount_rate_pct"], 0.5,
                                  key="yb_disc")
        cprice_yb = st.number_input("Carbon Price (USD/ton CO2)", 0.0, 300.0,
                                    ENHANCED_INPUT_DATA["economics"]["carbon_credit_price_usd_per_t_co2"], 5.0,
                                    key="yb_carbon")
        elec_yb = st.number_input("Electricity Price (USD/kWh)", 0.0, 1.0,
                                  ENHANCED_INPUT_DATA["economics"]["electricity_sales_price_usd_per_kwh"], 0.01,
                                  key="yb_elec")
        tip_yb = st.number_input("Tipping Fee (USD/ton)", 0.0, 300.0,
                                 ENHANCED_INPUT_DATA["economics"]["tipping_fee_usd_per_ton"], 5.0,
                                 key="yb_tip")

        run_yearly = st.button("Run Year-by-Year Simulation", key="yb_run")
        if run_yearly:
            ENHANCED_INPUT_DATA["financing"]["discount_rate_pct"] = disc_yb
            ENHANCED_INPUT_DATA["economics"]["carbon_credit_price_usd_per_t_co2"] = cprice_yb
            ENHANCED_INPUT_DATA["economics"]["electricity_sales_price_usd_per_kwh"] = elec_yb
            ENHANCED_INPUT_DATA["economics"]["tipping_fee_usd_per_ton"] = tip_yb

            results_list_yb = []
            yrs = ENHANCED_INPUT_DATA["financing"]["project_life"]
            for y in range(1, yrs+1):
                integrate_lca_or_aspen_sim(y)
                y_data = compute_annual_results_for_year(
                    y, ENHANCED_INPUT_DATA, YEARLY_FEEDSTOCK_DATA[y],
                    ENHANCED_INPUT_DATA["ghg_baseline"]
                )
                results_list_yb.append(y_data)

            st.session_state["yearly_results"] = results_list_yb
            st.success("Year-by-year simulation complete! See next tab for results.")

    ###########################################################################
    # TAB 5: YEAR-BY-YEAR RESULTS
    ###########################################################################
    with tabs[5]:
        st.subheader("Year-by-Year Results & Charts")
        if "yearly_results" not in st.session_state:
            st.warning("No results. Run year-by-year first.")
        else:
            df_yb = pd.DataFrame(st.session_state["yearly_results"]).replace({None: np.nan})
            st.dataframe(df_yb.style.format("{:,.2f}"), use_container_width=True)

            st.write("#### Net Cash Flow by Year (Line Chart)")
            st.line_chart(df_yb, x="Year", y="NetCashFlow")

            st.write("#### GHG Reduction by Year (Bar Chart)")
            st.bar_chart(df_yb.set_index("Year")["GHG_Reduction_kg"])

            final_yr = df_yb[df_yb["Year"] == df_yb["Year"].max()]
            if len(final_yr) > 0:
                st.write("#### Final Year Revenue Breakdown (Pie Chart)")
                row = final_yr.iloc[0]
                pdat = pd.DataFrame({
                    "Revenue Source": ["Carbon", "Electricity", "Tipping"],
                    "Amount": [row["Carbon_Revenue"], row["Electricity_Revenue"], row["Tipping_Revenue"]]
                })
                import altair as alt
                pchart = alt.Chart(pdat).mark_arc().encode(
                    theta=alt.Theta(field="Amount", type="quantitative"),
                    color=alt.Color(field="Revenue Source", type="nominal"),
                    tooltip=["Revenue Source", "Amount"]
                )
                st.altair_chart(pchart, use_container_width=True)

    ###########################################################################
    # TAB 6: COST ANALYSIS (Single-year repeated flow)
    ###########################################################################
    with tabs[6]:
        st.subheader("Cost Analysis: ROI, IRR, NPV (Single-Year)")

        colA, colB = st.columns(2)
        with colA:
            daily_capacity_ca = st.number_input("Daily Capacity (t/day, cost analysis)",
                                                50, 2000, 330, 10, key="ca_dailycapacity")
            carbon_price_ca = st.number_input("Carbon Price (USD/ton CO2, cost analysis)",
                                              0.0, 300.0, 10.0, 5.0, key="ca_carbonprice")
            discount_ca = st.number_input("Discount Rate (%)",
                                          0.0, 50.0, 8.0, 1.0, key="ca_discount")
        with colB:
            facility_ca = st.number_input("Facility Emissions (kgCO2/t)",
                                          0.0, 1000.0, 200.0, 50.0, key="ca_facility")
            landfill_ca = st.number_input("Landfill Emissions (kgCO2/t)",
                                          0.0, 2000.0, 500.0, 50.0, key="ca_landfill")
            methane_ca = st.number_input("Methane Factor (kgCO2eq/t)",
                                         0.0, 1000.0, 100.0, 10.0, key="ca_methane")

        electricity_ca = st.number_input("Electricity Price (USD/kWh)",
                                         0.0, 1.0, 0.11, 0.01, key="ca_elecprice")
        tipping_ca = st.number_input("Tipping Fee (USD/ton)",
                                     0.0, 500.0, 90.0, 5.0, key="ca_tippingprice")

        op_days = ENHANCED_INPUT_DATA["facility"]["operational_days"]
        annual_waste_ca = daily_capacity_ca * op_days

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

        # IRR
        if IRR_AVAILABLE:
            irr_value = nf.irr([-capex] + [annual_net_ca]*project_life_ca) * 100.0
        else:
            irr_value = None

        df_cost = pd.DataFrame([{
            "Annual Waste (t)": annual_waste_ca,
            "GHG Reduction (kg CO2/yr)": result_ca["GHG_Reduction_kg"],
            "Annual Net (USD/yr)": annual_net_ca,
            "NPV (USD)": npv_project,
            "ROI (%)": roi,
            "IRR (%)": irr_value
        }]).replace({None: np.nan})

        # Numeric formatting
        format_cols = {
            "Annual Waste (t)": "{:,.2f}",
            "GHG Reduction (kg CO2/yr)": "{:,.2f}",
            "Annual Net (USD/yr)": "{:,.2f}",
            "NPV (USD)": "{:,.2f}",
            "ROI (%)": "{:,.2f}",
            "IRR (%)": "{:,.2f}"
        }

        st.table(df_cost.style.format(format_cols))

        current_irr = df_cost["IRR (%)"].iloc[0]
        if not pd.isna(current_irr):
            st.info(f"IRR = {current_irr:.2f}%")
        else:
            st.info("Install `numpy-financial` to enable IRR calculation.")

        if npv_project > 0:
            st.success(f"Positive NPV => project is financially attractive (IRR={current_irr if not pd.isna(current_irr) else 'N/A'}).")
        else:
            st.warning(f"Negative NPV => project might not be viable (IRR={current_irr if not pd.isna(current_irr) else 'N/A'}).")

    ###########################################################################
    # TAB 7: CONCLUSIONS
    ###########################################################################
    with tabs[7]:
        st.subheader("Conclusions & Next Steps")
        st.markdown("""
        ### Summary
        - Single-year advanced OPEX & GHG calculations
        - Multi-scenario daily capacity & carbon price
        - Year-by-year ramp with OPEX inflation & maintenance
        - Charts & cost analysis (ROI, IRR, NPV)

        ### Next Steps
        1. Real LCA/Aspen integration for emissions & yields
        2. More feedstock mixes & partial loads
        3. Year-by-year financing structures

        **Thank you for using this unified feasibility tool!**
        """)

if __name__ == "__main__":
    main()
