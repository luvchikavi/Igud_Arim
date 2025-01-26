import streamlit as st
import pandas as pd
import numpy as np
import math

# For IRR/NPV calculations (optional):
try:
    import numpy_financial as nf
    IRR_AVAILABLE = True
except ImportError:
    IRR_AVAILABLE = False

# For PDF generation:
# pip install fpdf
from fpdf import FPDF

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
        "max_feedstock_tons_day": 500,       # single-scenario limit
        "nominal_daily_capacity_tons": 300   # year-by-year design
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
        # For the year-by-year approach:
        "base_opex_usd_year1": 6_000_000,
        "inflation_rate_pct": 2.0,
        "maintenance_schedule": {5: 1_000_000, 10: 2_000_000},
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

# Example year-by-year feedstock ramp
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
# 2. HELPER FUNCTIONS
###############################################################################

def integrate_lca_or_aspen_sim(year):
    """Placeholder if you want LCA/Aspen data integration for year."""
    pass

def compute_yearly_opex(year, base_opex_year1, inflation_rate_pct, maintenance_schedule):
    # OPEX inflates each year + lumps from maintenance_schedule
    inflated_opex = base_opex_year1 * ((1 + inflation_rate_pct/100)**(year - 1))
    additional_maintenance = maintenance_schedule.get(year, 0.0)
    return inflated_opex + additional_maintenance

def compute_annual_results_for_year(year, base_input, feedstock_info, ghg_baseline):
    daily_tons = feedstock_info["daily_feedstock_tons"]
    capacity_factor = feedstock_info["capacity_factor"]
    design_days = 330
    actual_days = design_days * capacity_factor
    annual_waste = daily_tons * actual_days

    baseline_co2 = ghg_baseline["landfill_emissions_kg_per_ton"] * annual_waste
    methane_co2  = ghg_baseline["methane_factor_kg_co2eq_per_ton"] * annual_waste
    facility_co2 = ghg_baseline["facility_emissions_kg_per_ton"] * annual_waste
    ghg_reduction_kg = (baseline_co2 + methane_co2) - facility_co2

    econ = base_input["economics"]
    carbon_revenue = (ghg_reduction_kg / 1000.0) * econ["carbon_credit_price_usd_per_t_co2"]
    electricity_revenue = (annual_waste *
                           econ["electricity_sales_price_usd_per_kwh"] *
                           base_input["facility"]["electricity_generation_kwh_per_ton"])
    tipping_revenue = annual_waste * econ["tipping_fee_usd_per_ton"]
    total_revenue = carbon_revenue + electricity_revenue + tipping_revenue

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

def calculate_advanced_opex(daily_capacity, input_data):
    base_opex = input_data["economics"]["base_opex_usd_per_year"]
    scaling_factor = input_data["economics"]["opex_scaling_factor"]
    design_capacity = (input_data["facility"]["annual_capacity_tons"] /
                       input_data["facility"]["operational_days"])
    if daily_capacity <= design_capacity:
        return base_opex
    else:
        ratio_over = (daily_capacity - design_capacity) / design_capacity
        return base_opex * (1 + ratio_over * scaling_factor)

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
    ghg_red_kg = calculate_ghg_reduction_advanced(
        user_params["landfill_emissions"],
        user_params["facility_emissions"],
        input_data["ghg_baseline"]["methane_factor_kg_co2eq_per_ton"],
        annual_waste
    )
    carbon_rev = calculate_carbon_revenue(ghg_red_kg, user_params["carbon_price"])
    elec_rev   = calculate_electricity_revenue(
        annual_waste,
        input_data["facility"]["electricity_generation_kwh_per_ton"],
        user_params["electricity_price"]
    )
    tip_rev    = calculate_tipping_revenue(annual_waste, user_params["tipping_fee"])
    total_rev  = carbon_rev + elec_rev + tip_rev
    adv_opex   = calculate_advanced_opex(daily_capacity, input_data)
    net        = total_rev - adv_opex

    return {
        "GHG_Reduction_kg": ghg_red_kg,
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
# 3. MAIN STREAMLIT APP
###############################################################################
def main():
    st.title("Oporto-Carbon Gasification Feasibility Tool")

    # We'll create 8 tabs
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

    # Helper function to show the same footer on each tab
    def show_footer():
        st.markdown(
            """
            ---
            **Operated by Oporto-Carbon | Designed by Dr. Avi Luvchik**  
            @ all rights reserved 2025
            """,
            unsafe_allow_html=True
        )

    ###########################################################################
    # TAB 0: HOME
    ###########################################################################
    with tabs[0]:
        st.subheader("Welcome to Oporto-Carbon's Gasification Feasibility Tool")
        # Display the Oporto Carbon logo (if in the same folder)
        st.image("oporto_logo.png", width=180)

        st.markdown("""
        **Oporto-Carbon** is a leading provider of waste-to-energy solutions.  
        This feasibility dashboard is created by **Dr. Avi Luvchik** to help 
        evaluate the economic and environmental viability of gasification projects, 
        including GHG reductions and cost metrics (ROI, IRR, NPV).
        
        Use the tabs above to explore various functionalities and final conclusions.
        """)
        show_footer()

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
        feedstock_list = []
        for y in sorted(YEARLY_FEEDSTOCK_DATA.keys()):
            feedstock_list.append({
                "Year": y,
                "Daily Feedstock (tons)": YEARLY_FEEDSTOCK_DATA[y]["daily_feedstock_tons"],
                "Capacity Factor": YEARLY_FEEDSTOCK_DATA[y]["capacity_factor"]
            })
        df_feedstock = pd.DataFrame(feedstock_list)
        st.dataframe(df_feedstock, use_container_width=True)

        show_footer()

    ###########################################################################
    # TAB 2: SINGLE SCENARIO
    ###########################################################################
    with tabs[2]:
        st.subheader("Single-Year Scenario")
        col1, col2 = st.columns(2)
        with col1:
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
                step=50.0, key="single_landfill"
            )
            facility_emissions = st.number_input(
                "Facility Emissions (kg CO2/ton)",
                0.0, 1000.0,
                float(ENHANCED_INPUT_DATA["ghg_baseline"]["facility_emissions_kg_per_ton"]),
                step=50.0, key="single_facility"
            )
            methane_factor = st.number_input(
                "Methane Factor (kg CO2e/ton)",
                0.0, 1000.0,
                float(ENHANCED_INPUT_DATA["ghg_baseline"]["methane_factor_kg_co2eq_per_ton"]),
                step=25.0, key="single_methane"
            )

        with col2:
            carbon_price = st.number_input(
                "Carbon Price (USD/ton CO2)",
                0.0, 300.0,
                float(ENHANCED_INPUT_DATA["economics"]["carbon_credit_price_usd_per_t_co2"]),
                step=5.0, key="single_carbon"
            )
            electricity_price = st.number_input(
                "Electricity Price (USD/kWh)",
                0.0, 1.0,
                float(ENHANCED_INPUT_DATA["economics"]["electricity_sales_price_usd_per_kwh"]),
                step=0.01, key="single_elec"
            )
            tipping_fee = st.number_input(
                "Tipping Fee (USD/ton)",
                0.0, 300.0,
                90.0, step=5.0, key="single_tipping"
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
        df_single = pd.DataFrame([{
            "Annual Waste (tons)": annual_waste,
            "GHG Reduction (kg CO2eq)": single_res["GHG_Reduction_kg"],
            "Carbon Revenue (USD/yr)": single_res["Carbon_Revenue"],
            "Electricity Revenue (USD/yr)": single_res["Electricity_Revenue"],
            "Tipping Revenue (USD/yr)": single_res["Tipping_Revenue"],
            "Total Revenue (USD/yr)": single_res["Total_Revenue"],
            "OPEX (USD/yr)": single_res["OPEX"],
            "Annual Net (USD/yr)": single_res["Annual_Net"]
        }])
        st.table(df_single.style.format("{:,.2f}"))
        st.session_state["single_scenario_df"] = df_single.copy()

        if single_res["Annual_Net"] >= 0:
            st.success("Profitable scenario (annual basis)!")
        else:
            st.warning("Not profitable for these inputs.")

        show_footer()

    ###########################################################################
    # TAB 3: MULTI-SCENARIO
    ###########################################################################
    with tabs[3]:
        st.subheader("Multi-Scenario (One-Year What-If)")

        col1, col2 = st.columns(2)
        with col1:
            min_cap = st.number_input("Min Capacity (t/day)", 50, 2000, 100, 10, key="ms_min_cap")
            max_cap = st.number_input("Max Capacity (t/day)", 50, 2000, 600, 10, key="ms_max_cap")
            step_cap = st.number_input("Capacity Step", 1, 500, 50, 10, key="ms_step_cap")
        with col2:
            min_c = st.number_input("Min Carbon (USD/t CO2)", 0.0, 300.0, 5.0, 5.0, key="ms_min_c")
            max_c = st.number_input("Max Carbon (USD/t CO2)", 0.0, 300.0, 30.0, 5.0, key="ms_max_c")
            step_c = st.number_input("Carbon Step", 1.0, 50.0, 5.0, 1.0, key="ms_step_c")

        st.write("Other fixed parameters:")
        col3, col4 = st.columns(2)
        with col3:
            landfill_ms = st.number_input("Landfill Emissions Ms (kgCO2/ton)", 0.0, 2000.0,
                                          500.0, 50.0, key="ms_landfill")
            facility_ms = st.number_input("Facility Emissions Ms (kgCO2/ton)", 0.0, 1000.0,
                                          200.0, 50.0, key="ms_facility")
        with col4:
            methane_ms = st.number_input("Methane Factor Ms (kgCO2eq/ton)", 0.0, 1000.0,
                                         100.0, 10.0, key="ms_methane")
            electricity_ms = st.number_input("Electricity Price Ms (USD/kWh)", 0.0, 1.0,
                                             0.11, 0.01, key="ms_elec")

        tipping_ms = st.number_input("Tipping Fee Ms (USD/ton)", 0.0, 500.0,
                                     90.0, 5.0, key="ms_tipping")

        run_multi = st.button("Run Multi-Scenario", key="ms_run_button")
        if run_multi:
            capacities = range(min_cap, max_cap+1, step_cap)
            carbon_prices = np.arange(min_c, max_c+0.0001, step_c)
            op_days = ENHANCED_INPUT_DATA["facility"]["operational_days"]
            scenario_list = []

            max_feed = ENHANCED_INPUT_DATA["facility"]["max_feedstock_tons_day"]
            for c in capacities:
                feasible = "Yes" if c <= max_feed else "No"
                for cp in carbon_prices:
                    annual_waste = c * op_days
                    user_params_ms = {
                        "landfill_emissions": landfill_ms,
                        "facility_emissions": facility_ms,
                        "carbon_price": cp,
                        "electricity_price": electricity_ms,
                        "tipping_fee": tipping_ms
                    }
                    ms_res = annual_cashflow(annual_waste, c, ENHANCED_INPUT_DATA, user_params_ms)
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

            df_multi = pd.DataFrame(scenario_list).replace({None: np.nan})
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
            format_dict = {col: "{:,.2f}" for col in numeric_cols if col in df_multi.columns}

            st.write("### Multi-Scenario Results (One Year)")
            st.dataframe(df_multi.style.format(format_dict), use_container_width=True)

            # Save for PDF
            st.session_state["multi_scenario_df"] = df_multi.copy()

            st.success("Multi-scenario run complete!")

        show_footer()

    ###########################################################################
    # TAB 4: YEAR-BY-YEAR MODEL
    ###########################################################################
    with tabs[4]:
        st.subheader("Year-by-Year Model (Feedstock Ramp, OPEX Inflation, Maintenance)")
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

        run_yearly = st.button("Run Year-by-Year Model", key="yb_run")
        if run_yearly:
            ENHANCED_INPUT_DATA["financing"]["discount_rate_pct"] = disc_yb
            ENHANCED_INPUT_DATA["economics"]["carbon_credit_price_usd_per_t_co2"] = cprice_yb
            ENHANCED_INPUT_DATA["economics"]["electricity_sales_price_usd_per_kwh"] = elec_yb
            ENHANCED_INPUT_DATA["economics"]["tipping_fee_usd_per_ton"] = tip_yb

            results_list_yb = []
            yrs = ENHANCED_INPUT_DATA["financing"]["project_life"]
            for y in range(1, yrs+1):
                integrate_lca_or_aspen_sim(y)
                row = compute_annual_results_for_year(
                    y, ENHANCED_INPUT_DATA, YEARLY_FEEDSTOCK_DATA[y],
                    ENHANCED_INPUT_DATA["ghg_baseline"]
                )
                results_list_yb.append(row)

            df_yb = pd.DataFrame(results_list_yb).replace({None: np.nan})
            st.dataframe(df_yb.style.format("{:,.2f}"), use_container_width=True)

            st.session_state["yearly_results_df"] = df_yb.copy()

            st.success("Year-by-year model complete! Check next tab for charts.")

        show_footer()

    ###########################################################################
    # TAB 5: YEAR-BY-YEAR RESULTS
    ###########################################################################
    with tabs[5]:
        st.subheader("Year-by-Year Results & Charts")
        df_yb = st.session_state.get("yearly_results_df")
        if df_yb is None:
            st.warning("No year-by-year results found. Please run the previous tab.")
        else:
            st.write("### Results Table")
            st.dataframe(df_yb.style.format("{:,.2f}"), use_container_width=True)

            st.write("#### Net Cash Flow by Year (Line Chart)")
            st.line_chart(df_yb, x="Year", y="NetCashFlow")

            st.write("#### GHG Reduction by Year (Bar Chart)")
            st.bar_chart(df_yb.set_index("Year")["GHG_Reduction_kg"])

            final_yr = df_yb.loc[df_yb["Year"] == df_yb["Year"].max()]
            if len(final_yr) > 0:
                st.write("#### Final Year Revenue Breakdown (Pie Chart)")
                row = final_yr.iloc[0]
                import altair as alt
                pdat = pd.DataFrame({
                    "Revenue Source": ["Carbon", "Electricity", "Tipping"],
                    "Amount": [row["Carbon_Revenue"], row["Electricity_Revenue"], row["Tipping_Revenue"]]
                })
                chart = alt.Chart(pdat).mark_arc().encode(
                    theta=alt.Theta(field="Amount", type="quantitative"),
                    color=alt.Color(field="Revenue Source", type="nominal"),
                    tooltip=["Revenue Source", "Amount"]
                )
                st.altair_chart(chart, use_container_width=True)

        show_footer()

    ###########################################################################
    # TAB 6: COST ANALYSIS (Single-year repeated flow approach)
    ###########################################################################
    with tabs[6]:
        st.subheader("Cost Analysis: ROI, IRR, NPV (Single-Year)")

        colA, colB = st.columns(2)
        with colA:
            daily_capacity_ca = st.number_input("Daily Capacity (t/day, cost analysis)",
                                                50, 2000, 330, 10, key="ca_dailycap")
            carbon_price_ca = st.number_input("Carbon Price (USD/ton CO2)",
                                              0.0, 300.0, 10.0, 5.0, key="ca_carbonpr")
            discount_ca = st.number_input("Discount Rate (%)",
                                          0.0, 50.0, 8.0, 1.0, key="ca_disc")
        with colB:
            facility_ca = st.number_input("Facility Emissions (kgCO2/t)",
                                          0.0, 1000.0, 200.0, 50.0, key="ca_fac")
            landfill_ca = st.number_input("Landfill Emissions (kgCO2/t)",
                                          0.0, 2000.0, 500.0, 50.0, key="ca_landf")
            methane_ca = st.number_input("Methane Factor (kgCO2eq/t)",
                                         0.0, 1000.0, 100.0, 10.0, key="ca_metha")

        elec_ca = st.number_input("Electricity Price (USD/kWh)",
                                  0.0, 1.0, 0.11, 0.01, key="ca_elecprice")
        tip_ca = st.number_input("Tipping Fee (USD/ton)",
                                 0.0, 500.0, 90.0, 5.0, key="ca_tipp")

        op_days = ENHANCED_INPUT_DATA["facility"]["operational_days"]
        annual_waste_ca = daily_capacity_ca * op_days

        user_params_ca = {
            "landfill_emissions": landfill_ca,
            "facility_emissions": facility_ca,
            "carbon_price": carbon_price_ca,
            "electricity_price": elec_ca,
            "tipping_fee": tip_ca
        }
        result_ca = annual_cashflow(annual_waste_ca, daily_capacity_ca, ENHANCED_INPUT_DATA, user_params_ca)
        annual_net_ca = result_ca["Annual_Net"]

        project_life_ca = ENHANCED_INPUT_DATA["financing"]["project_duration_years"]
        flows_ca = discounted_cash_flow_series(annual_net_ca, project_life_ca, discount_ca)
        capex = ENHANCED_INPUT_DATA["economics"]["capex_usd"]
        npv_project = sum(flows_ca) - capex

        total_net_undiscounted = annual_net_ca * project_life_ca
        roi = (total_net_undiscounted / capex) * 100.0

        if IRR_AVAILABLE:
            irr_value = nf.irr([-capex] + [annual_net_ca]*project_life_ca) * 100.0
        else:
            irr_value = None

        cost_dict = {
            "Annual Waste (t)": annual_waste_ca,
            "GHG Reduction (kg CO2/yr)": result_ca["GHG_Reduction_kg"],
            "Annual Net (USD/yr)": annual_net_ca,
            "NPV (USD)": npv_project,
            "ROI (%)": roi,
            "IRR (%)": irr_value
        }
        df_cost = pd.DataFrame([cost_dict]).replace({None: np.nan})
        fmt_dict = {
            "Annual Waste (t)": "{:,.2f}",
            "GHG Reduction (kg CO2/yr)": "{:,.2f}",
            "Annual Net (USD/yr)": "{:,.2f}",
            "NPV (USD)": "{:,.2f}",
            "ROI (%)": "{:,.2f}",
            "IRR (%)": "{:,.2f}"
        }
        st.table(df_cost.style.format(fmt_dict))

        # Store for PDF
        st.session_state["cost_analysis_df"] = df_cost.copy()

        if irr_value is not None:
            st.info(f"IRR = {irr_value:.2f}%")
        else:
            st.info("Install `numpy-financial` for IRR calculation.")

        if npv_project > 0:
            st.success(f"Positive NPV => financially attractive (IRR={irr_value if irr_value else 'N/A'}).")
        else:
            st.warning(f"Negative NPV => not viable (IRR={irr_value if irr_value else 'N/A'}).")

        show_footer()

    ###########################################################################
    # TAB 7: CONCLUSIONS (Footer in all tabs + PDF Download)
    ###########################################################################
    with tabs[7]:
        st.subheader("Conclusions & Next Steps")
        st.markdown("""
        ### Summary
        - Single-year advanced OPEX & GHG calculations
        - Multi-scenario daily capacity & carbon price
        - Year-by-year feedstock ramp (OPEX inflation & maintenance)
        - Charts & cost analysis (ROI, IRR, NPV)

        **Future Enhancements**:
        1. Real LCA/Aspen data integration  
        2. More feedstock mixes & partial loads  
        3. Year-by-year financing structures  

        **Thank you for using our feasibility tool!**
        ---
        """)

        # Retrieve DataFrames from session_state
        single_df = st.session_state.get("single_scenario_df")
        multi_df  = st.session_state.get("multi_scenario_df")
        yearly_df = st.session_state.get("yearly_results_df")
        cost_df   = st.session_state.get("cost_analysis_df")

        st.write("### Download Comprehensive PDF Report")

        # We'll check each DF for None explicitly, to avoid boolean errors
        has_single = single_df is not None
        has_multi  = multi_df is not None
        has_yearly = yearly_df is not None
        has_cost   = cost_df is not None

        if not (has_single or has_multi or has_yearly or has_cost):
            st.info("No scenario DataFrames found in session_state. Please run other tabs first.")
        else:
            # Build the PDF
            pdf = FPDF()
            pdf.add_page()
            pdf.set_auto_page_break(auto=True, margin=15)

            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, "Gasification Feasibility Summary", ln=True, align="C")
            pdf.ln(5)
            pdf.set_font("Arial", "", 12)

            def add_df_to_pdf(title, df: pd.DataFrame):
                pdf.set_font("Arial", "B", 12)
                pdf.cell(0, 8, title, ln=True)
                pdf.set_font("Arial", "", 11)
                pdf.ln(2)
                lines = df.to_string(index=False).split("\n")
                for line in lines:
                    pdf.cell(0, 5, line, ln=True)
                pdf.ln(5)

            if has_single:
                add_df_to_pdf("=== SINGLE SCENARIO RESULTS ===", single_df)
            if has_multi:
                add_df_to_pdf("=== MULTI-SCENARIO RESULTS ===", multi_df)
            if has_yearly:
                add_df_to_pdf("=== YEAR-BY-YEAR RESULTS ===", yearly_df)
            if has_cost:
                add_df_to_pdf("=== COST ANALYSIS RESULTS ===", cost_df)

            pdf_bytes = pdf.output(dest="S").encode("latin-1")
            st.download_button(
                label="Download PDF Report",
                data=pdf_bytes,
                file_name="gasification_report.pdf",
                mime="application/pdf",
                key="download_pdf"
            )

        show_footer()

if __name__ == "__main__":
    main()
