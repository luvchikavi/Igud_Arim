import streamlit as st
import pandas as pd
import numpy as np
import math

# If you want advanced financial calculations (IRR, NPV), install numpy-financial
try:
    import numpy_financial as nf
    IRR_AVAILABLE = True
except ImportError:
    IRR_AVAILABLE = False

###############################################################################
# 1. ADVANCED GLOBAL INPUT DATA
###############################################################################
# We add placeholders for a "yearly" approach:
# e.g., 20 years of data, with variable feedstock, capacity factor, OPEX scaling, etc.

BASE_INPUT_DATA = {
    "facility": {
        "nominal_daily_capacity_tons": 300,     # The "design" capacity
        "gasification_temp_c": 700,
        "base_electricity_generation_kwh_per_ton": 800, 
        "base_energy_consumption_kwh_per_ton": 75,
    },
    "economics": {
        "capex_usd": 100_000_000,
        "base_opex_usd_year1": 6_000_000,  # OPEX in the first year
        "inflation_rate_pct": 2.0,         # OPEX grows by 2% each year
        "maintenance_schedule": {          # Additional lumps of cost in certain years
            5: 1_000_000,                 # e.g., a big maintenance at year 5
            10: 2_000_000
        },
        "carbon_price_usd_per_t_co2": 10.0,
        "electricity_price_usd_per_kwh": 0.11,
        "tipping_fee_usd_per_ton": 90.0,
    },
    "financing": {
        "project_life": 20,               # 20-year project
        "discount_rate_pct": 8.0
    },
    "ghg_baseline": {
        "landfill_emissions_kg_per_ton": 500,
        "facility_emissions_kg_per_ton": 200,
        "methane_factor_kg_co2eq_per_ton": 100  
    }
}

###############################################################################
# 2. EXAMPLE "Year-by-Year" FEEDSTOCK & CAPACITY DATA
###############################################################################
# Suppose we load or define a schedule of how much feedstock is available each year,
# and how well the facility operates (capacity_factor, etc.). 
# In a real scenario, you might load this from CSV or Aspen results.
# We'll define a small dictionary for demonstration:

# year -> dict with { "daily_feedstock_tons", "capacity_factor", "feedstock_type_mix", ... }
# We'll say we start at lower feedstock in year 1 and ramp up over time.

YEARLY_FEEDSTOCK_DATA = {
    1:  {"daily_feedstock_tons": 200, "capacity_factor": 0.80},
    2:  {"daily_feedstock_tons": 250, "capacity_factor": 0.85},
    3:  {"daily_feedstock_tons": 300, "capacity_factor": 0.90},
    4:  {"daily_feedstock_tons": 320, "capacity_factor": 0.95},
    5:  {"daily_feedstock_tons": 350, "capacity_factor": 1.00},
    # for years 6-20, let's assume it stays around 350 or we can generate some pattern
}
# For missing years, we can fill them with a default approach
for y in range(6, 21):
    # let's assume feedstock stabilizes
    YEARLY_FEEDSTOCK_DATA[y] = {"daily_feedstock_tons": 350, "capacity_factor": 1.0}

###############################################################################
# 3. HELPER FUNCTIONS
###############################################################################

def integrate_lca_or_aspen_sim(year):
    """
    Placeholder function: in a real scenario, you'd query your LCA model 
    or Aspen Plus outputs for year-based data (emissions, syngas yields, etc.).
    Here, we just return a 'stub' or pass through. 
    """
    # Example: maybe aspen_data[year]["facility_emissions_kg_per_ton"] ...
    # We'll just keep it constant for demonstration.
    pass

def compute_yearly_opex(year, base_opex_year1, inflation_rate_pct, maintenance_schedule):
    """
    Example function that:
    1. Grows base OPEX by inflation each year
    2. Adds maintenance lumps in certain years
    """
    # OPEX in year n ~ base_opex_year1 * (1 + inflation)^(n-1)
    inflated_opex = base_opex_year1 * ((1 + inflation_rate_pct/100)**(year - 1))

    # If there's a scheduled maintenance cost in that year, add it
    additional_maintenance = maintenance_schedule.get(year, 0.0)

    return inflated_opex + additional_maintenance

def compute_annual_results_for_year(
    year,
    base_input,
    feedstock_info,
    ghg_baseline
):
    """
    This function calculates:
    - total waste processed
    - GHG reduction
    - revenues
    - OPEX
    - net cash flow (no discount)
    for a single year.
    """
    # daily feedstock from the schedule
    daily_tons = feedstock_info["daily_feedstock_tons"]
    capacity_factor = feedstock_info["capacity_factor"]  # how many days per year effectively

    # We might assume the facility can run 330 days, but capacity_factor adjusts it
    # for downtime or partial operation.
    # E.g., if capacity_factor=0.80, then effectively 330*0.80 days of operation
    design_operational_days = 330
    actual_operational_days = design_operational_days * capacity_factor
    annual_waste = daily_tons * actual_operational_days

    # GHG calculations (simple approach):
    #   baseline: (landfill + methane) 
    #   facility: facility_emissions_kg_per_ton * annual_waste
    baseline_co2 = ghg_baseline["landfill_emissions_kg_per_ton"] * annual_waste
    methane_co2 = ghg_baseline["methane_factor_kg_co2eq_per_ton"] * annual_waste
    facility_co2 = ghg_baseline["facility_emissions_kg_per_ton"] * annual_waste
    ghg_reduction_kg = (baseline_co2 + methane_co2) - facility_co2

    # Revenue
    carbon_revenue = (ghg_reduction_kg / 1000.0) * base_input["economics"]["carbon_price_usd_per_t_co2"]
    electricity_per_ton = base_input["facility"]["base_electricity_generation_kwh_per_ton"]
    electricity_revenue = electricity_per_ton * annual_waste * base_input["economics"]["electricity_price_usd_per_kwh"]
    tipping_revenue = annual_waste * base_input["economics"]["tipping_fee_usd_per_ton"]
    total_revenue = carbon_revenue + electricity_revenue + tipping_revenue

    # OPEX 
    opex_this_year = compute_yearly_opex(
        year,
        base_input["economics"]["base_opex_usd_year1"],
        base_input["economics"]["inflation_rate_pct"],
        base_input["economics"]["maintenance_schedule"]
    )
    # net cash flow (no discount)
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

def compute_sophisticated_irr(cash_flows):
    """
    If we have numpy-financial, we can do nf.irr(cash_flows).
    cash_flows: list of yearly flows with year 0 = -capex
    """
    if not IRR_AVAILABLE:
        return np.nan

    return nf.irr(cash_flows)*100.0  # in %

###############################################################################
# 4. STREAMLIT APP
###############################################################################

def main():
    st.title("Advanced Year-by-Year Gasification Feasibility & Visualization")

    # Additional file-based or real data would be integrated here
    # For demo, we rely on the dictionaries above.

    # --- TABS
    tabs = st.tabs(["Home", "Year-by-Year Inputs", "Run Year-by-Year Model", "Results & Visualization"])

    # -------------------------------------------------------------------------
    # TAB 1: HOME
    # -------------------------------------------------------------------------
    with tabs[0]:
        st.subheader("Welcome to the Advanced Model")
        st.markdown("""
        **Features**:
        1. **Year-by-year** OPEX, feedstock, and maintenance schedules.
        2. Hooks for **LCA/Aspen** data integration (simulated).
        3. **More sophisticated IRR** with changing cash flows each year.
        4. **Charts** (line, bar, pie) to visualize results.
        
        ---
        """)

    # -------------------------------------------------------------------------
    # TAB 2: YEAR-BY-YEAR INPUTS
    # -------------------------------------------------------------------------
    with tabs[1]:
        st.subheader("Year-by-Year Inputs")
        st.markdown("""
        Below is an example feedstock schedule (Yearly FEEDSTOCK DATA).
        You can imagine pulling this from a CSV or Aspen model results.
        """)

        feedstock_df = pd.DataFrame([
            {"Year": y,
             "Daily Feedstock (tons)": YEARLY_FEEDSTOCK_DATA[y]["daily_feedstock_tons"],
             "Capacity Factor": YEARLY_FEEDSTOCK_DATA[y]["capacity_factor"]}
            for y in range(1, 1+BASE_INPUT_DATA["financing"]["project_life"])
        ])
        st.dataframe(feedstock_df, use_container_width=True)

        st.write("**Base Economic & Financing Data**:")
        st.json(BASE_INPUT_DATA)

    # -------------------------------------------------------------------------
    # TAB 3: RUN YEAR-BY-YEAR MODEL
    # -------------------------------------------------------------------------
    with tabs[2]:
        st.subheader("Run the Advanced Year-by-Year Model")

        st.write("Adjust key parameters if desired before running:")

        col1, col2 = st.columns(2)
        with col1:
            discount_rate = st.number_input(
                "Discount Rate (%)",
                min_value=0.0,
                max_value=50.0,
                value=BASE_INPUT_DATA["financing"]["discount_rate_pct"],
                step=0.5
            )
            carbon_price = st.number_input(
                "Carbon Price (USD/ton CO2)",
                min_value=0.0,
                max_value=300.0,
                value=BASE_INPUT_DATA["economics"]["carbon_price_usd_per_t_co2"],
                step=5.0
            )
        with col2:
            electricity_price = st.number_input(
                "Electricity Price (USD/kWh)",
                min_value=0.0,
                max_value=1.0,
                value=BASE_INPUT_DATA["economics"]["electricity_price_usd_per_kwh"],
                step=0.01
            )
            tipping_fee = st.number_input(
                "Tipping Fee (USD/ton)",
                min_value=0.0,
                max_value=300.0,
                value=BASE_INPUT_DATA["economics"]["tipping_fee_usd_per_ton"],
                step=5.0
            )

        run_model = st.button("Run Year-by-Year Calculation")
        if run_model:
            # update the base dictionary with new user inputs
            BASE_INPUT_DATA["financing"]["discount_rate_pct"] = discount_rate
            BASE_INPUT_DATA["economics"]["carbon_price_usd_per_t_co2"] = carbon_price
            BASE_INPUT_DATA["economics"]["electricity_price_usd_per_kwh"] = electricity_price
            BASE_INPUT_DATA["economics"]["tipping_fee_usd_per_ton"] = tipping_fee

            # We'll compute each year's flows
            results_list = []
            project_life = BASE_INPUT_DATA["financing"]["project_life"]
            for y in range(1, project_life+1):
                # If we have an external Aspen or LCA, we'd call integrate_lca_or_aspen_sim(y) here
                integrate_lca_or_aspen_sim(y)

                yearly_data = compute_annual_results_for_year(
                    y,
                    BASE_INPUT_DATA,
                    YEARLY_FEEDSTOCK_DATA[y],
                    BASE_INPUT_DATA["ghg_baseline"]
                )
                results_list.append(yearly_data)

            # We'll store in session state for the next tab
            st.session_state["yearly_results"] = results_list
            st.success("Year-by-year simulation complete! Check the 'Results & Visualization' tab.")

    # -------------------------------------------------------------------------
    # TAB 4: RESULTS & VISUALIZATION
    # -------------------------------------------------------------------------
    with tabs[3]:
        st.subheader("Year-by-Year Results & Visualization")

        if "yearly_results" not in st.session_state:
            st.warning("No results found. Please run the model in the previous tab.")
        else:
            results_list = st.session_state["yearly_results"]
            df_res = pd.DataFrame(results_list)
            st.write("### Year-by-Year Table")
            st.dataframe(df_res.style.format("{:,.2f}"), use_container_width=True)

            st.write("#### Example Charts")

            # 1) LINE CHART: Net Cash Flow by Year
            st.line_chart(
                data=df_res,
                x="Year",
                y="NetCashFlow",
                height=300
            )

            # 2) BAR CHART: GHG Reduction by Year
            st.bar_chart(
                data=df_res.set_index("Year")["GHG_Reduction_kg"],
                height=300
            )

            # 3) Pie Chart for final year's revenue breakdown (Carbon vs. Elec vs. Tipping)
            final_year = df_res.loc[df_res["Year"] == df_res["Year"].max()]
            if len(final_year) > 0:
                final_row = final_year.iloc[0]
                pie_data = pd.DataFrame({
                    "Revenue Source": ["Carbon", "Electricity", "Tipping"],
                    "Amount": [
                        final_row["Carbon_Revenue"],
                        final_row["Electricity_Revenue"],
                        final_row["Tipping_Revenue"]
                    ]
                })

                st.write("#### Pie Chart of Revenue Breakdown (Final Year)")
                # Streamlit doesn't have a direct pie chart, but we can do an Altair or Plotly pie:
                import altair as alt
                pie_chart = alt.Chart(pie_data).mark_arc().encode(
                    theta=alt.Theta(field="Amount", type="quantitative"),
                    color=alt.Color(field="Revenue Source", type="nominal"),
                    tooltip=["Revenue Source", "Amount"]
                )
                st.altair_chart(pie_chart, use_container_width=True)

            # 4) Compute NPV, IRR with variable yearly flows
            # We assume: year 0 = -capex, years 1..n = NetCashFlow
            discount_rate = BASE_INPUT_DATA["financing"]["discount_rate_pct"] / 100.0
            capex = BASE_INPUT_DATA["economics"]["capex_usd"]
            cash_flows = [-capex] + df_res["NetCashFlow"].tolist()
            # discount them ourselves to get NPV
            npv_val = 0.0
            for i, flow in enumerate(cash_flows):
                # i=0 => year 0 => flow is negative capex, no discount
                # i>0 => discount by (1+discount_rate)^i
                if i == 0:
                    npv_val += flow
                else:
                    npv_val += flow / ((1+discount_rate)**i)

            # IRR
            if IRR_AVAILABLE:
                irr_val = nf.irr(cash_flows)*100.0
            else:
                irr_val = np.nan

            st.write("### Project Financial Metrics")
            colA, colB = st.columns(2)
            with colA:
                st.metric(
                    label="Net Present Value (NPV)",
                    value=f"${npv_val:,.2f}",
                    delta=None
                )
            with colB:
                if not np.isnan(irr_val):
                    st.metric(
                        label="Internal Rate of Return (IRR)",
                        value=f"{irr_val:.2f}%",
                        delta=None
                    )
                else:
                    st.warning("Install `numpy-financial` to compute IRR.")

            if npv_val > 0:
                st.success("Project has a positive NPV => financially attractive.")
            else:
                st.warning("Project has a negative NPV => might not be viable at these parameters.")

###############################################################################
# 5. RUN THE APP
###############################################################################
if __name__ == "__main__":
    main()
