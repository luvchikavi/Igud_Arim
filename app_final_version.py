import streamlit as st
import pandas as pd
import numpy as np
import math

# IRR / NPV (optional)
try:
    import numpy_financial as nf
    IRR_AVAILABLE = True
except ImportError:
    IRR_AVAILABLE = False

# PDF generation
from fpdf import FPDF

# For Monte Carlo random distributions
from scipy.stats import norm, uniform

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
    },
    "trucking": {
        "truck_capacity_tons": 36.0,
        "distance_to_landfill_km": 200.0,
        "emission_factor_kgco2_per_km": 0.8
    }
}

# YEAR-BY-YEAR feedstock data
YEARLY_FEEDSTOCK_DATA = {
    1: {"daily_feedstock_tons": 200, "capacity_factor": 0.80},
    2: {"daily_feedstock_tons": 250, "capacity_factor": 0.85},
    3: {"daily_feedstock_tons": 300, "capacity_factor": 0.90},
    4: {"daily_feedstock_tons": 320, "capacity_factor": 0.95},
    5: {"daily_feedstock_tons": 350, "capacity_factor": 1.00},
}
for y in range(6, 21):
    YEARLY_FEEDSTOCK_DATA[y] = {"daily_feedstock_tons": 350, "capacity_factor": 1.0}

# MULTI-FEEDSTOCK dictionary
FEEDSTOCKS = {
    "RDF": {
        "fraction": 0.70,
        "electricity_kwh_per_ton": 800,
        "ghg_facility_kg_per_ton": 200
    },
    "GreenWaste": {
        "fraction": 0.30,
        "electricity_kwh_per_ton": 650,
        "ghg_facility_kg_per_ton": 150
    }
}

###############################################################################
# 2. HELPER FUNCTIONS
###############################################################################
def show_footer():
    st.markdown(
        """
        ---
        **Operated by Oporto-Carbon | Designed & Developed by Dr. Avi Luvchik**  
        @ All Rights Reserved 2025
        """,
        unsafe_allow_html=True
    )

def compute_bau_ghg_tons(daily_capacity, input_data):
    days = input_data["facility"]["operational_days"]
    annual_waste = daily_capacity * days
    ghg_landfill_kg = (
        input_data["ghg_baseline"]["landfill_emissions_kg_per_ton"]
        + input_data["ghg_baseline"]["methane_factor_kg_co2eq_per_ton"]
    ) * annual_waste
    truck_capacity = input_data["trucking"]["truck_capacity_tons"]
    distance = input_data["trucking"]["distance_to_landfill_km"]
    ef_truck = input_data["trucking"]["emission_factor_kgco2_per_km"]
    trucks_per_year = annual_waste / truck_capacity
    ghg_trucking_kg = trucks_per_year * distance * ef_truck
    ghg_bau_kg = ghg_landfill_kg + ghg_trucking_kg
    ghg_bau_tons = ghg_bau_kg / 1000.0
    return ghg_bau_tons

def compute_project_ghg_tons(daily_capacity, feedstocks, input_data):
    days = input_data["facility"]["operational_days"]
    annual_waste = daily_capacity * days
    total_fac_ghg_kg = 0.0
    for _, fstock_data in feedstocks.items():
        frac = fstock_data["fraction"]
        portion = annual_waste * frac
        ghg_fac = fstock_data["ghg_facility_kg_per_ton"]
        total_fac_ghg_kg += ghg_fac * portion

    local_distance_km = 20.0
    ef_truck = input_data["trucking"]["emission_factor_kgco2_per_km"]
    truck_capacity = input_data["trucking"]["truck_capacity_tons"]
    trucks_per_year = annual_waste / truck_capacity
    ghg_local_kg = trucks_per_year * local_distance_km * ef_truck

    total_proj_kg = total_fac_ghg_kg + ghg_local_kg
    return total_proj_kg / 1000.0

def compute_revenue_pie(daily_capacity, feedstocks, input_data):
    days = input_data["facility"]["operational_days"]
    annual_waste = daily_capacity * days
    total_kwh = 0.0
    for _, fstock_data in feedstocks.items():
        frac = fstock_data["fraction"]
        portion = annual_waste * frac
        total_kwh += portion * fstock_data["electricity_kwh_per_ton"]
    ghg_bau = compute_bau_ghg_tons(daily_capacity, input_data)
    ghg_proj = compute_project_ghg_tons(daily_capacity, feedstocks, input_data)
    ghg_reduction_tons = max(0, ghg_bau - ghg_proj)
    carbon_price = input_data["economics"]["carbon_credit_price_usd_per_t_co2"]
    carbon_rev = ghg_reduction_tons * carbon_price
    elec_price = input_data["economics"]["electricity_sales_price_usd_per_kwh"]
    elec_rev = total_kwh * elec_price
    tipping_fee = input_data["economics"]["tipping_fee_usd_per_ton"]
    tipping_rev = annual_waste * tipping_fee
    return {"Carbon": carbon_rev, "Electricity": elec_rev, "Tipping": tipping_rev}

def discounted_cash_flow_series(annual_net, project_life, discount_rate):
    r = discount_rate / 100.0
    flows = []
    for y in range(1, project_life+1):
        flows.append(annual_net / ((1 + r)**y))
    return flows

def is_none_or_empty(df):
    """Helper to check if a DataFrame is None or empty (0 rows)."""
    return (df is None) or (df.empty)

###############################################################################
# 3. MAIN APP
###############################################################################
def main():
    st.title("Gasification Feasibility Tool- Igud Arim Hifa")

    # Define tabs
    tabs = st.tabs([
        "Home",
        "Base Input Data",
        "Single Scenario",
        "Multi-Scenario",
        "Year-by-Year Analysis",
        "Monte Carlo",
        "Cost Analysis",
        "Conclusions"
    ])

    ###########################################################################
    # TAB 0: HOME
    ###########################################################################
    with tabs[0]:
        st.subheader("Welcome to Oporto-Carbon's -- Gasification Feasibility Tool")
        # Show the Oporto Carbon logo if present
        st.image("oporto_logo.png", width=180)

        st.markdown("""
        **Oporto-Carbon**  specializes in sustainability consulting and carbon management solutions. 
        With a proven track record of helping organizations achieve their ESG goals, Oporto Carbon is 
        a trusted partner for businesses navigating the complexities of climate change and regulatory compliance.
        """)
        st.markdown("""
        
        **Dr. Avi Luvchik** is the founder and CEO of Oporto Carbon. With extensive experience in sustainability 
        and emissions reduction strategies, Dr. Luvchik has been instrumental in developing innovative solutions 
        for organizations worldwide. His leadership and expertise drive the success of Oporto Carbon's projects.
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
                new_key = parent_key + sep + str(k) if parent_key else str(k)
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)

        flat_input = flatten_dict(ENHANCED_INPUT_DATA)
        df_input = pd.DataFrame(list(flat_input.items()), columns=["Parameter", "Value"])
        st.write("### Core Input Parameters")
        st.dataframe(df_input, use_container_width=True)

        st.write("### Year-by-Year Feedstock Data")
        feedstock_rows = []
        for yk in sorted(YEARLY_FEEDSTOCK_DATA.keys()):
            feedstock_rows.append({
                "Year": yk,
                "Daily Feedstock (t/day)": YEARLY_FEEDSTOCK_DATA[yk]["daily_feedstock_tons"],
                "Capacity Factor": YEARLY_FEEDSTOCK_DATA[yk]["capacity_factor"]
            })
        st.dataframe(pd.DataFrame(feedstock_rows), use_container_width=True)

        st.write("### Multi-Feedstock Approach")
        multi_list = []
        for k, v in FEEDSTOCKS.items():
            multi_list.append({
                "Feedstock": k,
                "Fraction": v["fraction"],
                "Elec (kWh/ton)": v["electricity_kwh_per_ton"],
                "Facility GHG (kgCO2/ton)": v["ghg_facility_kg_per_ton"]
            })
        st.dataframe(pd.DataFrame(multi_list), use_container_width=True)

        show_footer()

    ###########################################################################
    # TAB 2: SINGLE SCENARIO
    ###########################################################################
    with tabs[2]:
        st.subheader("Single-Year Scenario: Baseline vs. Project")

        colA, colB = st.columns(2)
        with colA:
            daily_capacity = st.number_input(
                "Daily Capacity (t/day)",
                min_value=50,
                max_value=2000,
                value=int(ENHANCED_INPUT_DATA["facility"]["daily_capacity_tons"]),
                step=10,
                format="%d",
                key="single_dailycap"
            )
        with colB:
            new_rdf_frac = st.slider("RDF Fraction", 0.0, 1.0, FEEDSTOCKS["RDF"]["fraction"], 0.05)
            FEEDSTOCKS["RDF"]["fraction"] = new_rdf_frac
            FEEDSTOCKS["GreenWaste"]["fraction"] = 1.0 - new_rdf_frac

        ghg_bau = compute_bau_ghg_tons(daily_capacity, ENHANCED_INPUT_DATA)
        ghg_proj = compute_project_ghg_tons(daily_capacity, FEEDSTOCKS, ENHANCED_INPUT_DATA)
        ghg_reduction = max(0, ghg_bau - ghg_proj)

        st.write("#### Baseline vs. Project (Tons CO2eq)")
        df_compare = pd.DataFrame({
            "Scenario": ["BAU (Baseline)", "Project"],
            "GHG (tons)": [ghg_bau, ghg_proj]
        })
        # Style numeric columns
        numeric_cols = df_compare.select_dtypes(include=[np.number]).columns
        fmt_dict = {col: "{:,.2f}" for col in numeric_cols}
        st.table(df_compare.style.format(fmt_dict))

        st.write(f"**GHG Reduction**: {ghg_reduction:,.2f} tons CO2eq")

        st.bar_chart(df_compare.set_index("Scenario"))

        # Pie chart
        revs = compute_revenue_pie(daily_capacity, FEEDSTOCKS, ENHANCED_INPUT_DATA)
        import altair as alt
        df_pie = pd.DataFrame({
            "Revenue Source": list(revs.keys()),
            "USD Amount": list(revs.values())
        })
        chart_pie = alt.Chart(df_pie).mark_arc().encode(
            theta=alt.Theta(field="USD Amount", type="quantitative"),
            color=alt.Color(field="Revenue Source", type="nominal"),
            tooltip=["Revenue Source", "USD Amount"]
        )
        st.altair_chart(chart_pie, use_container_width=True)

        single_df = pd.DataFrame([{
            "Daily Capacity (t/day)": daily_capacity,
            "GHG BAU (t)": ghg_bau,
            "GHG Project (t)": ghg_proj,
            "GHG Reduction (t)": ghg_reduction,
            "Revenue Carbon (USD)": revs["Carbon"],
            "Revenue Electricity (USD)": revs["Electricity"],
            "Revenue Tipping (USD)": revs["Tipping"]
        }])
        st.session_state["single_scenario_df"] = single_df

        show_footer()

    ###########################################################################
    # TAB 3: MULTI-SCENARIO
    ###########################################################################
    with tabs[3]:
        st.subheader("Multi-Scenario (One-Year)")
        colm1, colm2 = st.columns(2)
        with colm1:
            min_cap = st.number_input("Min Capacity (t/day)", 50, 2000, 100, 10)
            max_cap = st.number_input("Max Capacity (t/day)", 50, 2000, 600, 10)
            step_cap = st.number_input("Step for Capacity", 1, 500, 50, 10)
        with colm2:
            min_c = st.number_input("Min Carbon (USD/t)", 0.0, 300.0, 5.0, 5.0)
            max_c = st.number_input("Max Carbon (USD/t)", 0.0, 300.0, 30.0, 5.0)
            step_c = st.number_input("Carbon Step", 1.0, 50.0, 5.0, 1.0)

        if st.button("Run Multi-Scenario"):
            scenario_list = []
            for ccap in range(min_cap, max_cap+1, step_cap):
                for cp_ in np.arange(min_c, max_c+0.001, step_c):
                    ENHANCED_INPUT_DATA["economics"]["carbon_credit_price_usd_per_t_co2"] = cp_
                    ghg_bau_ = compute_bau_ghg_tons(ccap, ENHANCED_INPUT_DATA)
                    ghg_proj_ = compute_project_ghg_tons(ccap, FEEDSTOCKS, ENHANCED_INPUT_DATA)
                    ghg_red_  = max(0, ghg_bau_ - ghg_proj_)
                    rev_dict_ = compute_revenue_pie(ccap, FEEDSTOCKS, ENHANCED_INPUT_DATA)
                    scenario_list.append({
                        "Capacity (t/day)": ccap,
                        "Carbon Price (USD/t)": cp_,
                        "GHG BAU (t)": ghg_bau_,
                        "GHG Project (t)": ghg_proj_,
                        "GHG Reduction (t)": ghg_red_,
                        "Rev Carbon": rev_dict_["Carbon"],
                        "Rev Elec": rev_dict_["Electricity"],
                        "Rev Tipping": rev_dict_["Tipping"]
                    })
            df_multi = pd.DataFrame(scenario_list)
            numeric_cols = df_multi.select_dtypes(include=[np.number]).columns
            fmt_dict = {col: "{:,.2f}" for col in numeric_cols}
            st.table(df_multi.style.format(fmt_dict))

            st.session_state["multi_scenario_df"] = df_multi
            st.success("Multi-scenario run complete!")

        show_footer()

    ###########################################################################
    # TAB 4: YEAR-BY-YEAR ANALYSIS
    ###########################################################################
    with tabs[4]:
        st.subheader("Year-by-Year Analysis (Merged)")
        disc_rate_ = st.number_input("Discount Rate (%) (YBY)", 0.0, 50.0,
                                     ENHANCED_INPUT_DATA["financing"]["discount_rate_pct"], 0.5)
        cprice_ = st.number_input("Carbon Price (USD/t) (YBY)", 0.0, 300.0,
                                  ENHANCED_INPUT_DATA["economics"]["carbon_credit_price_usd_per_t_co2"], 5.0)

        if st.button("Run Year-by-Year"):
            ENHANCED_INPUT_DATA["financing"]["discount_rate_pct"] = disc_rate_
            ENHANCED_INPUT_DATA["economics"]["carbon_credit_price_usd_per_t_co2"] = cprice_

            results_list = []
            for yk in sorted(YEARLY_FEEDSTOCK_DATA.keys()):
                daily_tons_ = YEARLY_FEEDSTOCK_DATA[yk]["daily_feedstock_tons"]
                ghg_bau_ = compute_bau_ghg_tons(daily_tons_, ENHANCED_INPUT_DATA)
                ghg_proj_ = compute_project_ghg_tons(daily_tons_, FEEDSTOCKS, ENHANCED_INPUT_DATA)
                red_ = max(0, ghg_bau_ - ghg_proj_)

                revs_ = compute_revenue_pie(daily_tons_, FEEDSTOCKS, ENHANCED_INPUT_DATA)
                total_rev_ = sum(revs_.values())
                opex_ = ENHANCED_INPUT_DATA["economics"]["base_opex_usd_per_year"]
                net_cf_ = total_rev_ - opex_
                results_list.append({
                    "Year": yk,
                    "Daily Tons": daily_tons_,
                    "GHG BAU (t)": ghg_bau_,
                    "GHG Project (t)": ghg_proj_,
                    "GHG Reduction (t)": red_,
                    "Revenue Carbon": revs_["Carbon"],
                    "Revenue Elec": revs_["Electricity"],
                    "Revenue Tipping": revs_["Tipping"],
                    "NetCF (USD)": net_cf_
                })

            df_yby = pd.DataFrame(results_list)
            numeric_cols = df_yby.select_dtypes(include=[np.number]).columns
            fmt_dict = {col: "{:,.2f}" for col in numeric_cols}
            st.table(df_yby.style.format(fmt_dict))

            st.line_chart(df_yby.set_index("Year")[["GHG BAU (t)", "GHG Project (t)"]])
            st.session_state["yearbyyear_df"] = df_yby
            st.success("Year-by-year complete!")
        show_footer()

    ###########################################################################
    # TAB 5: MONTE CARLO
    ###########################################################################
    with tabs[5]:
        st.subheader("Monte Carlo Simulation for Uncertain Parameters")

        runs_ = st.number_input("Number of MC runs", 100, 100000, 1000, 100)
        carbon_mean_ = st.number_input("Carbon Price Mean", 0.0, 300.0, 10.0, 1.0)
        carbon_std_  = st.number_input("Carbon Price StdDev", 0.0, 100.0, 2.0, 0.5)

        if st.button("Run Monte Carlo"):
            mc_results = []
            daily_cap_ = ENHANCED_INPUT_DATA["facility"]["daily_capacity_tons"]
            base_opex_ = ENHANCED_INPUT_DATA["economics"]["base_opex_usd_per_year"]
            for i in range(int(runs_)):
                c_ = np.random.normal(carbon_mean_, carbon_std_)
                if c_ < 0: c_ = 0
                ENHANCED_INPUT_DATA["economics"]["carbon_credit_price_usd_per_t_co2"] = c_
                ghg_bau_ = compute_bau_ghg_tons(daily_cap_, ENHANCED_INPUT_DATA)
                ghg_proj_ = compute_project_ghg_tons(daily_cap_, FEEDSTOCKS, ENHANCED_INPUT_DATA)
                red_ = max(0, ghg_bau_ - ghg_proj_)
                revs_ = compute_revenue_pie(daily_cap_, FEEDSTOCKS, ENHANCED_INPUT_DATA)
                net_annual_ = sum(revs_.values()) - base_opex_
                mc_results.append(net_annual_)
            df_mc = pd.DataFrame({"AnnualNet (USD)": mc_results})
            st.write("### Monte Carlo Results Summary")
            st.write(df_mc.describe())
            st.bar_chart(df_mc)
            st.session_state["monte_carlo_df"] = df_mc

        show_footer()

    ###########################################################################
    # TAB 6: COST ANALYSIS
    ###########################################################################
    with tabs[6]:
        st.subheader("Cost Analysis (Single-Year Repeated Flow)")

        colC1, colC2 = st.columns(2)
        with colC1:
            daily_cap_ca = st.number_input("Daily Capacity (t/day)", 50, 2000, 330, 10)
            carbon_ca = st.number_input("Carbon Price (USD/t CO2)", 0.0, 300.0, 10.0, 5.0)
            disc_ca = st.number_input("Discount Rate (%)", 0.0, 50.0, 8.0, 1.0)
        with colC2:
            pass

        ghg_bau_ca = compute_bau_ghg_tons(daily_cap_ca, ENHANCED_INPUT_DATA)
        ghg_proj_ca = compute_project_ghg_tons(daily_cap_ca, FEEDSTOCKS, ENHANCED_INPUT_DATA)
        ghg_red_ca = max(0, ghg_bau_ca - ghg_proj_ca)
        ENHANCED_INPUT_DATA["economics"]["carbon_credit_price_usd_per_t_co2"] = carbon_ca

        revs_ca = compute_revenue_pie(daily_cap_ca, FEEDSTOCKS, ENHANCED_INPUT_DATA)
        total_rev_ca = sum(revs_ca.values())
        base_opex_ = ENHANCED_INPUT_DATA["economics"]["base_opex_usd_per_year"]
        annual_net_ca = total_rev_ca - base_opex_

        flows_ca = discounted_cash_flow_series(annual_net_ca,
                                               ENHANCED_INPUT_DATA["financing"]["project_duration_years"],
                                               disc_ca)
        capex_ = ENHANCED_INPUT_DATA["economics"]["capex_usd"]
        npv_ca = sum(flows_ca) - capex_

        if IRR_AVAILABLE:
            irr_ca = nf.irr([-capex_] + [annual_net_ca]*ENHANCED_INPUT_DATA["financing"]["project_duration_years"]) * 100.0
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
        numeric_cols = df_cost.select_dtypes(include=[np.number]).columns
        fmt_dict = {col: "{:,.2f}" for col in numeric_cols}
        st.table(df_cost.style.format(fmt_dict))

        st.session_state["cost_analysis_df"] = df_cost

        show_footer()

    ###########################################################################
    # TAB 7: CONCLUSIONS (PDF Download)
    ###########################################################################
    with tabs[7]:
        st.subheader("Conclusions & Next Steps")
        st.markdown("""
        **Summary**:
        - Baseline (BAU) vs. Project GHG in **tons**,
        - Multi-feedstock,
        - Single-year + multi-scenario + year-by-year merged,
        - Monte Carlo approach for uncertainties,
        - Cost analysis with NPV, IRR,
        - Download PDF with results.
        ---
        """)

        single_df = st.session_state.get("single_scenario_df")
        multi_df  = st.session_state.get("multi_scenario_df")
        yearby_df = st.session_state.get("yearbyyear_df")
        cost_df   = st.session_state.get("cost_analysis_df")
        mc_df     = st.session_state.get("monte_carlo_df")

        # Check dataframes with is_none_or_empty
        if (is_none_or_empty(single_df)
            and is_none_or_empty(multi_df)
            and is_none_or_empty(yearby_df)
            and is_none_or_empty(cost_df)
            and is_none_or_empty(mc_df)):
            st.info("No scenario DataFrames found. Please run the other tabs first.")
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
                lines = df.to_string(index=False).split("\n")
                for line in lines:
                    pdf.cell(0, 5, line, ln=True)
                pdf.ln(5)

            add_df_to_pdf("=== SINGLE SCENARIO RESULTS ===", single_df)
            add_df_to_pdf("=== MULTI-SCENARIO RESULTS ===", multi_df)
            add_df_to_pdf("=== YEAR-BY-YEAR RESULTS ===", yearby_df)
            add_df_to_pdf("=== COST ANALYSIS RESULTS ===", cost_df)
            add_df_to_pdf("=== MONTE CARLO RESULTS ===", mc_df)

            pdf_bytes = pdf.output(dest="S").encode("latin-1")
            st.download_button(
                label="Download PDF Report",
                data=pdf_bytes,
                file_name="gasification_report_enhanced.pdf",
                mime="application/pdf",
                key="download_pdf_final"
            )

        show_footer()

if __name__ == "__main__":
    main()
