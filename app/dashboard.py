"""Green Home Engine dashboard (spec 3.4).

    streamlit run app/dashboard.py

Screens: home overview, retrofit planner, plan results, what-if (sliders + chat) and the
assumptions panel. The engine runs in-process; every number shown comes from the results JSON.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from engine import currency as cur  # noqa: E402
from engine import memory  # noqa: E402
from engine.inputs import AGE_BAND_LABELS, CustomerInput, CustomiseOptions, Usage  # noqa: E402
from engine.pipeline import analyse  # noqa: E402
from engine.whatif import run_whatif  # noqa: E402
from llm.explainer import explain  # noqa: E402
from llm.parser import parse  # noqa: E402
from llm.providers import status as llm_status  # noqa: E402
from sim.parameters import catalogue  # noqa: E402

st.set_page_config(page_title="Green Home Engine", page_icon="🏠", layout="wide")


def _is_dark() -> bool:
    try:
        return st.context.theme.type == "dark"
    except Exception:  # older Streamlit or bare mode
        return False


DARK = _is_dark()
# Reference palette (dataviz skill): slot 1 blue, slot 2 orange; separate steps for each theme.
if DARK:
    ELEC, GAS, OTHER = "#3987e5", "#d95926", "#199e70"
    TODAY = "#6f6e68"                  # neutral for "today" bars
    TRACK = "rgba(255,255,255,0.10)"   # unfilled part of a meter
    GRID = "rgba(255,255,255,0.10)"
    SOFT_BLUE = "#2b5a92"              # recessive points
    RING = "#111614"                   # marker ring = page background
else:
    ELEC, GAS, OTHER = "#2a78d6", "#eb6834", "#1baf7a"
    TODAY = "#b7b6b0"
    TRACK = "rgba(128,128,128,0.16)"
    GRID = "rgba(128,128,128,0.18)"
    SOFT_BLUE = "#9ec5f4"
    RING = "#ffffff"
FUEL_COLOR = {"electricity": ELEC, "gas": GAS}
SYM = cur.symbol()


def money(gbp) -> str:
    """Engine amounts are GBP; show them in rupees."""
    return cur.money(gbp)


def rs(gbp):
    """GBP -> display-currency number (for tables and charts)."""
    return None if gbp is None else round(cur.to_display(gbp))
CAT = catalogue()["retrofits"]
NS = "not_sure"

# Official EPC band colours (domain convention), always shown with the letter.
EPC_COLORS = {"A": "#008054", "B": "#19b459", "C": "#8dce46", "D": "#ffd500", "E": "#fcaa65", "F": "#ef8023", "G": "#e9153b"}
CLASS_STYLE = {  # status colours, always shown with the class name
    "Highly efficient": ("#0f7b4f", "🌿"), "Good": ("#3b9a3f", "✅"), "Moderate": ("#b98900", "🟡"),
    "Poor": ("#d2691e", "⚠️"), "Very poor": ("#c62828", "⛔"),
}

LABELS = {
    "not_sure": "Not sure", "yes": "Yes", "no": "No",
    "detached": "Detached house", "semi": "Semi-detached", "terraced": "Mid-terrace", "end_terrace": "End-terrace",
    "flat": "Flat", "bungalow": "Bungalow",
    "ground": "Ground floor", "middle": "Middle floor", "top": "Top floor",
    "solid": "Solid", "cavity_unfilled": "Cavity, not filled", "cavity_filled": "Cavity, filled",
    "system_build": "System-built", "solid_insulated": "Solid, insulated",
    "none_or_flat": "None / flat roof", "under_100": "Under 100 mm", "100_250": "100 to 250 mm", "over_250": "Over 250 mm",
    "suspended": "Suspended timber", "insulated": "Insulated",
    "single": "Single glazing", "double_old": "Double (before 2002)", "double_modern": "Double (2002 or later)",
    "triple": "Triple glazing", "mixed": "Mixed",
    "gas_boiler": "Gas boiler", "oil_boiler": "Oil boiler", "lpg_boiler": "LPG boiler", "electric": "Electric heaters",
    "heat_pump": "Heat pump", "other": "Other / solid fuel",
    "under_5": "Under 5 years", "5_10": "5 to 10 years", "10_15": "10 to 15 years", "over_15": "Over 15 years",
    "none": "None", "programmer": "Timer / programmer", "smart_thermostat": "Smart thermostat",
    "smart_trvs": "Smart radiator valves / zoning",
    "S": "South", "SE_SW": "South-east / south-west", "E_W": "East / west", "N": "North",
    "conservation": "Conservation area", "listed": "Listed building",
    "own": "I own it", "rent": "I rent it",
    "value_for_money": "Best value for money", "cut_bills": "Lower my bills", "cut_carbon": "Cut my carbon",
    "reach_c": "Reach EPC band C", "comfort": "A warmer, more comfortable home",
    "high": "Happy with bigger works", "medium": "Some disruption is fine", "low": "Minimal disruption",
    "quick_wins": "Quick wins", "fabric_first": "Insulation first", "heat_pump_ready": "Heat-pump ready",
    "solar_and_smart": "Solar + smart controls", "full_green": "Everything suitable",
    "electricity": "Electricity", "gas": "Gas", "oil": "Oil", "lpg": "LPG",
}
FIELD_LABELS = {
    "storeys": "Number of storeys", "occupants": "People living there", "bedrooms": "Bedrooms", "wall_type": "Walls",
    "loft": "Loft insulation", "floor_type": "Floor", "glazing": "Glazing", "draught_proofed": "Draught-proofing",
    "boiler_age": "Boiler age", "controls": "Heating controls", "hot_water": "Hot water", "gas_cooking": "Gas cooking",
    "roof_type": "Roof type", "outdoor_space": "Outdoor space", "flat_position": "Flat position",
}
TIER_TEXT = {
    "A": ("High confidence", "based on 9+ months of your meter readings"),
    "B": ("Medium confidence", "based on partial usage or bills"),
    "C": ("Provisional", "no usage given, so energy use is estimated by our model"),
}
BASIS_TEXT = {"recorded": "from your answers", "metered": "from your meter readings", "estimated": "estimated",
              "modelled": "modelled", "default": "assumed"}


def lbl(x) -> str:
    if x is None:
        return "–"
    return LABELS.get(x, str(x).replace("_", " ").capitalize())


st.markdown("""
<style>
.block-container {padding-top: 2.2rem; padding-bottom: 2rem; max-width: 1300px;}
[data-testid="stMetricValue"] {font-size: 1.65rem;}
[data-testid="stMetricLabel"] p {font-size: 0.9rem; opacity: 0.85;}
.ghe-hero {padding: 0.2rem 0 0.6rem 0;}
.ghe-hero h1 {margin: 0; font-size: 2.1rem;}
.ghe-hero p {margin: 0.2rem 0 0 0; opacity: 0.8; font-size: 1.02rem;}
.ghe-badge {display: inline-block; padding: 0.18rem 0.65rem; border-radius: 999px; font-weight: 600;
            font-size: 0.9rem; color: #fff; margin-right: 0.35rem;}
.ghe-epc {display: inline-flex; align-items: center; justify-content: center; width: 3.2rem; height: 3.2rem;
          border-radius: 10px; font-size: 1.9rem; font-weight: 800; color: #111;}
.ghe-card-title {font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.04em; opacity: 0.7; margin-bottom: 0.15rem;}
.ghe-big {font-size: 1.9rem; font-weight: 700; line-height: 1.15;}
.ghe-muted {opacity: 0.72; font-size: 0.9rem;}
.ghe-step {display: inline-block; padding: 0.3rem 0.7rem; margin: 0.15rem 0.2rem 0.15rem 0; border-radius: 8px;
           background: rgba(42,120,214,0.10); border: 1px solid rgba(42,120,214,0.35); font-size: 0.92rem;}
.ghe-step b {color: #2a78d6; margin-right: 0.3rem;}
</style>
""", unsafe_allow_html=True)
if DARK:
    st.markdown("""<style>
.ghe-step {background: rgba(57,135,229,0.16); border-color: rgba(57,135,229,0.45);}
.ghe-step b {color: #7fb3f0;}
.ghe-muted {opacity: 0.78;}
</style>""", unsafe_allow_html=True)

EXAMPLES = {
    "1950s semi, gas, metered (Tier A)": dict(
        postcode="LS6 2AB", property_type="semi", age_band="D", floor_area=90, bedrooms=3, wall_type="cavity_unfilled",
        loft="under_100", floor_type="suspended", glazing="double_old", main_heating="gas_boiler", boiler_age="10_15", controls="programmer",
        tenure="own", epc_band="D", outdoor_space="yes", current_value_estimate=250000,
        electricity=Usage(monthly_kwh=[320, 290, 280, 250, 230, 210, 200, 205, 230, 260, 300, 330]),
        gas=Usage(monthly_kwh=[2100, 1800, 1600, 1000, 600, 250, 180, 190, 350, 900, 1500, 2000])),
    "Victorian terrace, single glazing, no usage (Tier C)": dict(
        postcode="M14 5RT", property_type="terraced", age_band="A", floor_area=85, bedrooms=3, wall_type="solid", loft="under_100",
        glazing="single", main_heating="gas_boiler", boiler_age="over_15", controls="programmer", tenure="own", epc_band="E"),
    "Rented flat, electric heating, bills only (Tier B)": dict(
        postcode="E8 3PL", property_type="flat", age_band="E", floor_area=55, bedrooms=2, flat_position="top", main_heating="electric",
        controls="none", tenure="rent", glazing="double_old",
        electricity=Usage(monthly_bills_gbp=[160, 150, 130, 100, None, None, 60, 60, 80, 110, 140, 165])),
    "Conflict: heat pump with 12,000 kWh gas": dict(
        postcode="BS7 8AA", property_type="detached", age_band="H", floor_area=140, main_heating="heat_pump", tenure="own",
        gas=Usage(annual_kwh=12000)),
}


def _k(key: str) -> str:
    """Widget key scoped to the loaded example, so loading another example resets the form."""
    return f"{st.session_state.get('example', '')}::{key}"


def _opt(label, options, value, key, help_=None):
    idx = options.index(value) if value in options else 0
    return st.selectbox(label, options, index=idx, key=_k(key), help=help_, format_func=lbl)


def input_form(defaults: dict) -> CustomerInput | None:
    d = defaults
    with st.sidebar:
        st.markdown("### 🏡 Your home")
        st.caption("Answer what you know. Every question has a **Not sure** option, and we fill gaps with typical values.")
        with st.form("home"):
            with st.expander("📍 Location and basics", expanded=True):
                postcode = st.text_input("Postcode", d.get("postcode", ""), help="Used for your region's weather and solar potential")
                ptype = _opt("Property type", ["detached", "semi", "terraced", "end_terrace", "flat", "bungalow"], d.get("property_type"), "pt")
                age = st.selectbox("When was it built?", list(AGE_BAND_LABELS),
                                   index=list(AGE_BAND_LABELS).index(d.get("age_band", "D")) if d.get("age_band") else 3,
                                   format_func=lambda a: AGE_BAND_LABELS[a], key=_k("age"))
                area = st.number_input("Floor area (m²)", 15.0, 1500.0, float(d.get("floor_area", 90)), key=_k("area"),
                                       help="Total of all floors. It's on your EPC, or roughly 25 m² per room.")
                bedrooms = st.number_input("Bedrooms", 1, 10, int(d.get("bedrooms", 3)), key=_k("bed"))
                flat_pos = _opt("Which floor is the flat on?", ["not_sure", "ground", "middle", "top"], d.get("flat_position", NS), "fp") if ptype == "flat" else NS
            with st.expander("🧱 Walls, loft, floor and windows", expanded=True):
                wall = _opt("Walls", ["solid", "cavity_unfilled", "cavity_filled", "system_build", "solid_insulated", NS], d.get("wall_type", NS), "wall",
                            "Decides whether wall insulation is possible")
                loft = _opt("Loft insulation depth", ["none_or_flat", "under_100", "100_250", "over_250", NS], d.get("loft", NS), "loft",
                            "A quick look in the loft is enough; 270 mm is the recommended depth")
                floor = _opt("Ground floor", ["solid", "suspended", "insulated", NS], d.get("floor_type", NS), "floor")
                glazing = _opt("Windows", ["single", "double_old", "double_modern", "triple", "mixed", NS], d.get("glazing", NS), "glz")
            with st.expander("🔥 Heating", expanded=True):
                heating = _opt("Main heating", ["gas_boiler", "oil_boiler", "lpg_boiler", "electric", "heat_pump", "other"], d.get("main_heating"), "heat")
                boiler_age = _opt("Boiler age", ["under_5", "5_10", "10_15", "over_15", NS], d.get("boiler_age", NS), "bage")
                controls = _opt("Heating controls", ["none", "programmer", "smart_thermostat", "smart_trvs", NS], d.get("controls", NS), "ctl")
            with st.expander("⚡ Energy use (last 12 months)", expanded=True):
                st.caption("Strongly recommended: it makes results much more accurate. Leave at 0 if you don't know.")
                elec_annual = st.number_input("Electricity used (kWh per year)", 0.0, 50000.0, float(_annual(d.get("electricity"))), key=_k("ea"),
                                              help="On your bill or energy app. 0 = unknown.")
                gas_annual = st.number_input("Gas used (kWh per year)", 0.0, 80000.0, float(_annual(d.get("gas"))), key=_k("ga"),
                                             help="On your bill or energy app. 0 = unknown.")
            with st.expander("☀️ Solar and outdoor space"):
                solar = st.number_input("Existing solar panels (kWp, 0 = none)", 0.0, 20.0, float(d.get("solar_kwp", 0.0)), key=_k("pv"))
                battery = st.checkbox("I have a home battery", d.get("battery", False), key=_k("bat"))
                orientation = _opt("Main roof faces", [NS, "S", "SE_SW", "E_W", "N"], d.get("orientation", NS), "or")
                outdoor = _opt("Outdoor space for a heat pump unit", [NS, "yes", "no"], d.get("outdoor_space", NS), "out")
            with st.expander("📄 About the property"):
                heritage = _opt("Listed building or conservation area?", [NS, "no", "conservation", "listed"], d.get("heritage", NS), "her")
                tenure = _opt("Do you own or rent?", ["own", "rent"], d.get("tenure", "own"), "ten")
                epc = _opt("EPC rating (if you know it)", [NS, "A", "B", "C", "D", "E", "F", "G"], d.get("epc_band", NS), "epc")
                value = st.number_input(f"Current value estimate {SYM} (0 = skip)", 0.0, 2e9, float(rs(d.get("current_value_estimate")) or 0),
                                        step=100000.0, key=_k("val"))
            with st.expander("🎯 Your goals"):
                budget = st.number_input(f"Budget {SYM} (0 = not sure)", 0.0, 5e7, float(rs(d.get("budget")) or 0), step=25000.0, key=_k("bud"))
                goal = _opt("Main goal", ["value_for_money", "cut_bills", "cut_carbon", "reach_c", "comfort"], d.get("goal", "value_for_money"), "goal")
                disruption = _opt("How much disruption is OK?", ["high", "medium", "low"], d.get("disruption_tolerance", "high"), "dis")
                past = st.multiselect("Already done", list(CAT), d.get("past_retrofits", []), format_func=lambda r: CAT[r]["name"], key=_k("past"))
                declined = st.multiselect("Not interested in", list(CAT), d.get("declined", []), format_func=lambda r: CAT[r]["name"], key=_k("dec"))
            submitted = st.form_submit_button("Assess my home  →", type="primary", use_container_width=True)
    if not submitted and "inp" in st.session_state:
        return st.session_state["inp"]
    if not submitted:
        return None
    elec = d.get("electricity") if elec_annual == _annual(d.get("electricity")) and d.get("electricity") else (Usage(annual_kwh=elec_annual) if elec_annual > 0 else None)
    gas = d.get("gas") if gas_annual == _annual(d.get("gas")) and d.get("gas") else (Usage(annual_kwh=gas_annual) if gas_annual > 0 else None)
    return CustomerInput(
        customer_id=st.session_state.get("customer_id", "demo-customer"), postcode=postcode, property_type=ptype, age_band=age,
        floor_area=area, bedrooms=int(bedrooms), flat_position=flat_pos, wall_type=wall, loft=loft, floor_type=floor, glazing=glazing,
        main_heating=heating, boiler_age=boiler_age, controls=controls, electricity=elec, gas=gas, solar_kwp=solar, battery=battery,
        orientation=orientation, outdoor_space=outdoor, heritage=heritage, tenure=tenure, epc_band=epc,
        current_value_estimate=cur.from_display(value) if value else None, budget=cur.from_display(budget) if budget else None,
        goal=goal, disruption_tolerance=disruption,
        past_retrofits=past, declined=declined,
    )


def _annual(u) -> float:
    if u is None:
        return 0.0
    if u.annual_kwh:
        return float(u.annual_kwh)
    if u.monthly_kwh:
        vals = [x for x in u.monthly_kwh if x is not None]
        return round(sum(vals) * 12 / len(vals))
    return 0.0


def _fig_layout(fig, height=320, legend=True):
    fig.update_layout(height=height, margin=dict(l=8, r=16, t=36, b=8), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      hoverlabel=dict(namelength=-1), bargap=0.35, font=dict(size=13), showlegend=legend,
                      legend=dict(orientation="h", y=1.12, x=0, title=None))
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=True, zerolinecolor=GRID)
    return fig


def _gbp(x) -> str:
    """Format an engine (GBP) amount in rupees."""
    return money(x)


def _rng(q, unit=""):
    if not q:
        return ""
    if round(q[0]) == 0 and round(q[2]) == 0:
        return "–"
    return f"{q[0]:,.0f} to {q[2]:,.0f}{unit}"


def class_badge(cls: str) -> str:
    color, icon = CLASS_STYLE.get(cls, ("#666", ""))
    return f'<span class="ghe-badge" style="background:{color}">{icon} {cls}</span>'


def epc_badge(band: str) -> str:
    return f'<span class="ghe-epc" style="background:{EPC_COLORS.get(band, "#ddd")}">{band}</span>'


def estimate_banner(res):
    tier = res["confidence"]["tier"]
    head, why = TIER_TEXT.get(tier, (f"Tier {tier}", res["confidence"]["label"]))
    st.info(f"**{head}** ({why}). All figures are estimates from a simulation model, not guarantees. "
            "A qualified assessor should confirm before any work.", icon="ℹ️")


# ------------------------------------------------------------------ screens

def score_gauge(score: float, cls: str):
    color = CLASS_STYLE.get(cls, ("#2a78d6", ""))[0]
    a = 0.26 if DARK else 0.13
    bands = [(0, 30, f"rgba(198,40,40,{a})"), (30, 50, f"rgba(210,105,30,{a})"), (50, 65, f"rgba(185,137,0,{a})"),
             (65, 80, f"rgba(59,154,63,{a})"), (80, 100, f"rgba(15,123,79,{a + 0.03})")]
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score, number=dict(suffix=" / 100", font=dict(size=34)),
        gauge=dict(axis=dict(range=[0, 100], tickvals=[0, 30, 50, 65, 80, 100], tickfont=dict(size=11)),
                   bar=dict(color=color, thickness=0.32),
                   steps=[dict(range=[a, b], color=c) for a, b, c in bands], borderwidth=0),
    ))
    fig.update_layout(height=210, margin=dict(l=24, r=24, t=16, b=0), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def screen_overview(res):
    cur = res["score"]["current"]
    band = res["rating"]["recorded_band"] or res["rating"]["simulated"]["band"]
    b = res["baseline"]

    c1, c2, c3 = st.columns([1.25, 1, 1])
    with c1:
        with st.container(border=True):
            st.markdown('<div class="ghe-card-title">Green Home Score</div>', unsafe_allow_html=True)
            st.plotly_chart(score_gauge(cur["score_rounded"], cur["class"]), width="stretch", config={"displayModeBar": False})
            prov = ' <span class="ghe-muted">· provisional (some inputs estimated)</span>' if cur["provisional"] else ""
            st.markdown(f"<div style='text-align:center'>{class_badge(cur['class'])}{prov}</div>", unsafe_allow_html=True)
    with c2:
        with st.container(border=True):
            st.markdown('<div class="ghe-card-title">EPC-style rating</div>', unsafe_allow_html=True)
            src = "your recorded EPC" if res["rating"]["basis"] == "recorded" else "our estimate"
            st.markdown(f"{epc_badge(band)} <span class='ghe-muted' style='margin-left:0.6rem'>from {src}</span>", unsafe_allow_html=True)
            st.markdown("")
            st.markdown('<div class="ghe-card-title">Energy use per m²</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="ghe-big">{b["intensity_kwh_per_m2"]:,.0f} kWh</div><div class="ghe-muted">per m² per year</div>', unsafe_allow_html=True)
    with c3:
        with st.container(border=True):
            st.markdown('<div class="ghe-card-title">Yearly energy bill</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="ghe-big">{money(b["bill_gbp"])}</div><div class="ghe-muted">unit costs, excluding standing charges</div>', unsafe_allow_html=True)
            st.markdown("")
            st.markdown('<div class="ghe-card-title">Carbon footprint</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="ghe-big">{b["carbon_kg"] / 1000:.1f} t CO₂e</div><div class="ghe-muted">per year</div>', unsafe_allow_html=True)

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown("#### Why this score")
        st.caption("Each bar shows the points this home earns out of the points available for that part of the score.")
        comps = cur["components"]
        names = [f"{c['label']}  ({c['weight_pct']:.0f})" for c in comps][::-1]
        earned = [c["contribution"] for c in comps][::-1]
        avail = [c["weight_pct"] for c in comps][::-1]
        fig = go.Figure()
        fig.add_bar(y=names, x=earned, orientation="h", name="Points earned", marker=dict(color=ELEC, cornerradius=4),
                    customdata=[[c["sub_score"], c["input"], BASIS_TEXT.get(c["basis"], c["basis"])] for c in comps][::-1],
                    hovertemplate="%{y}<br>%{customdata[1]}<br>sub-score %{customdata[0]:.0f}/100 (%{customdata[2]})<extra></extra>")
        fig.add_bar(y=names, x=[a - e for a, e in zip(avail, earned)], orientation="h", name="Points missed",
                    marker=dict(color=TRACK), hoverinfo="skip",
                    text=[f"<b>{e:.1f}</b> / {a:.0f}" for e, a in zip(earned, avail)], textposition="outside", cliponaxis=False)
        fig.update_layout(barmode="stack", xaxis=dict(range=[0, 30], title="points"), bargap=0.3, legend=dict(traceorder="normal"))
        st.plotly_chart(_fig_layout(fig, 330), width="stretch", config={"displayModeBar": False})
        with st.expander("Show the score table"):
            st.dataframe(pd.DataFrame([{
                "Part of score": c["label"], "Weight (%)": c["weight_pct"], "Sub-score (0-100)": round(c["sub_score"]),
                "Points earned": round(c["contribution"], 1), "Based on": c["input"].replace("_", " "),
                "Source": BASIS_TEXT.get(c["basis"], c["basis"]),
            } for c in comps]), hide_index=True, width="stretch")
    with right:
        st.markdown("#### Main issues")
        if res["efficiency_flag"]["is_poor"]:
            st.warning(f"This home is rated **{cur['class']}**. The issues below cost it the most points.", icon="⚠️")
        for i in res["efficiency_flag"]["key_issues"]:
            with st.container(border=True):
                st.markdown(f"**{i['issue']}**  \n<span class='ghe-muted'>{i['detail'].replace('_', ' ')} · "
                            f"{i['points_lost']:.1f} points lost</span>", unsafe_allow_html=True)
        ip = res["efficiency_flag"]["improvement_potential"]
        with st.container(border=True):
            st.markdown('<div class="ghe-card-title">Improvement potential</div>', unsafe_allow_html=True)
            if ip.get("pct") is not None:
                st.markdown(f'<div class="ghe-big">{ip["pct"]:.0f}%</div><div class="ghe-muted">less grid electricity + gas '
                            'with every suitable improvement</div>', unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='ghe-muted'>{ip.get('reason', 'not available')}</div>", unsafe_allow_html=True)
            if ip.get("note"):
                st.caption(ip["note"])

    st.markdown("#### Your energy use today")
    e1, e2 = st.columns([2, 3], gap="large")
    with e1:
        fuels = [f for f, v in b["purchased_kwh"].items() if v]
        fig = go.Figure(go.Bar(x=[b["purchased_kwh"][f] for f in fuels], y=[lbl(f) for f in fuels], orientation="h",
                               marker=dict(color=[FUEL_COLOR.get(f, OTHER) for f in fuels], cornerradius=4),
                               text=[f"{b['purchased_kwh'][f]:,.0f} kWh" for f in fuels], textposition="outside",
                               hovertemplate="%{y}: %{x:,.0f} kWh/yr<extra></extra>"))
        fig.update_traces(cliponaxis=False)
        fig.update_layout(xaxis=dict(title="kWh per year (bought from the grid / supplier)",
                                     range=[0, max([b["purchased_kwh"][f] for f in fuels] + [1]) * 1.25]))
        st.plotly_chart(_fig_layout(fig, 60 + 70 * len(fuels), legend=False), width="stretch", config={"displayModeBar": False})
    with e2:
        rows = [{"Fuel": lbl(r["fuel"]), "Your readings (yearly)": r["metered_kwh"], "Weather-adjusted": r["normalised_kwh"],
                 "Model after calibration": r["model_kwh"]} for r in b["reconciliation"]]
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                         column_config={c: st.column_config.NumberColumn(format="localized") for c in ("Your readings (yearly)", "Weather-adjusted", "Model after calibration")})
            st.caption("We scale your readings to a full year, adjust for weather, then tune the model to your home so the "
                       "before and after figures line up.")
        else:
            st.caption("No usage was given, so this is our model's estimate for a home like yours. Add your meter readings for a more accurate result.")
    if res["confidence"]["warnings"]:
        with st.expander(f"Notes about this estimate ({len(res['confidence']['warnings'])})"):
            for w in res["confidence"]["warnings"]:
                st.markdown(f"- {w}")


def screen_planner(res, inp):
    st.markdown("#### Improvements that suit this home")
    st.caption("Listed in the order they should be done: insulation first, then heating, then solar, then controls. "
               "Each saving assumes the steps above it are already done, so standalone savings are not additive.")
    rows = res["retrofits"]["feasible_list"]
    suit_icon = {"High": "🟢 High", "Medium": "🟡 Medium", "Low": "⚪ Low"}
    table = []
    for r in rows:
        s = r["saving_in_sequence_kwh"]
        table.append({
            "#": r["rank"], "Improvement": r["name"], "Suitability": suit_icon.get(r["suitability"], r["suitability"]),
            "Electricity saved (kWh/yr)": s["electricity"], "Gas saved (kWh/yr)": s["gas"],
            f"Bill saving ({SYM}/yr)": rs(r["bill_saving_gbp"]),
            "Payback": f"{r['payback_years']} yrs" if r["payback_years"] is not None else "–",
            f"Cost ({SYM})": f"{money(r['cost_gbp'][0])} to {money(r['cost_gbp'][2])}", "Stage": r["tier_name"],
            "Do first": ", ".join(r["needs_first"]) or "–", "Why": r["reason"],
        })
    if table:
        st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch", column_config={
            "#": st.column_config.NumberColumn(width="small"),
            "Electricity saved (kWh/yr)": st.column_config.NumberColumn(format="localized"),
            "Gas saved (kWh/yr)": st.column_config.NumberColumn(format="localized"),
            f"Bill saving ({SYM}/yr)": st.column_config.NumberColumn(format="localized"),
            "Why": st.column_config.TextColumn(width="large"),
        })
        st.caption("Negative numbers mean use goes up (for example, a heat pump uses more electricity while cutting gas). "
                   "Payback '–' means it doesn't lower bills at today's prices.")

        c1, c2 = st.columns([3, 2], gap="large")
        with c1:
            names = [r["name"] for r in rows][::-1]
            vals = [rs(r["bill_saving_gbp"]) for r in rows][::-1]
            fig = go.Figure(go.Bar(x=vals, y=names, orientation="h",
                                   marker=dict(color=[ELEC if v >= 0 else GAS for v in vals], cornerradius=4),
                                   text=[cur.fmt(v) for v in vals], textposition="outside",
                                   hovertemplate="%{y}: " + SYM + "%{x:,.0f}/yr<extra></extra>"))
            lo, hi = min(vals + [0]), max(vals + [0])
            pad = 0.22 * max(hi - lo, 1)
            fig.update_traces(cliponaxis=False)
            fig.update_layout(title=dict(text="Yearly bill saving per step (blue saves, orange costs more)", font=dict(size=14)),
                              xaxis=dict(title=f"{SYM} per year", tickprefix=SYM, range=[lo - (pad if lo < 0 else 0), hi + pad]))
            st.plotly_chart(_fig_layout(fig, 70 + 42 * len(rows), legend=False), width="stretch", config={"displayModeBar": False})
        with c2:
            pk = res["retrofits"]["package_row"]
            if pk.get("valid"):
                with st.container(border=True):
                    st.markdown('<div class="ghe-card-title">If you did everything on the list</div>', unsafe_allow_html=True)
                    m1, m2 = st.columns(2)
                    m1.metric("Electricity", f"{pk['delta_kwh']['electricity']:,.0f} kWh/yr", help="Saved per year (negative = more used)")
                    m2.metric("Gas", f"{pk['delta_kwh']['gas']:,.0f} kWh/yr", help="Saved per year")
                    m3, m4 = st.columns(2)
                    m3.metric("Bill saving", f"{_gbp(pk['bill_saving_gbp'])}/yr")
                    m4.metric("Carbon saved", f"{pk['carbon_saving_kg'] / 1000:.1f} t/yr")
                    st.markdown(f"Cost about **{money(pk['cost_gbp'][1])}** · EPC-style band after: {epc_badge(pk['band_after'])}",
                                unsafe_allow_html=True)
                    st.caption("Simulated as one package, so interactions between improvements are included.")
    na = res["retrofits"]["not_available"]
    if na:
        with st.expander(f"Not suitable for this home ({len(na)})"):
            for n in na:
                st.markdown(f"- **{n['name']}**: {n['reason']}")

    st.markdown("#### Tailor the plan")
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4)
        goals = ["value_for_money", "cut_bills", "cut_carbon", "reach_c", "comfort"]
        goal = c1.selectbox("Main goal", goals, index=goals.index(inp.goal), format_func=lbl, key=_k("cg"))
        budget_rs = c2.number_input(f"Budget {SYM} (0 = no limit)", 0.0, 5e7, float(rs(inp.budget) or 0), step=25000.0, key=_k("cb"))
        budget = cur.from_display(budget_rs) if budget_rs else None
        dis = c3.selectbox("Disruption", ["high", "medium", "low"], format_func=lbl, key=_k("cd"))
        preset = c4.selectbox("Quick preset", ["none", "quick_wins", "fabric_first", "heat_pump_ready", "solar_and_smart", "full_green"],
                              format_func=lambda x: "No preset" if x == "none" else lbl(x), key=_k("cp"))
        feasible_ids = [r["retrofit"] for r in rows]
        t1, t2 = st.columns(2)
        tick = t1.multiselect("Must include", feasible_ids, format_func=lambda r: CAT[r]["name"], key=_k("ct"))
        untick = t2.multiselect("Leave out", feasible_ids, format_func=lambda r: CAT[r]["name"], key=_k("cu"))
        grants = st.checkbox("Include the heat pump grant (check you are eligible)", key=_k("cgr"))
    opts = CustomiseOptions(goal=goal, budget=budget or None, disruption_tolerance=dis, tick=tick, untick=untick,
                            preset=None if preset == "none" else preset, declined=inp.declined, apply_grants=grants)
    st.session_state["opts"] = opts
    return opts


def screen_results(res):
    plan = res["plan"]
    estimate_banner(res)
    for w in plan["warnings"]:
        st.warning(w)
    if plan.get("message"):
        st.info(plan["message"])
    best = plan.get("best")
    if not best:
        if plan.get("cheapest_useful"):
            st.write("Cheapest useful option:", ", ".join(plan["cheapest_useful"]["names"]))
        return

    st.markdown(f"#### Recommended plan · goal: {lbl(plan['goal']).lower()}")
    steps = "".join(f'<span class="ghe-step"><b>{i}</b>{n}</span>' for i, n in enumerate(best["names"], 1))
    st.markdown(steps, unsafe_allow_html=True)
    st.markdown("")
    c = st.columns(5)
    br = best.get("bill_saving_range_gbp")
    cr = best.get("carbon_saving_range_kg")
    c[0].metric("Bill saving", f"{_gbp(best['bill_saving_gbp'])}/yr", help="Middle estimate; the range below is the likely spread (10th to 90th percentile)")
    if br:
        c[0].caption(f"likely {_gbp(br[0])} to {_gbp(br[2])}")
    c[1].metric("Upfront cost", money(best["cost_gbp"][1]))
    c[1].caption(f"range {money(best['cost_gbp'][0])} to {money(best['cost_gbp'][2])}")
    c[2].metric("Pays for itself in", f"{best['payback_years']} yrs" if best["payback_years"] else "–")
    c[2].caption("simple payback" if best["payback_years"] else "doesn't lower bills at today's prices")
    c[3].metric("Carbon saved", f"{best['carbon_saving_kg'] / 1000:.2f} t/yr")
    if cr:
        c[3].caption(f"likely {cr[0] / 1000:.2f} to {cr[2] / 1000:.2f} t")
    sa = best.get("score_after") or {}
    cur_score = res["score"]["current"]["score_rounded"]
    c[4].metric("Green Home Score after", f"{sa.get('score_rounded', '–')} / 100",
                f"{sa.get('score_rounded', cur_score) - cur_score:+d} points" if sa else None)
    if best.get("electricity_rises_note"):
        st.warning(best["electricity_rises_note"], icon="⚡")

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("##### Energy bought each year: today vs after the plan")
        fuels = [f for f in ("electricity", "gas", "oil", "lpg", "solid") if best["baseline_purchased_kwh"][f] or best["delta_kwh"][f]]
        fig = go.Figure()
        fig.add_bar(name="Today", y=[lbl(f) for f in fuels], x=[best["baseline_purchased_kwh"][f] for f in fuels], orientation="h",
                    marker=dict(color=TODAY, cornerradius=4), text=[f"{best['baseline_purchased_kwh'][f]:,.0f}" for f in fuels],
                    textposition="outside", hovertemplate="%{y} today: %{x:,.0f} kWh<extra></extra>")
        fig.add_bar(name="After the plan", y=[lbl(f) for f in fuels], x=[best["after_purchased_kwh"][f] for f in fuels], orientation="h",
                    marker=dict(color=ELEC, cornerradius=4), text=[f"{best['after_purchased_kwh'][f]:,.0f}" for f in fuels],
                    textposition="outside", hovertemplate="%{y} after: %{x:,.0f} kWh<extra></extra>")
        top = max([best["baseline_purchased_kwh"][f] for f in fuels] + [best["after_purchased_kwh"][f] for f in fuels] + [1])
        fig.update_traces(cliponaxis=False)
        fig.update_layout(barmode="group", xaxis=dict(title="kWh per year", range=[0, top * 1.18]), yaxis=dict(autorange="reversed"))
        st.plotly_chart(_fig_layout(fig, 110 + 90 * len(fuels)), width="stretch", config={"displayModeBar": False})
        df = pd.DataFrame({
            "Fuel": [lbl(f) for f in fuels],
            "Today (kWh/yr)": [best["baseline_purchased_kwh"][f] for f in fuels],
            "After (kWh/yr)": [best["after_purchased_kwh"][f] for f in fuels],
            "Saved (kWh/yr)": [best["delta_kwh"][f] for f in fuels],
            "Likely range saved": [_rng((best.get("delta_kwh_range") or {}).get(f)) for f in fuels],
        })
        st.dataframe(df, hide_index=True, width="stretch",
                     column_config={k: st.column_config.NumberColumn(format="localized") for k in ("Today (kWh/yr)", "After (kWh/yr)", "Saved (kWh/yr)")})
        st.caption("Negative 'saved' means that fuel goes up.")
    with right:
        ylab = {"cut_carbon": "carbon saved (kg/yr)", "comfort": "heat loss reduced (W/K)", "reach_c": "EPC-style points"}.get(plan["goal"], f"bill saving ({SYM}/yr)")
        st.markdown(f"##### Every possible package: cost vs {ylab.split(' (')[0]}")
        fr = pd.DataFrame(plan["frontier"])
        if not fr.empty:
            fig = go.Figure()
            money_y = plan["goal"] not in ("cut_carbon", "comfort", "reach_c")
            fr = fr.assign(cost_rs=fr.cost_gbp * cur.rate(), y=fr.saving * (cur.rate() if money_y else 1))
            for label, mask, color, size in (("Other packages", ~fr.on_frontier & ~fr.is_best, SOFT_BLUE, 7),
                                             ("Best for the money", fr.on_frontier & ~fr.is_best, ELEC, 10),
                                             ("Recommended", fr.is_best, GAS, 16)):
                sub = fr[mask]
                fig.add_scatter(x=sub.cost_rs, y=sub.y, mode="markers", name=label,
                                marker=dict(color=color, size=size, line=dict(width=2, color=RING)),
                                text=[", ".join(CAT[r]["name"] for r in ids) or "nothing" for ids in sub.retrofits],
                                hovertemplate="%{text}<br>cost " + SYM + "%{x:,.0f}<br>" + ylab + ": %{y:,.0f}<extra></extra>")
            bp = fr[fr.is_best]
            if not bp.empty:
                fig.add_annotation(x=bp.cost_rs.iloc[0], y=bp.y.iloc[0], text="Recommended", showarrow=True,
                                   arrowhead=2, ax=-50, ay=-35, font=dict(size=12))
            fig.update_layout(xaxis=dict(title=f"upfront cost ({SYM})", tickprefix=SYM),
                              yaxis=dict(title=ylab, tickprefix=SYM if money_y else ""))
            st.plotly_chart(_fig_layout(fig, 380), width="stretch", config={"displayModeBar": False})
            st.caption("Each dot is one combination of improvements we simulated. Hover a dot to see what's in it.")

    st.markdown("##### Step by step")
    st.dataframe(pd.DataFrame([{"Step": r["rank"], "Improvement": r["name"], "Electricity saved (kWh/yr)": r["saving_in_sequence_kwh"]["electricity"],
                                "Gas saved (kWh/yr)": r["saving_in_sequence_kwh"]["gas"], f"Bill saving ({SYM}/yr)": rs(r["bill_saving_gbp"]),
                                "Carbon saved (kg/yr)": r["carbon_saving_kg"], "Notes": " ".join(r["flags"]) or "–"} for r in best["rows"]]),
                 hide_index=True, width="stretch",
                 column_config={k: st.column_config.NumberColumn(format="localized") for k in
                                ("Electricity saved (kWh/yr)", "Gas saved (kWh/yr)", f"Bill saving ({SYM}/yr)", "Carbon saved (kg/yr)")})
    if plan["optional"]:
        with st.expander(f"Optional extras on top of this plan ({len(plan['optional'])})"):
            st.dataframe(pd.DataFrame([{"Improvement": o["name"], f"Extra bill saving ({SYM}/yr)": rs(o.get("extra_bill_saving_gbp")),
                                        "Extra carbon saved (kg/yr)": o.get("extra_carbon_saving_kg"),
                                        f"Cost ({SYM}, typical)": rs((o.get("cost_gbp") or [None, None])[1]), "Note": o.get("note", "") or "–"}
                                       for o in plan["optional"]]), hide_index=True, width="stretch")

    v = res["value"]
    st.markdown("##### Property value (indicative)")
    with st.container(border=True):
        if v["status"] == "ok":
            a, b2 = st.columns([1, 2])
            a.metric("Possible value change", f"{v['uplift_pct_range'][0]}% to {v['uplift_pct_range'][2]}%")
            txt = f"EPC-style band {v['current_band']} → {v['after_band']}"
            if v.get("uplift_gbp_range"):
                txt += f" · about {money(v['uplift_gbp_range'][0])} to {money(v['uplift_gbp_range'][2])}"
            b2.markdown(txt)
            b2.caption(v["label"])
            if v.get("note"):
                b2.caption(v["note"])
            if v.get("warning"):
                b2.caption("⚠️ " + v["warning"])
        else:
            st.caption(v["reason"])


def screen_whatif(res, inp, opts):
    st.markdown("#### What if prices or the grid change?")
    st.caption("Move the sliders: the best plan is re-picked instantly from the combinations already simulated.")
    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        a_prices = res["assumptions"]["prices"]["unit_rate_gbp_per_kwh"]
        ep = c1.slider(f"Electricity price ({SYM}/kWh)", cur.per_kwh(0.10), cur.per_kwh(0.50), cur.per_kwh(a_prices["electricity"]), 0.5)
        gp = c2.slider(f"Gas price ({SYM}/kWh)", cur.per_kwh(0.03), cur.per_kwh(0.20), cur.per_kwh(a_prices["gas"]), 0.1)
        year = c3.slider("Grid carbon in year", 2024, 2040, 2024, help="Later years use a cleaner electricity grid (illustrative)")
    o = opts.model_copy(update={"electricity_price": cur.from_display(ep), "gas_price": cur.from_display(gp), "grid_carbon_year": year if year != 2024 else None})
    r = analyse(inp, o, with_ranges=False)
    base = (res.get("plan") or {}).get("best")
    if r["status"] == "ok" and r["plan"].get("best"):
        b = r["plan"]["best"]
        st.markdown("**Best plan at these settings:** " + "".join(f'<span class="ghe-step"><b>{i}</b>{n}</span>' for i, n in enumerate(b["names"], 1)),
                    unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        db = b["bill_saving_gbp"] - base["bill_saving_gbp"] if base else 0
        dc = b["carbon_saving_kg"] - base["carbon_saving_kg"] if base else 0
        m1.metric("Bill saving", f"{_gbp(b['bill_saving_gbp'])}/yr",
                  f"{'+' if db >= 0 else '-'}{money(abs(db))} vs current settings" if round(rs(db)) else None)
        m2.metric("Carbon saved", f"{b['carbon_saving_kg']:,.0f} kg/yr", f"{dc:+,.0f} kg vs current settings" if round(dc) else None)
        m3.metric("Upfront cost", money(b["cost_gbp"][1]))
    elif r["status"] == "ok":
        st.info(r["plan"].get("message") or "No improvement helps at these settings.")

    st.markdown("#### Ask a question")
    avail = llm_status()["available"]
    st.caption(("🤖 Answers written by " + ", ".join(avail)) if avail else "🤖 No AI provider configured, so answers use built-in templates.")
    st.caption("Try: *What if I add a heat pump?* · *Can I reach band C for ₹5,00,000?* · *What if gas goes to ₹10 per kWh?*")
    msg = st.chat_input("Ask about your home, e.g. What if I add solar panels?")
    hist = st.session_state.setdefault("chat", [])
    if msg:
        sc, src = parse(msg)
        w = run_whatif(inp, sc, opts, with_ranges=False)
        ex = explain(w["results"], msg, requested=w["requested_infeasible"], diff=w["diff"], question_type=sc.question_type)
        hist.append((msg, ex, sc.model_dump(exclude_defaults=True), src))
    for q, ex, sc, src in hist[::-1]:
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant"):
            st.write(ex["text"])
            ok = "✅ Every number checked against the engine" if ex["verified"] else "⚠️ Some numbers could not be checked"
            st.caption(f"{ok} · answer by {ex['source']}")
            with st.expander("What we changed to answer this"):
                st.json(sc)
                st.caption(f"Question understood by: {src}")


def screen_assumptions(res, inp):
    st.markdown("#### What we assumed")
    defaults = res["confidence"]["defaults_used"]
    if defaults:
        st.caption("You answered 'not sure' or skipped these, so we used typical values. Fill them in for a more accurate result.")
        st.dataframe(pd.DataFrame([{"Question": FIELD_LABELS.get(d["field"], d["field"].replace("_", " ").capitalize()),
                                    "We assumed": (("Yes" if d["assumed"] else "No") if isinstance(d["assumed"], bool)
                                                   else lbl(d["assumed"]) if isinstance(d["assumed"], str) else str(d["assumed"])),
                                    "Why": d["reason"]} for d in defaults]), hide_index=True, width="stretch")
    else:
        st.success("No assumptions needed: every question was answered.")

    st.markdown("#### Prices and carbon factors used")
    a = res["assumptions"]
    p, e = a["prices"]["unit_rate_gbp_per_kwh"], a["emissions"]["kg_co2e_per_kwh"]
    st.dataframe(pd.DataFrame([{"Fuel": lbl(f), f"Price ({SYM}/kWh)": cur.per_kwh(p[f]), "Carbon (kg CO₂e/kWh)": e.get(f)} for f in p]),
                 hide_index=True, width="stretch")
    st.caption(f"Prices: {a['prices']['label']} · Carbon: {a['emissions']['label']} · Solar export paid at "
               f"{SYM}{cur.per_kwh(a['prices']['export_gbp_per_kwh'])}/kWh. Amounts are converted to rupees at a placeholder exchange rate "
               "(data/reference/currency.json); prices and costs are placeholders to be confirmed.")
    with st.expander("All sources, versions and model notes"):
        st.json(a, expanded=False)

    st.markdown("#### Customer memory")
    st.caption("Tell us what you've done or don't want; future plans adapt to it.")
    mem = memory.load(inp.customer_id)
    m1, m2 = st.columns(2)
    m1.markdown("**Completed:** " + (", ".join(CAT[r]["name"] for r in mem.get("completed", []) if r in CAT) or "nothing yet"))
    m2.markdown("**Not interested:** " + (", ".join(CAT[r]["name"] for r in mem.get("declined", []) if r in CAT) or "nothing yet"))
    c1, c2 = st.columns(2)
    done = c1.multiselect("Mark as completed", list(CAT), format_func=lambda r: CAT[r]["name"], key="mdone")
    dec = c2.multiselect("Mark as not interested", list(CAT), format_func=lambda r: CAT[r]["name"], key="mdec")
    if st.button("Save to memory"):
        before = [r["name"] for r in res["retrofits"]["feasible_list"]]
        memory.record(inp.customer_id, done, dec)
        after = analyse(inp, use_memory=True, with_ranges=False)
        st.write("**Ranking before:**", " → ".join(before))
        if after["status"] == "ok":
            st.write("**Ranking after:**", " → ".join(r["name"] for r in after["retrofits"]["feasible_list"]))
            if after["plan"].get("best"):
                st.write("**Plan after (declined items excluded):**", ", ".join(after["plan"]["best"]["names"]))
    st.caption(res["disclaimer"] + " Tenants may need landlord consent.")


def welcome():
    st.markdown("#### How it works")
    c = st.columns(3)
    for col, (icon, title, text) in zip(c, [
        ("📝", "1. Tell us about your home", "Fill in the form on the left, or pick an example home. 'Not sure' is always fine."),
        ("🧮", "2. We simulate it", "A physics model works out your energy use and tests every combination of suitable improvements."),
        ("📊", "3. Get your plan", "Your Green Home Score, what to improve first, savings, carbon and possible value change."),
    ]):
        with col:
            with st.container(border=True):
                st.markdown(f"### {icon}")
                st.markdown(f"**{title}**")
                st.caption(text)


def main():
    st.markdown('<div class="ghe-hero"><h1>🏠 Green Home Engine</h1>'
                '<p>How green is your home, what should you improve, and what could you save?</p></div>', unsafe_allow_html=True)
    ex = st.sidebar.selectbox("Start from an example home", ["(enter my own)"] + list(EXAMPLES))
    defaults = EXAMPLES.get(ex, {})
    if st.session_state.get("example") != ex:
        st.session_state.pop("inp", None)
        st.session_state.pop("opts", None)
        st.session_state.pop("chat", None)
        st.session_state["example"] = ex
    inp = input_form(defaults)
    if inp is None:
        st.info("Fill in the form on the left (or pick an example home) and press **Assess my home**.", icon="👈")
        welcome()
        return
    st.session_state["inp"] = inp
    opts = st.session_state.get("opts")
    with st.spinner("Simulating every suitable combination of improvements…"):
        res = analyse(inp, opts)
    if res["status"] != "ok":
        st.error("We need a little more information before we can give a result.", icon="✋")
        for q in res["questions"]:
            st.markdown(f"- {q['question']}")
        for c in res["conflicts"]:
            st.markdown(f"- {c}")
        st.caption("Update the form on the left and press **Assess my home** again.")
        return
    p = res["property"]
    st.caption(f"{lbl(p['property_type'])} · built {p['age_band_label']} · {p['floor_area_m2']:,.0f} m² · {p['region_label']} · "
               f"{lbl(p['heating'])}")
    tabs = st.tabs(["🏠 Overview", "🛠️ Improvements", "📋 Your plan", "🔮 What if", "📚 Assumptions"])
    with tabs[0]:
        estimate_banner(res)
        screen_overview(res)
        with st.expander("💬 Explain my results in plain English"):
            if st.button("Explain"):
                e = explain(res, "Explain my results")
                st.write(e["text"])
                st.caption(("✅ Every number checked against the engine" if e["verified"] else "⚠️ Some numbers could not be checked")
                           + f" · answer by {e['source']}")
    with tabs[1]:
        new_opts = screen_planner(res, inp)
        if new_opts != opts:
            st.button("Update my plan", on_click=lambda: None, type="primary")
    with tabs[2]:
        screen_results(res)
    with tabs[3]:
        screen_whatif(res, inp, st.session_state.get("opts") or CustomiseOptions(goal=inp.goal, budget=inp.budget))
    with tabs[4]:
        screen_assumptions(res, inp)
    st.caption(f"Engine {res['engine_version']} · planned in {res['timing_ms']['total']:.0f} ms · "
               f"{res['timing_ms']['subsets_cached']} combinations simulated · estimates, not guarantees")


main()
