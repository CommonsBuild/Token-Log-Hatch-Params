from tech.tech import read_impact_hour_data, TECH
import panel as pn
import holoviews as hv
import pandas as pd

def test_TECH():
    df, _ = read_impact_hour_data()
    t = TECH(total_impact_hours=500, impact_hour_data=df, total_cstk_tokens=8500)
    pn.Row(pn.Column(t, t.funding_pool_data_view), pn.Column(t.impact_hours_view, t.funding_pool_view, t.payout_view))
    scenarios = t.get_raise_scenarios()
    target_raise = t.target_raise
    rates = t.impact_hours_formula(0, t.min_max_raise[1])
    target_rate_1 = t.target_impact_hour_rate
    target_rate_2 = t.get_impact_hour_rate(raise_amount=t.target_raise)
    assert((target_rate_1 - target_rate_2) < 1)
    funding_pools = t.get_funding_pool_data()
