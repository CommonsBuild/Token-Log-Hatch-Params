from tech import read_impact_hour_data, read_cstk_data
from tech import ImpactHoursData, ImpactHoursFormula, Hatch, DandelionVoting
import pandas as pd
import panel as pn
import holoviews as hv
import numpy as np


react = pn.template.ReactTemplate(title='TEC Hatch Dashboard')
pn.config.sizing_mode = 'stretch_both'

impact_hour_data_1, impact_hour_data_2 = read_impact_hour_data()
impact_hours_data = ImpactHoursData()
# impact_hours_view = pn.Row(impact_hours_data, impact_hours_data.impact_hours_accumulation)

impact_rewards = ImpactHoursFormula(impact_hours_data.total_impact_hours, impact_hour_data_1)
# impact_rewards_view = pn.Row(impact_rewards, pn.Column(impact_rewards.impact_rewards, impact_rewards.funding_pools), impact_rewards.payout_view)


i = ImpactHoursData()
impact_data_view = pn.Row(i.impact_hours_accumulation)

react.main[:4, :6] = impact_data_view

impact_hours_rewards = ImpactHoursFormula(i.total_impact_hours, impact_hour_data_1)
impact_rewards_view = pn.Row(pn.Column(impact_hours_rewards.impact_hours_rewards, impact_hours_rewards.funding_pools), impact_hours_rewards.payout_view)


react.main[4:8, :6] = impact_rewards_view

cstk_data = read_cstk_data()
h = Hatch(cstk_data)
hatch_view = pn.Row( h.hatch_raise_view)

react.main[8:12, :6] = hatch_view

cstk_data = read_cstk_data()
hatch = Hatch(cstk_data)
hatch_view = pn.Row(hatch.hatch_raise_view)


react.sidebar.append(i)
react.sidebar.append(impact_hours_rewards) 
react.sidebar.append(h)
react.sidebar.append(hatch)
# Unlike other templates the `ReactTemplate.main` area acts like a GridSpec 
# react.main[:4, :6] = pn.Card(hv.DynamicMap(sine), title='Sine')
# react.main[:4, 6:] = pn.Card(hv.DynamicMap(cosine), title='Cosine')

react.servable();
