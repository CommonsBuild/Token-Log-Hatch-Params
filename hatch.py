from operator import index
from tech.tech import read_impact_hour_data, read_cstk_data, TECH
from tech.tech import ImpactHoursData, ImpactHoursFormula, Hatch, DandelionVoting
from jinja2 import Environment, FileSystemLoader
from bokeh.plotting import curdoc
import tech.config_bounds as config_bounds
from dotenv import load_dotenv
import pandas as pd
import panel as pn
import holoviews as hv
import numpy as np
import urllib
import urllib.parse as p
from tabulate import tabulate
import requests
import codecs
import os

load_dotenv()

env = Environment(loader=FileSystemLoader('.'))
template = env.get_template('template/index.html')

# API settings
HCTI_API_ENDPOINT = "https://hcti.io/v1/image"
HCTI_API_USER_ID = os.environ.get('HCTI_API_USER_ID')
HCTI_API_KEY = os.environ.get('HCTI_API_KEY')

pn.config.sizing_mode = 'stretch_both'

impact_hour_data_1, impact_hour_data_2 = read_impact_hour_data()
impact_hours_data = ImpactHoursData()

# ImpactHoursData
i = ImpactHoursData()

# TECH
t = TECH(total_impact_hours=i.total_impact_hours,
         impact_hour_data=impact_hour_data_1, total_cstk_tokens=8500,
         config=config_bounds.hatch['tech'])


# ImpactHoursFormula
#impact_hours_rewards = ImpactHoursFormula(i.total_impact_hours, impact_hour_data_1)
#impact_rewards_view = pn.Column(impact_hours_rewards.impact_hours_rewards,
# impact_hours_rewards.redeemable,
# impact_hours_rewards.cultural_build_tribute)

# Hatch
cstk_data = read_cstk_data()
#hatch = Hatch(cstk_data, impact_hours_rewards.target_raise,
# i.total_impact_hours,
# impact_hours_rewards.target_impact_hour_rate)

# DandelionVoting
dandelion = DandelionVoting(17e6,config=config_bounds.hatch['dandelion_voting'])

# Import Params Button
import_params_button = pn.widgets.Button(name='Import params', button_type = 'primary')
import_description = pn.pane.Markdown('<h4>To import the parameters, click on the button below:</h4>')

# Share Button
comments = pn.widgets.TextAreaInput(name='Comments', max_length=1024, placeholder='Explain your thoughts on why you choose the params...')
share_button = pn.widgets.Button(name='Share your results on GitHub!', button_type = 'primary')
url = pn.widgets.TextInput(name='URL', value = '')
share_button.js_on_click(args={'target': url}, code='window.open(target.value)')
results_button = pn.widgets.Button(name='See your results', button_type = 'success')

def update_params_by_url_query():
    queries = curdoc().session_context.request.arguments
    queries = { i: j[0] for i, j in queries.items() }
    if queries:
        if 'ihminr' in queries and 'ihmaxr' in queries:
            t.min_max_raise = (int(queries['ihminr']), int(queries['ihmaxr']))
        if 'hs' in queries:
            t.impact_hour_slope = float(queries['hs'])
        if 'maxihr' in queries:
            t.maximum_impact_hour_rate = float(queries['maxihr'])
        if 'ihtr' in queries:
            t.target_raise = int(queries['ihtr'])
        if 'hor' in queries:
            t.hatch_oracle_ratio = float(queries['hor'])
        if 'hpd' in queries:
            t.hatch_period_days = int(queries['hpd'])
        if 'her' in queries:
            t.hatch_exchange_rate = float(queries['her'])
        if 'ht' in queries:
            t.hatch_tribute = int(queries['ht'])
        if 'sr' in queries:
            dandelion.support_required_percentage = int(queries['sr'])
        if 'maq' in queries:
            dandelion.minimum_accepted_quorum_percentage = int(queries['maq'])
        if 'vdd' in queries:
            dandelion.vote_duration_days = float(queries['vdd'])
        if 'vbh' in queries:
            dandelion.vote_buffer_hours = float(queries['vbh'])
        if 'rqh' in queries:
            dandelion.rage_quit_hours = float(queries['rqh'])
        if 'tfx' in queries:
            dandelion.tollgate_fee_xdai = float(queries['tfx'])

@pn.depends(results_button)
def update_result_score(results_button):
    data_table = {'Parameters': ["Target raise (wxDai)", "Maximum raise (wxDai)", "Minimum raise (wxDai)",
    "Impact hour slope (wxDai/IH)", "Maximum impact hour rate (wxDai/IH)",
    "Hatch oracle ratio (wxDai/CSTK)", "Hatch period (days)",
    "Hatch exchange rate (TESTTECH/wxDai)", "Hatch tribute (%)", "Support required (%)",
    "Minimum accepted quorum (%)", "Vote duration (days)", "Vote buffer (hours)",
    "Rage quit (hours)", "Tollgate fee (wxDai)"],
    'Values': [int(t.target_raise), int(t.min_max_raise[1]),
    int(t.min_max_raise[0]), t.impact_hour_slope,
    t.maximum_impact_hour_rate, t.hatch_oracle_ratio,
    t.hatch_period_days, t.hatch_exchange_rate, t.hatch_tribute,
    dandelion.support_required_percentage, dandelion.minimum_accepted_quorum_percentage, dandelion.vote_duration_days,
    dandelion.vote_buffer_hours, dandelion.rage_quit_hours, dandelion.tollgate_fee_xdai]}
    df = pd.DataFrame(data=data_table)

    if results_button:
        # Define output pane
        output_pane = pn.Row(pn.Column(t.impact_hours_view,
                                       t.redeemable_plot,
                                       t.cultural_build_tribute_plot),
        pn.Column(dandelion.vote_pass_view, t.funding_pool_view))
        output_pane.save('output.html')
        pn.panel(t.output_scenarios_out_issue().hvplot.table()).save('out_scenarios.html')

        scenarios = codecs.open("out_scenarios.html", 'r')
        charts = codecs.open("output.html", 'r')

        data_charts = { 'html': charts.read(),
        'css': ".box { color: white; background-color: #0f79b9; padding: 10px; font-family: Roboto }",
        'google_fonts': "Roboto" }
        data_scenarios = { 'html': scenarios.read(),
        'css': ".box { color: white; background-color: #0f79b9; padding: 10px; font-family: Roboto }",
        'google_fonts': "Roboto" }

        charts = requests.post(url = HCTI_API_ENDPOINT, data = data_charts, auth=(HCTI_API_USER_ID, HCTI_API_KEY))
        scenarios = requests.post(url = HCTI_API_ENDPOINT, data = data_scenarios, auth=(HCTI_API_USER_ID, HCTI_API_KEY))

        output_data = """

<h1>Output Charts</h1>

![image]({image_charts})

<h1>Output Scenarios</h1>

![image]({image_scenarios})
        """.format(image_charts=charts.json()['url'],
        image_scenarios=scenarios.json()['url'])

        parameters_data = """

<h1>Parameters</h1>

{params_table}
        """.format(params_table=df.to_markdown(index=False, floatfmt=".2f"))

        string_data = """
<h1>Results</h1>

<p>{comments}</p>

- It costs {tollgate_fee_xdai} wxDAI to make a proposal

- Votes will be voted on for {vote_duration_days} days

- TECH token holders will have {rage_quit_hours} Hours to exit the DAO if they don't like the result of a vote (as long as they don't vote yes).

- There can be a maximum of {max_proposals_month} votes per year.

- A proposal that passes can be executed {proposal_execution_hours} hours after it was proposed.

- A CSTK Token holder that has 2000 CSTK can send a max of {max_wxdai_ratio} wxDai to the Hatch

Play with my parameters [here](http://localhost:5006/hatch?ihminr={ihf_minimum_raise}&hs={hour_slope}&maxihr={maximum_impact_hour_rate}&ihtr={ihf_target_raise}&ihmaxr={ifh_maximum_raise}&hor={hatch_oracle_ratio}&hpd={hatch_period_days}&her={hatch_exchange_rate}&ht={hatch_tribute}&sr={support_required}&maq={minimum_accepted_quorum}&vdd={vote_duration_days}&vbh={vote_buffer_hours}&rqh={rage_quit_hours}&tfx={tollgate_fee_xdai}).

        """.format(comments=comments.value,
        tollgate_fee_xdai=dandelion.tollgate_fee_xdai,
        vote_duration_days=dandelion.vote_duration_days,
        rage_quit_hours=dandelion.rage_quit_hours,
        ihf_minimum_raise=t.min_max_raise[0],
        hour_slope=t.impact_hour_slope,
        maximum_impact_hour_rate=t.maximum_impact_hour_rate,
        ihf_target_raise=t.target_raise,
        ifh_maximum_raise=t.min_max_raise[1],
        hatch_oracle_ratio=t.hatch_oracle_ratio,
        hatch_period_days=t.hatch_period_days,
        hatch_exchange_rate=t.hatch_exchange_rate,
        hatch_tribute=t.hatch_tribute,
        support_required=dandelion.support_required_percentage,
        minimum_accepted_quorum=dandelion.minimum_accepted_quorum_percentage,
        vote_buffer_hours=dandelion.vote_buffer_hours,
        max_proposals_month=int(365*24/dandelion.vote_buffer_hours),
        proposal_execution_hours=dandelion.vote_buffer_hours+dandelion.rage_quit_hours,
        max_wxdai_ratio=int(2000*t.hatch_oracle_ratio))

        markdown_panel = pn.pane.Markdown(parameters_data + string_data + output_data)
        body = urllib.parse.quote(markdown_panel.object, safe='')
        url.value = "https://github.com/TECommons/Token-Log-Hatch-Params/issues/new?title=Vote%20for%20My%20Params&labels=TEST%20VOTE&body=" + body

    else:
        string_data=""
    markdown_panel = pn.pane.Markdown(string_data)
    return pn.Row(df.hvplot.table(),markdown_panel)

pn.state.onload(update_params_by_url_query)

# Front-end
tmpl = pn.Template(template=template)
tmpl.add_variable('app_title', 'TEC Hatch Dashboard')
tmpl.add_panel('A', i.impact_hours_accumulation)
tmpl.add_panel('B', t)
tmpl.add_panel('C', t.funding_pool_data_view)
tmpl.add_panel('E', t.payout_view)
tmpl.add_panel('D', pn.Column(t.impact_hours_view, t.redeemable_plot, t.cultural_build_tribute_plot))
tmpl.add_panel('F', t.funding_pool_view)
tmpl.add_panel('V', dandelion)
tmpl.add_panel('W', dandelion.vote_pass_view)
tmpl.add_panel('R', update_result_score)
tmpl.add_panel('CO', comments)
tmpl.add_panel('BU', pn.Column(results_button, share_button, url))
tmpl.servable(title="TEC Hatch Dashboard")

