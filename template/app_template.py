from operator import index
from jinja2 import Environment, FileSystemLoader
from bokeh.plotting import curdoc
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
import sys
import os


from tech.tech import read_impact_hour_data, read_cstk_data, TECH
from tech.tech import ImpactHoursData, ImpactHoursFormula, Hatch, DandelionVoting
from template.config_tooltips import tooltips
#import tech.config_bounds as config_bounds
import data

load_dotenv()

env = Environment(loader=FileSystemLoader('.'))
template = env.get_template('template/index.html')

# API settings
HCTI_API_ENDPOINT = "https://hcti.io/v1/image"
HCTI_API_USER_ID = os.environ.get('HCTI_API_USER_ID')
HCTI_API_KEY = os.environ.get('HCTI_API_KEY')


def load_app(config_file):
    pn.config.sizing_mode = 'stretch_both'

    impact_hour_data = read_impact_hour_data()
    # ImpactHoursData
    i = ImpactHoursData()

    # TECH
    t = TECH(total_impact_hours = impact_hour_data['Assumed IH'].sum(),
            impact_hour_data=impact_hour_data, total_cstk_tokens=1000000,
            config=config_file['tech'])


    # ImpactHoursFormula
    #impact_hours_rewards = ImpactHoursFormula(i.total_impact_hours, impact_hour_data_1)
    #impact_rewards_view = pn.Column(impact_hours_rewards.impact_hours_rewards,
    # impact_hours_rewards.ragequit,
    # impact_hours_rewards.impact_hour_minting)

    # Hatch
    cstk_data = read_cstk_data()
    #hatch = Hatch(cstk_data, impact_hours_rewards.target_raise,
    # i.total_impact_hours,
    # impact_hours_rewards.target_impact_hour_rate)

    # DandelionVoting
    dandelion = DandelionVoting(17e6,config=config_file['dandelion_voting'])

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
            if 'ihminr' in queries:
                t.min_raise = int(queries['ihminr'])
            if 'ihmaxr' in queries:
                t.max_raise = int(queries['ihmaxr'])
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
                t.hatch_tribute_percentage = int(queries['ht'])
            if 'sr' in queries:
                dandelion.support_required_percentage = int(queries['sr'])
            if 'maq' in queries:
                dandelion.minimum_accepted_quorum_percentage = int(queries['maq'])
            if 'vdd' in queries:
                dandelion.vote_duration_days = int(queries['vdd'])
            if 'vbh' in queries:
                dandelion.vote_buffer_hours = int(queries['vbh'])
            if 'rqh' in queries:
                dandelion.rage_quit_hours = int(queries['rqh'])
            if 'tfx' in queries:
                dandelion.tollgate_fee_xdai = float(queries['tfx'])

            t.param.trigger('action')  # Update dashboard
            dandelion.param.trigger('action')


    @pn.depends(results_button)
    def update_input_output_pane(results_button_on):
        if results_button_on:
            input_output_pane = pn.pane.GIF('media/inputs_outputs.gif')
        else:
            input_output_pane = pn.pane.Markdown('')
        
        return input_output_pane 
    

    @pn.depends(results_button)
    def update_result_score(results_button_on):
        if results_button_on:
            t.param.trigger('action')  # Update dashboard
            dandelion.param.trigger('action')
            data_table = {'Parameters': ["Target raise (wxDai)", "Maximum raise (wxDai)", "Minimum raise (wxDai)",
            "Impact hour slope (wxDai/IH)", "Maximum impact hour rate (wxDai/IH)",
            "Hatch oracle ratio (wxDai/CSTK)", "Hatch period (days)",
            "Hatch exchange rate (TECH/wxDai)", "Hatch Tribute (%)", "Support required (%)",
            "Minimum accepted quorum (%)", "Vote duration (days)", "Vote buffer (hours)",
            "Rage quit (hours)", "Tollgate fee (wxDai)"],
            'Values': [int(t.target_raise), int(t.max_raise),
            int(t.min_raise), t.impact_hour_slope,
            t.maximum_impact_hour_rate, t.hatch_oracle_ratio,
            t.hatch_period_days, t.hatch_exchange_rate, t.hatch_tribute_percentage,
            dandelion.support_required_percentage, dandelion.minimum_accepted_quorum_percentage, dandelion.vote_duration_days,
            dandelion.vote_buffer_hours, dandelion.rage_quit_hours, dandelion.tollgate_fee_xdai]}
            df = pd.DataFrame(data=data_table)

            # Define output pane
            output_pane = pn.Row(pn.Column(t.impact_hours_view,
                                        t.ragequit_plot),
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
            """.format(params_table=df.to_markdown(index=False, floatfmt=",.2f"))

            string_data = """
<h1>Results</h1>

<p>{comments}</p>

- It costs {tollgate_fee_xdai} wxDai to make a proposal.

- Votes will be voted on for {vote_duration_days} days.

- TECH token holders will have {rage_quit_hours} Hours to exit the DAO if they don't like the result of a vote (as long as they don't vote yes).

- There will be a minimum of {vote_buffer_hours} hours between proposals so people can exit safely in weird edge case scenarios.

- A proposal that passes can be executed {proposal_execution_hours} hours after it was proposed.

- A CSTK Token holder that has 2000 CSTK can send a max of {max_wxdai_ratio} wxDai to the Hatch.

Play with my parameters [here]({url}?ihminr={ihf_minimum_raise}&hs={hour_slope}&maxihr={maximum_impact_hour_rate}&ihtr={ihf_target_raise}&ihmaxr={ifh_maximum_raise}&hor={hatch_oracle_ratio}&hpd={hatch_period_days}&her={hatch_exchange_rate}&ht={hatch_tribute_percentage}&sr={support_required}&maq={minimum_accepted_quorum}&vdd={vote_duration_days}&vbh={vote_buffer_hours}&rqh={rage_quit_hours}&tfx={tollgate_fee_xdai}).

To see the value of your individual Impact Hours, click [here to go to the Hatch Config Dashboard with these parameters]({url}?ihminr={ihf_minimum_raise}&hs={hour_slope}&maxihr={maximum_impact_hour_rate}&ihtr={ihf_target_raise}&ihmaxr={ifh_maximum_raise}&hor={hatch_oracle_ratio}&hpd={hatch_period_days}&her={hatch_exchange_rate}&ht={hatch_tribute_percentage}&sr={support_required}&maq={minimum_accepted_quorum}&vdd={vote_duration_days}&vbh={vote_buffer_hours}&rqh={rage_quit_hours}&tfx={tollgate_fee_xdai}) and explore the Impact Hour Results table.
            """.format(comments=comments.value,
            tollgate_fee_xdai=dandelion.tollgate_fee_xdai,
            vote_duration_days=dandelion.vote_duration_days,
            rage_quit_hours=dandelion.rage_quit_hours,
            ihf_minimum_raise=int(t.min_raise),
            hour_slope=t.impact_hour_slope,
            maximum_impact_hour_rate=t.maximum_impact_hour_rate,
            ihf_target_raise=t.target_raise,
            ifh_maximum_raise=int(t.max_raise),
            hatch_oracle_ratio=t.hatch_oracle_ratio,
            hatch_period_days=t.hatch_period_days,
            hatch_exchange_rate=t.hatch_exchange_rate,
            hatch_tribute_percentage=t.hatch_tribute_percentage,
            support_required=dandelion.support_required_percentage,
            minimum_accepted_quorum=dandelion.minimum_accepted_quorum_percentage,
            vote_buffer_hours=dandelion.vote_buffer_hours,
            proposal_execution_hours=dandelion.vote_buffer_hours+dandelion.rage_quit_hours,
            max_wxdai_ratio=int(2000*t.hatch_oracle_ratio),
            url=config_file['url'])

            markdown_panel = pn.pane.Markdown(parameters_data + string_data + output_data)
            body = urllib.parse.quote(markdown_panel.object, safe='')
            url.value = config_file['repo'] + "/issues/new?title=Vote%20for%20My%20Params&labels=" + config_file['label'] + "&body=" + body
            results_button.name = "Update your results"
            markdown_panel = pn.pane.Markdown(string_data)
            return pn.Row(df.hvplot.table(),markdown_panel)


    pn.state.onload(update_params_by_url_query)

    def help_icon(text):
        return """
        <style>
        .tooltip {{
            position: relative;
            display: inline-block;
            align-self: flex-end;
        }}

        .tooltip .tooltiptext {{
            visibility: hidden;
            width: 200px;
            background-color: #555;
            color: #fff;
            text-align: center;
            border-radius: 6px;
            padding: 10px;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            margin-left: -110px;
            opacity: 0;
            transition: opacity 0.3s;
        }}

        .tooltip .tooltiptext::after {{
            content: "";
            position: absolute;
            top: 100%;
            left: 50%;
            margin-left: -5px;
            border-style: solid;
            border-color: #555 transparent transparent transparent;
        }}

        .tooltip:hover .tooltiptext {{
            visibility: visible;
            opacity: 1;
        }}

        .icon {{
            width: 24px;
            height: 24px;
        }}

        .flex {{
            height: 100%;
            display: flex;
            justify-content: center;
        }}
        </style>
        <div class="flex">
            <div class="tooltip">
                <a href="https://forum.tecommons.org/t/tec-test-hatch-implementation-specification/226" target="_blank">
                    <img class="icon" src="http://cdn.onlinewebfonts.com/svg/img_295214.png" />
                </a>
                <span class="tooltiptext">{text}</span>
            </div>
        </div>
        """.format(text=text)

    def param_with_tooltip(param, tooltip, height=50):
        return pn.Row(pn.Column(param, sizing_mode="stretch_width"), pn.pane.HTML(help_icon(tooltips[tooltip]), sizing_mode="fixed", width=30, height=height, align="end"))

    # Front-end
    tmpl = pn.Template(template=template)
    tmpl.add_variable('app_title', config_file['title'])
    tmpl.add_panel('B', pn.Column(
        param_with_tooltip(t.param.target_raise, tooltip='target_raise'), 
        param_with_tooltip(t.param.min_raise, tooltip='min_raise'),
        param_with_tooltip(t.param.max_raise, tooltip='max_raise'),
        param_with_tooltip(t.param.hatch_tribute_percentage, tooltip='hatch_tribute_percentage'),
        param_with_tooltip(t.param.maximum_impact_hour_rate, tooltip='maximum_impact_hour_rate', height=40),
        param_with_tooltip(t.param.impact_hour_slope, tooltip='impact_hour_slope', height=40),
        param_with_tooltip(t.param.hatch_oracle_ratio, tooltip='hatch_oracle_ratio'),
        param_with_tooltip(t.param.hatch_period_days, tooltip='hatch_period_days'),
        param_with_tooltip(t.param.hatch_exchange_rate, tooltip='hatch_exchange_rate'),
        t.param.action,
        #t.param.target_impact_hour_rate,
        #t.param.target_ragequit,
        #t.param.target_impact_hour_minting
    ))
    tmpl.add_panel('C', t.funding_pool_data_view)
    tmpl.add_panel('E', t.payout_view)
    tmpl.add_panel('D', pn.Column(t.impact_hours_view, t.redeemable_plot))
    tmpl.add_panel('M', t.trigger_unbalanced_parameters)
    tmpl.add_panel('F', t.funding_pool_view)
    tmpl.add_panel('V', pn.Column(
        param_with_tooltip(pn.Column(dandelion.param.support_required_percentage), tooltip='support_required_percentage', height=40), 
        param_with_tooltip(dandelion.param.minimum_accepted_quorum_percentage, tooltip='minimum_accepted_quorum_percentage', height=40),
        param_with_tooltip(dandelion.param.vote_duration_days, tooltip='vote_duration_days'),
        param_with_tooltip(dandelion.param.vote_buffer_hours, tooltip='vote_buffer_hours'),
        param_with_tooltip(dandelion.param.rage_quit_hours, tooltip='rage_quit_hours'),
        param_with_tooltip(dandelion.param.tollgate_fee_xdai, tooltip='tollgate_fee_xdai'),
        dandelion.param.action
    ))
    tmpl.add_panel('W', dandelion.vote_pass_view)
    tmpl.add_panel('G', update_input_output_pane)
    tmpl.add_panel('R', update_result_score)
    tmpl.add_panel('CO', comments)
    tmpl.add_panel('BU', pn.Column(results_button, share_button, url))
    tmpl.servable(title=config_file['title'])
