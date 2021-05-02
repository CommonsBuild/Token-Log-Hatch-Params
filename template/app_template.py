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


from tech.tech import TECH, DandelionVoting, read_impact_hour_data
from template.config_tooltips import tooltips

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

    # TECH
    t = TECH(
        total_impact_hours=impact_hour_data['Assumed IH'].sum(),
        impact_hour_data=impact_hour_data, total_cstk_tokens=1000000,
        config=config_file['tech'])

    # DandelionVoting
    dandelion = DandelionVoting(17e6, config=config_file['dandelion_voting'])

    # Share Button
    comments_tech = pn.widgets.TextAreaInput(
                                        name='What is your Hatch Strategy?',
                                        max_length=1024,
                                        placeholder='Tell us why you configured the Hatch this way')
    comments_dandelion = pn.widgets.TextAreaInput(
                                        name='What is your Dandelion Voting strategy?',
                                        max_length=1024,
                                        placeholder='What intended effects will your Dandelion Voting Parameters have?')
    share_button = pn.widgets.Button(name='Share your results on GitHub!',
                                     button_type='primary')
    url = pn.widgets.TextInput(name='URL', value='')
    share_button.js_on_click(args={'target': url},
                             code='window.open(target.value)')
    results_button = pn.widgets.Button(name='See your results',
                                       button_type='success')

    # Run buttons
    run_dandelion = pn.widgets.Button(name='Run simulation',
                                      button_type='success')
    run_impact_hours = pn.widgets.Button(name='Run simulation',
                                         button_type='success')

    def update_params_by_url_query():
        queries = curdoc().session_context.request.arguments
        queries = {i: j[0] for i, j in queries.items()}
        if queries:
            if 'ihminr' in queries:
                t.min_raise = int(queries['ihminr'])
            if 'ihmaxr' in queries:
                t.max_raise = int(queries['ihmaxr'])
            if 'tgihr' in queries:
                t.impact_hour_rate_at_target_goal = float(queries['tgihr'])
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
    def update_output_scenarios(results_button_on):
        if results_button_on:
            output_scenarios = pn.panel(t.output_scenarios_view()
                                         .hvplot.table())
        else:
            output_scenarios = pn.pane.Markdown('')

        return output_scenarios

    @pn.depends(results_button)
    def update_result_score(results_button_on):
        if results_button_on:
            t.param.trigger('action')  # Update dashboard
            dandelion.param.trigger('action')
            data_table = {
                'Parameters': [
                    "Target raise (wxDai)",
                    "Maximum raise (wxDai)",
                    "Minimum raise (wxDai)",
                    "Impact Hour Rate at Target Goal (wxDai/IH)",
                    "Maximum impact hour rate (wxDai/IH)",
                    "Hatch oracle ratio (wxDai/CSTK)",
                    "Hatch period (days)",
                    "Hatch exchange rate (TECH/wxDai)",
                    "Hatch tribute (%)",
                    "Support required (%)",
                    "Minimum accepted quorum (%)",
                    "Vote duration (days)",
                    "Vote buffer (hours)",
                    "Rage quit (hours)",
                    "Tollgate fee (wxDai)"],
                'Values': [
                    int(t.target_raise),
                    int(t.max_raise),
                    int(t.min_raise),
                    t.impact_hour_rate_at_target_goal,
                    t.maximum_impact_hour_rate,
                    t.hatch_oracle_ratio,
                    t.hatch_period_days,
                    t.hatch_exchange_rate,
                    t.hatch_tribute_percentage,
                    dandelion.support_required_percentage,
                    dandelion.minimum_accepted_quorum_percentage,
                    dandelion.vote_duration_days,
                    dandelion.vote_buffer_hours,
                    dandelion.rage_quit_hours,
                    dandelion.tollgate_fee_xdai]
                }
            df = pd.DataFrame(data=data_table)

            # Define output pane
            output_pane = pn.Row(pn.Column(t.impact_hours_plot,
                                           t.redeemable_plot),
                                 pn.Column(dandelion.vote_pass_view,
                                           t.pie_charts_view))
            output_pane.save('output.html')
            pn.panel(t.output_scenarios_view()
                      .hvplot.table()).save('out_scenarios.html')

            scenarios = codecs.open("out_scenarios.html", 'r')
            charts = codecs.open("output.html", 'r')

            data_charts = {
                'html': charts.read(),
                'css': ".box { color: white; background-color: #0f79b9; padding: 10px; font-family: Roboto }",
                'google_fonts': "Roboto"}
            data_scenarios = {
                'html': scenarios.read(),
                'css': ".box { color: white; background-color: #0f79b9; padding: 10px; font-family: Roboto }",
                'google_fonts': "Roboto"}

            charts = requests.post(url=HCTI_API_ENDPOINT,
                                   data=data_charts,
                                   auth=(HCTI_API_USER_ID, HCTI_API_KEY))
            scenarios = requests.post(url=HCTI_API_ENDPOINT,
                                      data=data_scenarios,
                                      auth=(HCTI_API_USER_ID, HCTI_API_KEY))

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
            """.format(params_table=df.to_markdown(index=False,
                                                   floatfmt=",.2f"))

            results_header = "<h1>Summary</h1>"

            string_comments_tech = """

<h2>Hatch Comments</h2>

<p>{comments}</p>
            """.format(comments=comments_tech.value)

            string_comments_dandelion = """

<h2>Dandelion Voting Comments</h2>

<p>{comments}</p>
            """.format(comments=comments_dandelion.value)

            string_data = """

 <h2>Hatch Details</h2>

- Trusted Seed members can send wxDai to the Hatch for {hatch_period_days} days.

- The target goal will be {ihf_target_raise} wxDai, with a minimum of {ihf_minimum_raise} wxDai necessary for the TEC Hatch DAO to be launched and a cap at {ifh_maximum_raise} wxDai.

- Backers will need to send in {single_tech_mint} wxDai to mint 1 TECH.

- The membership ratio is set at {hatch_oracle_ratio} wxDai/CSTK, so a Trusted Seed member with the minimum CSTK Score of 1125 CSTK can send up to {max_wxdai_ratio}  wxDai to the Hatch.

<h2>TEC Hatch DAO Voting Details</h2>

- Proposals will be voted on for {vote_duration_days} days. They will require at least {support_required}% support, and a minimum of {minimum_accepted_quorum}% of the TECH Tokens will have to vote yes for a proposal to pass.

- TECH token holders will have {rage_quit_hours} hours to exit the DAO if they don't like the result of a vote (as long as they didn't vote yes) before it is executed.

- There will be a minimum of {vote_buffer_hours} hours between proposals so people always have time to exit safely if they voted yes on a previous vote, this means we can have at most {total_votes_per_year} votes per year.

- To prevent griefing attacks, it will cost {tollgate_fee_xdai} wxDai to make a proposal.


Play with my parameters <a href="{url}?ihminr={ihf_minimum_raise}&tgihr={impact_hour_rate_at_target_goal}&maxihr={maximum_impact_hour_rate}&ihtr={ihf_target_raise}&ihmaxr={ifh_maximum_raise}&hor={hatch_oracle_ratio}&hpd={hatch_period_days}&her={hatch_exchange_rate}&ht={hatch_tribute_percentage}&sr={support_required}&maq={minimum_accepted_quorum}&vdd={vote_duration_days}&vbh={vote_buffer_hours}&rqh={rage_quit_hours}&tfx={tollgate_fee_xdai}" target="_blank">here</a>.

To see the value of your individual Impact Hours, click <a href="{url}?ihminr={ihf_minimum_raise}&tgihr={impact_hour_rate_at_target_goal}&maxihr={maximum_impact_hour_rate}&ihtr={ihf_target_raise}&ihmaxr={ifh_maximum_raise}&hor={hatch_oracle_ratio}&hpd={hatch_period_days}&her={hatch_exchange_rate}&ht={hatch_tribute_percentage}&sr={support_required}&maq={minimum_accepted_quorum}&vdd={vote_duration_days}&vbh={vote_buffer_hours}&rqh={rage_quit_hours}&tfx={tollgate_fee_xdai}" target="_blank">here to go to the Hatch Config Dashboard with these parameters</a> and explore the Impact Hour Results table.
            """.format(tollgate_fee_xdai=dandelion.tollgate_fee_xdai,
            vote_duration_days=dandelion.vote_duration_days,
            rage_quit_hours=dandelion.rage_quit_hours,
            ihf_minimum_raise=int(t.min_raise),
            impact_hour_rate_at_target_goal=t.impact_hour_rate_at_target_goal,
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
            max_wxdai_ratio=int(1125*t.hatch_oracle_ratio),
            total_votes_per_year=int(24/dandelion.vote_buffer_hours*365),
            single_tech_mint=float(1/t.hatch_exchange_rate),
            url=config_file['url'])

            markdown_panel = pn.pane.Markdown(parameters_data +
                                              string_comments_tech +
                                              string_comments_dandelion +
                                              results_header +
                                              string_data +
                                              output_data)
            body = urllib.parse.quote(markdown_panel.object, safe='')
            url.value = (config_file['repo'] +
                         "/issues/new?title=Vote%20for%20My%20Params&labels=" +
                         config_file['label'] + "&body=" + body)
            results_button.name = "Update your results"
            markdown_panel = pn.pane.Markdown(results_header + string_data)
            return markdown_panel

    pn.state.onload(update_params_by_url_query)

    def help_icon(href, text):
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
                <a href="{href}" target="_blank">
                    <img class="icon" src="http://cdn.onlinewebfonts.com/svg/img_295214.png" />
                </a>
                <span class="tooltiptext">{text}</span>
            </div>
        </div>
        """.format(href=href, text=text)

    def param_with_tooltip(param, tooltip, height=50):
        return pn.Row(pn.Column(param, sizing_mode="stretch_width"),
                      pn.pane.HTML(help_icon(tooltips[tooltip]['href'],
                                   tooltips[tooltip]['text']),
                                   sizing_mode="fixed",
                                   width=30, height=height, align="end"))

    def run_simulation_dandelion(event):
        dandelion.param.trigger('action')

    def run_simulation_impact_hours(event):
        t.param.trigger('action')

    run_dandelion.on_click(run_simulation_dandelion)
    run_impact_hours.on_click(run_simulation_impact_hours)

    # Front-end
    tmpl = pn.Template(template=template)
    tmpl.add_variable('app_title', config_file['title'])
    tmpl.add_panel('B', pn.Column(
        param_with_tooltip(
            t.param.target_raise,
            tooltip='target_raise'),
        param_with_tooltip(
            t.param.min_raise,
            tooltip='min_raise'),
        param_with_tooltip(
            t.param.max_raise,
            tooltip='max_raise'),
        param_with_tooltip(
            t.param.hatch_tribute_percentage,
            tooltip='hatch_tribute_percentage'),
        param_with_tooltip(
            t.param.maximum_impact_hour_rate,
            tooltip='maximum_impact_hour_rate', height=40),
        param_with_tooltip(
            t.param.impact_hour_rate_at_target_goal,
            tooltip='impact_hour_rate_at_target_goal', height=40),
        param_with_tooltip(
            t.param.hatch_oracle_ratio,
            tooltip='hatch_oracle_ratio'),
        param_with_tooltip(
            t.param.hatch_period_days,
            tooltip='hatch_period_days'),
        param_with_tooltip(
            t.param.hatch_exchange_rate,
            tooltip='hatch_exchange_rate'),
        run_impact_hours
    ))
    tmpl.add_panel('C', t.outputs_overview_view)
    tmpl.add_panel('E', t.payout_view)
    tmpl.add_panel('D', pn.Column(t.redeemable_plot, t.impact_hours_plot))
    tmpl.add_panel('M', t.trigger_unbalanced_parameters)
    tmpl.add_panel('F', t.pie_charts_view)
    tmpl.add_panel('V', pn.Column(
        param_with_tooltip(
            pn.Column(dandelion.param.support_required_percentage),
            tooltip='support_required_percentage', height=40),
        param_with_tooltip(
            dandelion.param.minimum_accepted_quorum_percentage,
            tooltip='minimum_accepted_quorum_percentage', height=40),
        param_with_tooltip(
            dandelion.param.vote_duration_days,
            tooltip='vote_duration_days'),
        param_with_tooltip(
            dandelion.param.vote_buffer_hours,
            tooltip='vote_buffer_hours'),
        param_with_tooltip(
            dandelion.param.rage_quit_hours,
            tooltip='rage_quit_hours'),
        param_with_tooltip(
            dandelion.param.tollgate_fee_xdai,
            tooltip='tollgate_fee_xdai'),
        run_dandelion
    ))
    tmpl.add_panel('W', dandelion.vote_pass_view)
    tmpl.add_panel('G', update_input_output_pane)
    tmpl.add_panel('R', update_result_score)
    tmpl.add_panel('CO', pn.Column(comments_tech, comments_dandelion))
    tmpl.add_panel('BU', pn.Column(results_button, share_button, url))
    tmpl.add_panel('OU', update_output_scenarios)
    tmpl.servable(title=config_file['title'])
