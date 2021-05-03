import param
import panel as pn
import pandas as pd
import hvplot.pandas
import holoviews as hv
import numpy as np
import os
from tech.utils import pie_chart
pn.extension()


def read_impact_hour_data():
    impact_hour_data = pd.read_csv(os.path.join("data", "praise_quantification.csv"))
    return impact_hour_data


class TECH(param.Parameterized):
    target_raise = param.Number(500, label="Target Goal (wxDai)")
    min_raise = param.Integer(1, label="Minimum Goal (wxDai)")
    max_raise = param.Integer(1, label="Maximum Goal (wxDai)")
    hatch_oracle_ratio = param.Number(0.005, label="Membership Ratio (wxDai/CSTK)")
    hatch_period_days = param.Integer(15, label="Hatch Period (days)")
    hatch_exchange_rate = param.Number(10000, label="Hatch Minting Rate (TECH/wxDai)")
    hatch_tribute_percentage = param.Number(5, step=1, label="Hatch Tribute (%)")
    maximum_impact_hour_rate = param.Number(0.01, bounds=(0, 1), label="Impact Hour Rate at Infinity (wxDai/IH)")
    impact_hour_rate_at_target_goal = param.Number(10, step=1, label="Impact Hour Rate at Target Goal (wxDai/IH)")
    action = param.Action(lambda x: x.param.trigger('action'), label='Run Simulation')

    def __init__(self, total_impact_hours, impact_hour_data, total_cstk_tokens,
                 config, **params):
        super(TECH, self).__init__(**params, name="Hatch")
        self.config_bounds = config
        self.total_impact_hours = total_impact_hours
        self.impact_hour_data = impact_hour_data
        self.total_cstk_tokens = total_cstk_tokens
        self.output_scenario_raise = config["output_scenario_raise"]

        # Change the parameter bound according the config_bound argument
        self.min_raise = config['min_max_raise']['value'][0]
        self.param.min_raise.step = config['min_max_raise']['step']
        self.max_raise = config['min_max_raise']['value'][1]
        self.param.max_raise.step = config['min_max_raise']['step']

        self.target_raise = config['target_raise']['value']
        self.param.target_raise.step = config['target_raise']['step']

        self.impact_hour_rate_at_target_goal = config['impact_hour_rate_at_target_goal']['value']

        self.param.maximum_impact_hour_rate.bounds = config['maximum_impact_hour_rate']['bounds']
        self.param.maximum_impact_hour_rate.step = config['maximum_impact_hour_rate']['step']
        self.maximum_impact_hour_rate = config['maximum_impact_hour_rate']['value']

        self.hatch_oracle_ratio = config['hatch_oracle_ratio']['value']
        self.hatch_period_days = config['hatch_period_days']['value']
        self.hatch_exchange_rate = config['hatch_exchange_rate']['value']
        self.hatch_tribute_percentage = config['hatch_tribute_percentage']['value']

    def bounds_impact_hour_rate_at_target_goal(self):
        if self.impact_hour_rate_at_target_goal <= 0:
            self.impact_hour_rate_at_target_goal = 1
        elif self.impact_hour_rate_at_target_goal > self.maximum_impact_hour_rate:
            self.impact_hour_rate_at_target_goal = self.maximum_impact_hour_rate

        return self.impact_hour_rate_at_target_goal

    def get_impact_hour_slope(self):
        self.bounds_impact_hour_rate_at_target_goal()
        impact_hour_slope = (self.maximum_impact_hour_rate / self.impact_hour_rate_at_target_goal - 1) * (self.target_raise / self.total_impact_hours)
        return impact_hour_slope

    # Utils
    def impact_hours_formula(self, minimum_raise=0, maximum_raise=0,
                             raise_scenarios=None, single_value=None):
        """
        This function returns the impact hour rate based on the raised amount
        (wxDai), the maximum impact hour rate, the impact hour slope and the
        total impact hours. The function accepts 3 different types of inputs.
        The first option is to add just the arguments 'minimum_raise', and
        'maximum_raise', and the function will return a dataframe with 500 or
        100 points (depending on the size of the range) between the
        'minimum_raise' and 'maximum_raise' using np.linspace() with the total
        raised and its respective impact hour rate. The second option is to
        input just the 'raise_scenarios' argument as a list of raise amounts,
        where it will be returned a dataframe with only the raised amounts added
        in the argument list and their respective impact hour rates. The third
        option is to input only the argument 'single_value', where only one
        value of raise amount is inputted and the function returns its respective
        impact hour rate.
        """
        R = self.maximum_impact_hour_rate
        m = self.get_impact_hour_slope()
        H = self.total_impact_hours

        if single_value is None:
            xlim = self.config_bounds['min_max_raise']['xlim'][1]
            if raise_scenarios is None:
                if maximum_raise > xlim:
                    x_1 = np.linspace(minimum_raise, xlim, num=500)
                    x_2 = np.linspace(xlim, maximum_raise, num=100)
                    x = np.concatenate([x_1, x_2])
                else:
                    x = np.linspace(minimum_raise, maximum_raise, num=500)
            else:
                x = raise_scenarios

            y = [R * (x / (x + m * H)) for x in x]
            df = pd.DataFrame([x, y]).T
            df.columns = ['Total wxDai Goal', 'Impact Hour Rate']
            return df

        else:
            impact_hour = R * (single_value / (single_value + m * H))
            return impact_hour

    def get_impact_hour_rate(self, raise_amount):
        """
        This is a simple wrapper for the impact_hours_formula using only one
        raise_amount argument. It returns the impact hour rate given a raise
        amount.
        """
        rate = self.impact_hours_formula(single_value=raise_amount)
        return rate

    def get_rage_quit_percentage(self, raise_amount):
        """
        This function returns the rage quit percentage based on a raise amount.
        The rage quit percentage defines how much % of wxDai a backer can get
        from its initial investment in the commons.
        """
        impact_hour_rate = self.get_impact_hour_rate(raise_amount)
        hatch_tribute = self.hatch_tribute_percentage / 100
        redeemable_reserve = raise_amount * (1 - hatch_tribute)
        rage_quit_percentage = (redeemable_reserve / (impact_hour_rate * self.total_impact_hours + raise_amount))

        return rage_quit_percentage

    def bounds_target_raise(self):
        """
        This is a simple utility function for scenarios where the target raise
        is not between the minimum raise and the maximum raise. If the target
        goal is higher than the maximum goal, the function will set the target
        goal to the same value of the maximum goal. The same will happen if the
        target goal is smaller than the minimum goal, in this case the target
        goal will be set to the same value of the minimum goal.
        """
        if self.target_raise > self.max_raise:
            self.target_raise = self.max_raise
        elif self.target_raise < self.min_raise:
            self.target_raise = self.min_raise

        return self.target_raise

    def get_overview_data(self):
            """
            This function return key metrics for the 3 goal scenarios (Minimum goal,
            Target goal, Maximum goal): it calculates the impact_hour_rate for each
            scenario based on the impact hours formula; the backers ragequit
            percentage based on the total raised and the total TECH tokens minted
            for the builders; the total supply held by builders, simply the
            proportion of total builders TECH tokens to the total TECH tokens; the
            total TECH minted, that is the total TECH held by the builders and
            backers; the non-redeemable, that is the hatch tribute in wxDai; the
            backers amount, that is the total wxDai the backers can redeem for their
            TECH tokens; the builders' amount, that is the wxDai the builders can
            redeem for their TECH tokens.
            """
            self.bounds_target_raise()
            hatch_tribute = self.hatch_tribute_percentage / 100
            scenarios = {
                'min_raise': int(self.min_raise),
                'target_raise': self.target_raise,
                'max_raise': int(self.max_raise)
            }

            funding_pool_data = {}
            for scenario, raise_amount in scenarios.items():
                impact_hour_rate = self.get_impact_hour_rate(raise_amount)
                rage_quit_percentage = 100 * self.get_rage_quit_percentage(raise_amount)
                total_tech_backers = raise_amount * self.hatch_exchange_rate
                total_tech_builders = self.total_impact_hours * self.get_impact_hour_rate(raise_amount) * self.hatch_exchange_rate
                total_tech_minted = total_tech_backers + total_tech_builders
                tech_builders_percentage = 100 * total_tech_builders / total_tech_minted
                redeemable_reserve = raise_amount * (1 - hatch_tribute)
                tech_token_ratio = (redeemable_reserve / (impact_hour_rate * self.total_impact_hours + raise_amount))
                non_redeemable = raise_amount * hatch_tribute

                backers_amount = (total_tech_backers/self.hatch_exchange_rate) * (tech_token_ratio)
                builders_amount = (total_tech_builders/self.hatch_exchange_rate) * (tech_token_ratio)
                funding_pool_data[scenario] = {
                    "Impact Hour Rate (wxDai/hour)": impact_hour_rate,
                    "Backer's RageQuit (%)": rage_quit_percentage,
                    "Total Supply held by Builders (%)": tech_builders_percentage,
                    "Total TECH Minted (TECH)": total_tech_minted,
                    "non_redeemable": non_redeemable,
                    "backers_amount": backers_amount,
                    "builders_amount": builders_amount,
                }
            return pd.DataFrame(funding_pool_data).T

    # Views
    @param.depends('action')
    def impact_hours_plot(self):
        self.bounds_target_raise()
        # Limits the target raise bounds when ploting the charts
        df = self.impact_hours_formula(self.config_bounds['min_max_raise']['bounds'][0], self.max_raise)
        df_fill_minimum = df[df['Total wxDai Goal'] <= self.min_raise]

        try:
            target_impact_hour_rate = df[df['Total wxDai Goal'] >= self.target_raise].iloc[0]['Impact Hour Rate']
        except:
            target_impact_hour_rate = 0

        impact_hours_plot = df.hvplot.area(title='Impact Hour Rate',
                                           x='Total wxDai Goal',
                                           xformatter='%.0f',
                                           yformatter='%.0f',
                                           hover=True,
                                           xlim=self.config_bounds['min_max_raise']['xlim'],
                                           label='Hatch happens ✅'
                                           ).opts(axiswise=True)
        minimum_raise_plot = df_fill_minimum.hvplot.area(x='Total wxDai Goal',
                                                         xformatter='%.0f',
                                                         yformatter='%.0f',
                                                         color='red',
                                                         xlim=self.config_bounds['min_max_raise']['xlim'],
                                                         label='Hatch fails 🚫'
                                                         ).opts(axiswise=True)

        return (impact_hours_plot *
                minimum_raise_plot *
                hv.VLine(self.target_raise).opts(color='#E31212') *
                hv.HLine(target_impact_hour_rate).opts(color='#E31212')
                ).opts(legend_position='top_left')

    @param.depends('action')
    def payout_view(self):
        self.bounds_target_raise()
        self.impact_hour_data['Equiv To Mint at Target Goal (wxDai)'] = (self.impact_hour_data['Assumed IH'] * self.get_impact_hour_rate(self.target_raise)).round(2)
        self.impact_hour_data['Minted at Target Goal (TECH)'] = (self.impact_hour_data['Equiv To Mint at Target Goal (wxDai)'] * self.hatch_exchange_rate).round(2)
        self.impact_hour_data['RageQuit Value at Target Goal (wxDai)'] = (self.impact_hour_data['Equiv To Mint at Target Goal (wxDai)'] * self.get_rage_quit_percentage(self.target_raise)).round(2)
        self.impact_hour_data = self.impact_hour_data[['Handle', 'Assumed IH',
                                                       'Minted at Target Goal (TECH)',
                                                       'Equiv To Mint at Target Goal (wxDai)',
                                                       'RageQuit Value at Target Goal (wxDai)']]

        self.impact_hour_data = self.impact_hour_data.round(2)
        return self.impact_hour_data.hvplot.table(title='Predicted Individual Impact Hour Results', width=1350)

    @param.depends('action')
    def redeemable_plot(self):
        # Limits the target raise bounds when ploting the charts
        self.bounds_target_raise()
        df_hatch_params = self.impact_hours_formula(self.config_bounds['min_max_raise']['bounds'][0], self.max_raise)
        df_hatch_params['Cultural Build Tribute'] = (self.total_impact_hours * df_hatch_params['Impact Hour Rate'])/df_hatch_params['Total wxDai Goal']
        df_hatch_params['Hatch tribute'] = self.hatch_tribute_percentage / 100
        df_hatch_params["Backer's RageQuit (%)"] = df_hatch_params['Total wxDai Goal'].apply(self.get_rage_quit_percentage).round(4)
        df_hatch_params['label'] = ""

        # Add label case there is already a row with raise value
        df_hatch_params.loc[df_hatch_params['Total wxDai Goal'] == int(self.min_raise), 'label'] = "Min Goal"
        df_hatch_params.loc[df_hatch_params['Total wxDai Goal'] == self.target_raise, 'label'] = "Target Goal"
        df_hatch_params.loc[df_hatch_params['Total wxDai Goal'] == int(self.max_raise), 'label'] = "Max Goal"
        df_hatch_params.loc[df_hatch_params['Total wxDai Goal'] < int(self.min_raise), ['Impact Hour Rate',
                                                                                          'Cultural Build Tribute',
                                                                                          'Hatch tribute']] = 0
        df_hatch_params.loc[df_hatch_params['Total wxDai Goal'] < int(self.min_raise), "Backer's RageQuit (%)"] = 1
        df_hatch_params.loc[df_hatch_params['Total wxDai Goal'] > int(self.max_raise), ['Impact Hour Rate',
                                                                                          'Cultural Build Tribute',
                                                                                          'Hatch tribute',
                                                                                          "Backer's RageQuit (%)"]] = np.nan

        df_hatch_params_to_plot = df_hatch_params
        # Drop NaN rows
        df_hatch_params_to_plot = df_hatch_params_to_plot.dropna()
        with pd.option_context('mode.chained_assignment', None):
            df_hatch_params_to_plot["Backer's RageQuit (%)"] = df_hatch_params_to_plot["Backer's RageQuit (%)"].mul(100)
        redeemable_plot = df_hatch_params_to_plot.hvplot.area(title="Backer's RageQuit (%)",
                                                              x='Total wxDai Goal',
                                                              y="Backer's RageQuit (%)",
                                                              xformatter='%.0f',
                                                              yformatter='%.0f',
                                                              hover=True,
                                                              ylim=(0, 100),
                                                              xlim=self.config_bounds['min_max_raise']['xlim']
                                                              ).opts(axiswise=True)
        try:
            redeemable_target = df_hatch_params_to_plot[df_hatch_params_to_plot['Total wxDai Goal'] >= self.target_raise].iloc[0]["Backer's RageQuit (%)"]
        except:
            redeemable_target = 0

        return redeemable_plot * hv.VLine(self.target_raise).opts(color='#E31212') * hv.HLine(redeemable_target).opts(color='#E31212')

    def output_scenarios_view(self):
            hatch_tribute = self.hatch_tribute_percentage / 100
            R = self.maximum_impact_hour_rate
            m = self.get_impact_hour_slope()
            H = self.total_impact_hours

            df_hatch_params = self.impact_hours_formula(raise_scenarios=self.output_scenario_raise)
            df_hatch_params['Impact Hour Rate'] = df_hatch_params['Impact Hour Rate'].round(2)
            df_hatch_params['Cultural Build Tribute'] = (H * df_hatch_params['Impact Hour Rate'])/df_hatch_params['Total wxDai Goal']
            df_hatch_params['Hatch tribute'] = df_hatch_params['Total wxDai Goal'].mul(hatch_tribute)
            df_hatch_params['Redeemable'] = df_hatch_params['Total wxDai Goal'].apply(self.get_rage_quit_percentage).round(4)
            df_hatch_params['Total TECH builders'] = df_hatch_params['Impact Hour Rate'] * self.total_impact_hours * self.hatch_exchange_rate
            df_hatch_params['Total TECH backers'] = df_hatch_params['Total wxDai Goal'] * self.hatch_exchange_rate
            df_hatch_params['Total Supply held by Builders (%)'] = (100 * df_hatch_params['Total TECH builders'] / (df_hatch_params['Total TECH builders'] + df_hatch_params['Total TECH backers'])).round(2)

            df_hatch_params['label'] = ""

            minimum_raise = int(self.min_raise)
            target_raise = int(self.target_raise)
            maximum_raise = int(self.max_raise)

            def add_goal_to_table(df, amount_raised, label):
                # Add label case there is already a row with amount_raised value
                df.loc[df['Total wxDai Goal'] == amount_raised, 'label'] = label

                # Add a new row with amount_raised value case there is no row with its value
                if label not in df['label']:
                    impact_hour_rate = R * (amount_raised / (amount_raised + m * H))
                    cultural_build_tribute = (H * impact_hour_rate)/amount_raised
                    total_tech_backers = amount_raised * self.hatch_exchange_rate
                    total_tech_builders = self.total_impact_hours * self.get_impact_hour_rate(amount_raised) * self.hatch_exchange_rate
                    total_tech_minted = total_tech_backers + total_tech_builders
                    tech_builders_percentage = round(100 * total_tech_builders / total_tech_minted, 2)

                    df = df.append({
                        'Total wxDai Goal': amount_raised,
                        'Impact Hour Rate': round(impact_hour_rate, 2),
                        'Hatch tribute': amount_raised * hatch_tribute,
                        'Redeemable': round(self.get_rage_quit_percentage(amount_raised), 2),
                        'Total Supply held by Builders (%)': tech_builders_percentage,
                        'label': label},
                                   ignore_index=True)
                    df = df.sort_values(['Total wxDai Goal'])

                df_min_raise = df.query("label == '{}'".format(label))
                if len(df_min_raise) > 1:
                    df = df.drop(df_min_raise.first_valid_index())

                return df

            df_hatch_params = add_goal_to_table(df=df_hatch_params, amount_raised=minimum_raise, label='Minimum Goal')
            df_hatch_params = add_goal_to_table(df=df_hatch_params, amount_raised=target_raise, label='Target Goal')
            df_hatch_params = add_goal_to_table(df=df_hatch_params, amount_raised=maximum_raise, label='Maximum Goal')

            # Send to zero the IH rate, cultural build tribute and hatch tribue of
            # amount raises smaller than the min target
            df_hatch_params.loc[df_hatch_params['Total wxDai Goal'] < minimum_raise, ['Impact Hour Rate',
                                                                                        'Hatch tribute',
                                                                                        'Total Supply held by Builders (%)']] = 0
            df_hatch_params.loc[df_hatch_params['Total wxDai Goal'] < minimum_raise, 'Redeemable'] = 1
            df_hatch_params['Redeemable'] = df_hatch_params['Redeemable'].mul(100).round(2)
            df_hatch_params.loc[df_hatch_params['Total wxDai Goal'] > maximum_raise, ['Impact Hour Rate',
                                                                                        'Hatch tribute',
                                                                                        'Redeemable',
                                                                                        'Total Supply held by Builders (%)']] = np.nan

            # Format final table columns
            df_hatch_params = df_hatch_params.rename(columns={'Total wxDai Goal': 'Total wxDai Goal (wxDai)',
                                                              'Impact Hour Rate': 'Impact Hour Rate (wxDai)',
                                                              'Hatch tribute': 'Non-redeemable (wxDai)',
                                                              'Redeemable': "Backer's RageQuit (%)",
                                                              'label': 'Label'})
            df_hatch_params = df_hatch_params.filter(items=['Total wxDai Goal (wxDai)',
                                                            'Impact Hour Rate (wxDai)',
                                                            'Total Supply held by Builders (%)',
                                                            'Non-redeemable (wxDai)',
                                                            "Backer's RageQuit (%)",
                                                            'Label'])
            df_hatch_params = df_hatch_params.round(2)
            df_hatch_params = df_hatch_params.fillna("Beyond Max Goal")

            return df_hatch_params

    @param.depends('action')
    def pie_charts_view(self):
        self.bounds_target_raise()
        funding_pools = self.get_overview_data()
        funding_pools = funding_pools.filter(items=['non_redeemable', 'builders_amount', 'backers_amount'])
        funding_pools = funding_pools.rename(columns={'builders_amount': 'Builders can RageQuit',
                                                      'non_redeemable': 'Non-redeemable',
                                                      'backers_amount': 'Backers can RageQuit'})

        # Plot pie charts
        colors = ['#0b0a15', '#0F2EEE', '#DEFB48']
        chart_data = funding_pools
        p1 = pie_chart(data=pd.Series(chart_data.loc['min_raise', :]),
                       radius=0.65,
                       title="Min Goal", toolbar_location=None, plot_width=300,
                       show_legend=False, colors=colors)
        p2 = pie_chart(data=pd.Series(chart_data.loc['target_raise', :]),
                       radius=0.65,
                       title="Target Goal", toolbar_location=None, plot_width=300,
                       show_legend=False, colors=colors)
        p3 = pie_chart(data=pd.Series(chart_data.loc['max_raise', :]),
                       radius=0.65,
                       title="Max Goal", colors=colors)

        return pn.Row(p1, p2, p3)

    @param.depends('action')
    def trigger_unbalanced_parameters(self):
        self.bounds_target_raise()
        target_impact_hour_rate = self.get_impact_hour_rate(self.target_raise)
        if target_impact_hour_rate < 15:
            return pn.pane.JPG('https://i.imgflip.com/54l0iv.jpg')
        else:
            return pn.pane.Markdown('')

    @param.depends('action')
    def outputs_overview_view(self):
        funding_pools = self.get_overview_data()
        funding_pools = funding_pools.filter(items=["Impact Hour Rate (wxDai/hour)",
                                                    "Backer's RageQuit (%)",
                                                    "Total Supply held by Builders (%)",
                                                    "Total TECH Minted (TECH)"])
        funding_pools = funding_pools.round(2)
        funding_pools = funding_pools.T.reset_index()
        funding_pools = funding_pools.rename(columns={'index': 'Output',
                                                      'min_raise': 'Min Goal',
                                                      'target_raise': 'Target Goal',
                                                      'max_raise': 'Max Goal'})

        return funding_pools.hvplot.table(title='Outputs Overview', width=880)


class DandelionVoting(param.Parameterized):
    support_required_percentage = param.Number(60, bounds=(50, 90), step=1, label="Support Required (%)")
    minimum_accepted_quorum_percentage = param.Number(2, bounds=(1, 100), step=1, label="Minimum Quorum (%)")
    vote_duration_days = param.Integer(3, label="Vote Duration (days)")
    vote_buffer_hours = param.Integer(8, label="Vote Proposal Buffer (hours)")
    rage_quit_hours = param.Integer(24, label="Ragequit (hours)")
    tollgate_fee_xdai = param.Number(3, label="Tollgate Fee (wxDai)")
    action = param.Action(lambda x: x.param.trigger('action'), label='Run Simulation')

    def __init__(self, total_tokens, config, **params):
        super(DandelionVoting, self).__init__(**params, name="TEC Hatch DAO")
        self.total_tokens = total_tokens

        # Change the parameter bound according the config_bound argument
        self.param.support_required_percentage.bounds = config['support_required_percentage']['bounds']
        self.param.support_required_percentage.step = config['support_required_percentage']['step']
        self.support_required_percentage = config['support_required_percentage']['value']

        self.param.minimum_accepted_quorum_percentage.bounds = config['minimum_accepted_quorum_percentage']['bounds']
        self.param.minimum_accepted_quorum_percentage.step = config['minimum_accepted_quorum_percentage']['step']
        self.minimum_accepted_quorum_percentage = config['minimum_accepted_quorum_percentage']['value']

        self.vote_duration_days = config['vote_duration_days']['value']
        self.vote_buffer_hours = config['vote_buffer_hours']['value']
        self.rage_quit_hours = config['rage_quit_hours']['value']
        self.tollgate_fee_xdai = config['tollgate_fee_xdai']['value']

    def support_required(self):
        return self.support_required_percentage/100

    def minimum_accepted_quorum(self):
        return self.minimum_accepted_quorum_percentage/100

    @param.depends('action')
    def vote_pass_view(self):
        x = np.linspace(0, 100, num=100)
        y = [a*self.support_required() for a in x]
        df = pd.DataFrame(zip(x, y))
        y_fill = x.tolist()
        df_fill = pd.DataFrame(zip(x, y_fill))
        y_fill_quorum = [a for i, a in enumerate(x) if i < self.minimum_accepted_quorum()*len(x)]
        df_fill_q = pd.DataFrame(zip(x, y_fill_quorum))
        total_votes_plot = df_fill.hvplot.area(
                title="Proposal Acceptance Criteria",
                x='0', y='1', xformatter='%.0f', yformatter='%.0f', color='green',
                xlabel='Total Token Votes (%)', ylabel='Yes Token Votes (%)', label='Proposal Passes ✅')
        support_required_plot = df.hvplot.area(x='0', y='1', xformatter='%.0f', yformatter='%.0f', color='red', label='Proposal Fails 🚫')
        quorum_accepted_plot = df_fill_q.hvplot.area(x='0', y='1', xformatter='%.0f', yformatter='%.0f', color='yellow', label='Minimum quorum')
        return (total_votes_plot * support_required_plot * quorum_accepted_plot).opts(legend_position='top_left')
