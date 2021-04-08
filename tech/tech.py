doc = """
This jupyter notebook is authored by ygg_anderson for the Token Engineering Commons. See appropriate licensing. 🐧 🐧 🐧
"""

import param
import panel as pn
import pandas as pd
import hvplot.pandas
import holoviews as hv
import numpy as np
from scipy.stats.mstats import gmean
import os
pn.extension()

from tech.utils import pie_chart


APP_PATH = './'

sheets = [
    "Total Impact Hours so far",
    "IH Predictions",
    "#8 Jan 1",
    "#7 Dec 18",
    "#6 Dec 4",
    "#5 Nov 20",
    "#4 Nov 6",
    "#3 Oct 23",
    "#2 Oct 9",
    "#1 Sept 24",
    "#0 Sept 7 (historic)",
] + [f"#{i} IH Results" for i in range(9)]
sheets = {i:sheet for i, sheet in enumerate(sheets)}

def read_excel(sheet_name="Total Impact Hours so far", header=1, index_col=0, usecols=None) -> pd.DataFrame:
    data = pd.read_excel(
            os.path.join("data", "TEC Praise Quantification.xlsx"),
            sheet_name=sheet_name,
            engine='openpyxl',
            header=header,
            index_col=index_col,
            usecols=usecols
    ).reset_index().dropna(how='any')
    return data


def read_impact_hour_data():
    impact_hour_data_1 = read_excel()
    impact_hour_data_2 = read_excel(sheet_name="IH Predictions", header=0, index_col=0, usecols='A:I').drop(index=19)
    return (impact_hour_data_1, impact_hour_data_2)


def read_cstk_data():
    # Load CSTK data
    cstk_data = pd.read_csv('data/CSTK_DATA.csv', header=None).reset_index().head(100)
    cstk_data.columns = ['CSTK Token Holders', 'CSTK Tokens']
    cstk_data['CSTK Tokens Capped'] = cstk_data['CSTK Tokens'].apply(lambda x: min(x, cstk_data['CSTK Tokens'].sum()/10))
    return cstk_data


class TECH(param.Parameterized):
    #min_max_raise = param.Range((1, 1000), bounds=(1,1000), label="Minimum/Maximum Goal (wxDai)")
    target_raise = param.Number(500, label="Target Goal (wxDai)")
    min_raise = param.Integer(1, label="Minimum Goal (wxDai)")
    max_raise = param.Integer(1, label="Maximum Goal (wxDai)")
    hatch_oracle_ratio = param.Number(0.005, label="Membership Ratio (wxDai/CSTK)")
    hatch_period_days = param.Integer(15, label="Hatch Period (days)")
    hatch_exchange_rate = param.Number(10000, label="Hatch Minting Rate (TECH/wxDai)")
    hatch_tribute_percentage = param.Number(5, step=1, label="Hatch Tribute (%)")
    maximum_impact_hour_rate = param.Number(0.01, bounds=(0, 1), label="Maximum Impact Hour Rate (wxDai/IH)")
    impact_hour_slope = param.Number(0.012, bounds=(0,1), step=0.001, label="Impact Hour Slope (wxDai/IH)")
    target_impact_hour_rate = param.Parameter(0, label="Target Impact Hour Rate (wxDai/hour)", constant=True)
    target_redeemable = param.Parameter(0, label="Target Redeemable (%)", constant=True)
    target_impact_hour_rate = param.Parameter(0, label="Target Impact Hour Rate (wxDai/hour)", constant=True)
    target_cultural_build_tribute = param.Parameter(0, label="Target Cultural Build Tribute (%)", constant=True)
    action = param.Action(lambda x: x.param.trigger('action'), label='Run simulation')

    def __init__(self, total_impact_hours, impact_hour_data, total_cstk_tokens,
                 config, **params):
        super(TECH, self).__init__(**params, name="Hatch")
        self.config_bounds = config
        self.total_impact_hours = total_impact_hours
        self.impact_hour_data = impact_hour_data
        self.total_cstk_tokens = total_cstk_tokens
        self.output_scenario_raise = config["output_scenario_raise"]

        # Change the parameter bound according the config_bound argument
        #self.param.min_max_raise.bounds = config['min_max_raise']['bounds']
        #self.min_max_raise = config['min_max_raise']['value']
        self.min_raise = config['min_max_raise']['value'][0]
        self.param.min_raise.step = config['min_max_raise']['step']
        self.max_raise = config['min_max_raise']['value'][1]
        self.param.max_raise.step = config['min_max_raise']['step']

        #self.param.target_raise.bounds = config['target_raise']['bounds']
        self.target_raise = config['target_raise']['value']
        self.param.target_raise.step = config['target_raise']['step']

        self.param.impact_hour_slope.bounds = config['impact_hour_slope']['bounds']
        self.param.impact_hour_slope.step = config['impact_hour_slope']['step']
        self.impact_hour_slope = config['impact_hour_slope']['value']

        self.param.maximum_impact_hour_rate.bounds = config['maximum_impact_hour_rate']['bounds']
        self.param.maximum_impact_hour_rate.step = config['maximum_impact_hour_rate']['step']
        self.maximum_impact_hour_rate = config['maximum_impact_hour_rate']['value']

        #self.param.hatch_oracle_ratio.bounds = config['hatch_oracle_ratio']['bounds']
        #self.param.hatch_oracle_ratio.step = config['hatch_oracle_ratio']['step']
        self.hatch_oracle_ratio = config['hatch_oracle_ratio']['value']

        #self.param.hatch_period_days.bounds = config['hatch_period_days']['bounds']
        #self.param.hatch_period_days.step = config['hatch_period_days']['step']
        self.hatch_period_days = config['hatch_period_days']['value']

        #self.param.hatch_exchange_rate.bounds = config['hatch_exchange_rate']['bounds']
        #self.param.hatch_exchange_rate.step = config['hatch_exchange_rate']['step']
        self.hatch_exchange_rate = config['hatch_exchange_rate']['value']

        #self.param.hatch_tribute_percentage.bounds = config['hatch_tribute_percentage']['bounds']
        #self.param.hatch_tribute_percentage.step = config['hatch_tribute_percentage']['step']
        self.hatch_tribute_percentage = config['hatch_tribute_percentage']['value']

    @param.depends('action')
    def payout_view(self):
        scenario_rates = self.get_rate_scenarios()
        self.impact_hour_data['Minimum Payout (wXDAI)'] = self.impact_hour_data['Impact Hours'] * scenario_rates['min_rate']
        self.impact_hour_data['Target Payout (wXDAI)'] = self.impact_hour_data['Impact Hours'] * self.target_impact_hour_rate
        self.impact_hour_data['Maximum Payout (wXDAI)'] = self.impact_hour_data['Impact Hours'] * scenario_rates['max_rate']
        return self.impact_hour_data.hvplot.table(title='Impact Hour Results', width=450)

    def impact_hours_formula(self, minimum_raise, maximum_raise, raise_scenarios=None):
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
            
        R = self.maximum_impact_hour_rate
        m = self.impact_hour_slope
        H = self.total_impact_hours
        y = [R* (x / (x + m*H)) for x in x]
        df = pd.DataFrame([x,y]).T
        df.columns = ['Total wxDai Raised','Impact Hour Rate']
        return df

    @param.depends('action')
    def impact_hours_view(self):
        # Limits the target raise bounds when ploting the charts
        self.bounds_target_raise()
        self.df_impact_hours = self.impact_hours_formula(self.config_bounds['min_max_raise']['bounds'][0], self.max_raise)
        df = self.df_impact_hours
        df_fill_minimum = df[df['Total wxDai Raised'] <= self.min_raise]

        try:
            target_impact_hour_rate = df[df['Total wxDai Raised'] >= self.target_raise].iloc[0]['Impact Hour Rate']
        except:
            target_impact_hour_rate = 0

        impact_hours_plot = df.hvplot.area(title='Impact Hour Rate',
                                           x='Total wxDai Raised',
                                           xformatter='%.0f',
                                           yformatter='%.4f',
                                           hover=True,
                                           xlim=self.config_bounds['min_max_raise']['xlim'],
                                           label='Hatch happens ✅'
                                           ).opts(axiswise=True)
        minimum_raise_plot = df_fill_minimum.hvplot.area(x='Total wxDai Raised',
                                                         xformatter='%.0f',
                                                         yformatter='%.4f',
                                                         color='red',
                                                         xlim=self.config_bounds['min_max_raise']['xlim'],
                                                         label='Hatch fails 🚫'
                                                         ).opts(axiswise=True)

        # Enables the edition of constant params
        with param.edit_constant(self):
            self.target_impact_hour_rate = round(target_impact_hour_rate, 2)

        #return impact_hours_plot * hv.VLine(expected_raise) * hv.HLine(expected_impact_hour_rate) * hv.VLine(self.target_raise) * hv.HLine(target_impact_hour_rate)
        return (impact_hours_plot * 
                minimum_raise_plot *
                hv.VLine(self.target_raise).opts(color='#E31212') *
                hv.HLine(self.target_impact_hour_rate).opts(color='#E31212')
                ).opts(legend_position='top_left')

    def output_scenarios(self):
        df_hatch_params = self.df_impact_hours
        df_hatch_params['Cultural Build Tribute'] = (self.total_impact_hours * df_hatch_params['Impact Hour Rate'])/df_hatch_params['Total wxDai Raised']
        df_hatch_params['Hatch tribute'] = self.hatch_tribute_percentage / 100
        df_hatch_params['Redeemable'] = (1 - df_hatch_params['Hatch tribute'])/(1 + df_hatch_params['Cultural Build Tribute'])
        df_hatch_params['label'] = ""

        # Add label case there is already a row with raise value
        df_hatch_params.loc[df_hatch_params['Total wxDai Raised'] == int(self.min_raise), 'label'] = "Min Raise"
        df_hatch_params.loc[df_hatch_params['Total wxDai Raised'] == self.target_raise, 'label'] = "Target Raise"
        df_hatch_params.loc[df_hatch_params['Total wxDai Raised'] == int(self.max_raise), 'label'] = "Max Raise"
        df_hatch_params.loc[df_hatch_params['Total wxDai Raised'] < int(self.min_raise), ['Impact Hour Rate','Cultural Build Tribute', 'Hatch tribute']] = 0
        df_hatch_params.loc[df_hatch_params['Total wxDai Raised'] < int(self.min_raise), 'Redeemable'] = 1
        df_hatch_params.loc[df_hatch_params['Total wxDai Raised'] > int(self.max_raise), ['Impact Hour Rate','Cultural Build Tribute', 'Hatch tribute', 'Redeemable']] = np.nan

        return df_hatch_params

    def output_scenarios_out_issue(self):
        hatch_tribute = self.hatch_tribute_percentage / 100
        R = self.maximum_impact_hour_rate
        m = self.impact_hour_slope
        H = self.total_impact_hours

        df_hatch_params = self.impact_hours_formula(None, None, raise_scenarios=self.output_scenario_raise)
        df_hatch_params['Cultural Build Tribute'] = (H * df_hatch_params['Impact Hour Rate'])/df_hatch_params['Total wxDai Raised']
        df_hatch_params['Hatch tribute'] = df_hatch_params['Total wxDai Raised'].mul(hatch_tribute)
        df_hatch_params['Redeemable'] = (1 - hatch_tribute)/(1 + df_hatch_params['Cultural Build Tribute'])
        df_hatch_params['label'] = ""

        minimum_raise = int(self.min_raise)
        maximum_raise = int(self.max_raise)
        # Add 'Min Raise' label case there is already a row with min_raise value
        df_hatch_params.loc[df_hatch_params['Total wxDai Raised'] == minimum_raise, 'label'] = "Min Raise"

        # Add a new row with min_raise vale case there is no row with its value
        if "Min Raise" not in df_hatch_params['label']:
            impact_hour_rate = R* (minimum_raise / (minimum_raise + m*H))
            cultural_build_tribute = (H * impact_hour_rate)/minimum_raise
            f_hatch_params = df_hatch_params.sort_values(['Total wxDai Raised'])

        df_min_raise = df_hatch_params.query("label == 'Min Raise'")
        if len(df_min_raise) > 1:
            df_hatch_params = df_hatch_params.drop(df_min_raise.first_valid_index())

        # Add 'Target Raise' label case there is already a row with target_raise value
        df_hatch_params.loc[df_hatch_params['Total wxDai Raised'] == self.target_raise, 'label'] = "Target Raise"

        # Add a new row with target_raise vale case there is no row with its value
        if "Target Raise" not in df_hatch_params['label']:
            impact_hour_rate = R* (self.target_raise / (self.target_raise + m*H))
            cultural_build_tribute = (H * impact_hour_rate)/self.target_raise
            df_hatch_params = df_hatch_params.append({'Total wxDai Raised': self.target_raise, 'Impact Hour Rate':impact_hour_rate, 'Cultural Build Tribute':cultural_build_tribute, 'Hatch tribute':self.target_raise * hatch_tribute, 'Redeemable':(1 - hatch_tribute)/(1 + cultural_build_tribute), 'label':'Target Raise'}, ignore_index=True)
            df_hatch_params = df_hatch_params.sort_values(['Total wxDai Raised'])

        df_target_raise = df_hatch_params.query("label == 'Target Raise'")
        if len(df_target_raise) > 1:
            df_hatch_params = df_hatch_params.drop(df_target_raise.first_valid_index())

        # Add a new row with max_raise vale case there is no row with its value
        if "Max Raise" not in df_hatch_params['label']:
            impact_hour_rate = R* (maximum_raise / (maximum_raise + m*H))
            cultural_build_tribute = (H * impact_hour_rate)/maximum_raise
            df_hatch_params = df_hatch_params.append({'Total wxDai Raised': maximum_raise, 'Impact Hour Rate':impact_hour_rate, 'Cultural Build Tribute':cultural_build_tribute, 'Hatch tribute':maximum_raise * hatch_tribute, 'Redeemable':(1 - hatch_tribute)/(1 + cultural_build_tribute), 'label':'Max Raise'}, ignore_index=True)
            df_hatch_params = df_hatch_params.sort_values(['Total wxDai Raised'])

        df_max_raise = df_hatch_params.query("label == 'Max Raise'")
        if len(df_max_raise) > 1:
            df_hatch_params = df_hatch_params.drop(df_max_raise.first_valid_index())

        # Send to zero the IH rate, cultural build tribute and hatch tribue of
        # amount raises smaller than the min target
        df_hatch_params.loc[df_hatch_params['Total wxDai Raised'] < minimum_raise, ['Impact Hour Rate','Cultural Build Tribute', 'Hatch tribute']] = 0
        df_hatch_params.loc[df_hatch_params['Total wxDai Raised'] < minimum_raise, 'Redeemable'] = 1
        df_hatch_params.loc[df_hatch_params['Total wxDai Raised'] > maximum_raise, ['Impact Hour Rate','Cultural Build Tribute', 'Hatch tribute', 'Redeemable']] = np.nan

        # Format final table columns
        df_hatch_params['Redeemable'] = df_hatch_params['Redeemable'].mul(100)
        df_hatch_params['Cultural Build Tribute'] = df_hatch_params['Cultural Build Tribute'].mul(100)
        df_hatch_params = df_hatch_params.rename(columns={'Total wxDai Raised': 'Total wxDai Raised (wxDai)',
                                                          'Impact Hour Rate': 'Impact Hour Rate (wxDai)',
                                                          'Cultural Build Tribute': 'Cultural Build Tribute (%)',
                                                          'Hatch tribute': 'Hatch Tribute (wxDai)',
                                                          'Redeemable': 'Redeemable (%)',
                                                          'label': 'Label'})
        df_hatch_params = df_hatch_params.round(2)

        #df_hatch_params = df_hatch_params[df_hatch_params['Total XDAI Raised'].isin(x) | df_hatch_params['label'].isin(["Min Raise", "Target Raise", "Max Raise"])]
        return df_hatch_params

    @param.depends('action')
    def redeemable_plot(self):
        # Limits the target raise bounds when ploting the charts
        self.bounds_target_raise()
        df_hatch_params_to_plot = self.output_scenarios()
        # Drop NaN rows
        df_hatch_params_to_plot = df_hatch_params_to_plot.dropna()
        with pd.option_context('mode.chained_assignment', None):
            df_hatch_params_to_plot['Redeemable'] = df_hatch_params_to_plot['Redeemable'].mul(100)
        redeemable_plot = df_hatch_params_to_plot.hvplot.area(title='Redeemable (%)',
                                                              x='Total wxDai Raised',
                                                              y='Redeemable',
                                                              xformatter='%.0f',
                                                              yformatter='%.1f',
                                                              hover=True,
                                                              ylim=(0, 100),
                                                              xlim=self.config_bounds['min_max_raise']['xlim']
                                                              ).opts(axiswise=True)
        try:
            redeemable_target = df_hatch_params_to_plot[df_hatch_params_to_plot['Total wxDai Raised'] >= self.target_raise].iloc[0]['Redeemable']
        except:
            redeemable_target = 0
        
        with param.edit_constant(self):
            self.target_redeemable = round(redeemable_target, 2)

        return redeemable_plot * hv.VLine(self.target_raise).opts(color='#E31212') * hv.HLine(redeemable_target).opts(color='#E31212')

    @param.depends('action')
    def cultural_build_tribute_plot(self):
        # Limits the target raise bounds when ploting the charts
        self.bounds_target_raise()
        df_hatch_params_to_plot = self.output_scenarios()
        # Drop NaN rows
        df_hatch_params_to_plot = df_hatch_params_to_plot.dropna()
        with pd.option_context('mode.chained_assignment', None):
            df_hatch_params_to_plot['Cultural Build Tribute'] = df_hatch_params_to_plot['Cultural Build Tribute'].mul(100)
        cultural_build_tribute_plot = df_hatch_params_to_plot.hvplot.area(title='Cultural Build Tribute (%)',
                                                                          x='Total wxDai Raised',
                                                                          y='Cultural Build Tribute',
                                                                          xformatter='%.0f',
                                                                          yformatter='%.1f',
                                                                          hover=True,
                                                                          ylim=(0, 100),
                                                                          xlim=self.config_bounds['min_max_raise']['xlim']
                                                                          ).opts(axiswise=True)
        try:
            #cultural_build_tribute_target = df_hatch_params_to_plot.loc[df_hatch_params_to_plot['Total XDAI Raised'] == self.target_raise]['Cultural Build Tribute'].values[0]
            cultural_build_tribute_target = df_hatch_params_to_plot[df_hatch_params_to_plot['Total wxDai Raised'] >= self.target_raise].iloc[0]['Cultural Build Tribute']
        except:
            cultural_build_tribute_target = 0

        with param.edit_constant(self):
            self.target_cultural_build_tribute = round(cultural_build_tribute_target, 2)
        
        return cultural_build_tribute_plot * hv.VLine(self.target_raise).opts(color='#E31212') * hv.HLine(cultural_build_tribute_target).opts(color='#E31212')
        #return cultural_build_tribute_plot * hv.VLine(self.target_raise).opts(color='#E31212')
    
    def get_impact_hour_rate(self, raise_amount):
        rates = self.impact_hours_formula(0, int(self.max_raise))
        try:
            rate = rates[rates['Total wxDai Raised'].gt(raise_amount)].iloc[0]['Impact Hour Rate']
        except:
            rate = rates['Impact Hour Rate'].max()
        return rate

    def get_rate_scenarios(self):
        funding_pools = self.get_funding_pool_data().T
        scenarios = {
            'min_rate': self.get_impact_hour_rate(raise_amount=funding_pools.sum()['min_raise']),
            'target_rate': self.get_impact_hour_rate(raise_amount=funding_pools.sum()['target_raise']),
            'max_rate': self.get_impact_hour_rate(raise_amount=funding_pools.sum()['max_raise']),
        }
        return scenarios

    def get_raise_scenarios(self):
        scenarios = {
            'min_raise' : int(self.min_raise),
            'target_raise' : self.target_raise,
            'max_raise' : int(self.max_raise)
            #'max_raise' : min(int(self.min_max_raise[1]), self.hatch_oracle_ratio * self.total_cstk_tokens),
        }
        return scenarios

    def get_funding_pool_data(self):
        hatch_tribute = self.hatch_tribute_percentage / 100
        scenarios = self.get_raise_scenarios()
        funding_pool_data = {}
        for scenario, raise_amount in scenarios.items():
            impact_hour_rate = self.get_impact_hour_rate(raise_amount)
            cultural_tribute = min(raise_amount, self.get_impact_hour_rate(raise_amount) * self.total_impact_hours)
            redeemable_reserve = (raise_amount-cultural_tribute) * (1 - hatch_tribute)
            non_redeemable_reserve = (raise_amount-cultural_tribute) * hatch_tribute
            funding_pool_data[scenario] = {
                'Impact Hour Rate (wxDai/hour)': impact_hour_rate,
                'Hatch tribute': non_redeemable_reserve,
                'Cultural tribute': cultural_tribute,
                'Redeemable reserve': redeemable_reserve,
                'total': raise_amount,
            }
        return pd.DataFrame(funding_pool_data).T

    @param.depends('action')
    def funding_pool_view(self):
        funding_pools = self.get_funding_pool_data()
        funding_pools = funding_pools.filter(items=['Cultural tribute', 'Hatch tribute', 'Redeemable reserve', 'total'])
        # return funding_pools.hvplot.bar(title="Funding Pools", ylim=(0,self.param['hatch_oracle_ratio'].bounds[1]*self.param['min_max_raise'].bounds[1]), rot=45, yformatter='%.0f').opts(color=hv.Cycle(['#0F2EEE', '#0b0a15', '#DEFB48']))
        # raise_bars = bar_data.hvplot.bar(yformatter='%.0f', title="Funding Pools", stacked=True, y=['Funding Pool', 'Hatch Tribute']).opts(color=hv.Cycle(['#0F2EEE', '#0b0a15', '#DEFB48']))
        funding_pools['rank'] = funding_pools['total'] / funding_pools['total'].sum()
        idx_rank = funding_pools.sort_values(by='rank', ascending=False).index

        # Plot pie charts
        colors = ['#0F2EEE', '#0b0a15', '#DEFB48']
        chart_data = funding_pools.iloc[:,:-2]
        p1 = pie_chart(data=pd.Series(chart_data.loc['min_raise',:]),
                       radius=0.1 + 0.55 * int(self.min_raise)/int(self.config_bounds['min_max_raise']['bounds'][1]),
                       title="Min Raise", toolbar_location=None, plot_width=300,
                       show_legend=False, colors=colors)
        p2 = pie_chart(data=pd.Series(chart_data.loc['target_raise',:]),
                       radius=0.1 + 0.55 * int(self.target_raise)/int(self.config_bounds['min_max_raise']['bounds'][1]),
                       title="Target Raise", toolbar_location=None, plot_width=300,
                       show_legend=False, colors=colors)
        p3 = pie_chart(data=pd.Series(chart_data.loc['max_raise',:]),
                       radius=0.1 + 0.55 * int(self.max_raise)/int(self.config_bounds['min_max_raise']['bounds'][1]),
                       title="Max Raise", colors=colors)


        #return pn.Column('## Funding Pool', pn.Row(p1, p2, p3))
        return pn.Row(p1, p2, p3)
        
    @param.depends('action')
    def funding_pool_data_view(self):
        funding_pools = self.get_funding_pool_data()
        funding_pools['Cultural tribute'] = 100 * funding_pools['Cultural tribute'] / funding_pools['total']
        funding_pools['Redeemable reserve'] = 100 * funding_pools['Redeemable reserve'] / funding_pools['total']
        funding_pools = funding_pools.rename(columns={'Redeemable reserve': 'Redeemable %',
                                                      'Cultural tribute': 'Cultural tribute %'})
        funding_pools = funding_pools.T.reset_index()
        funding_pools = funding_pools.rename(columns={'index': 'Output',
                                                      'min_raise': 'Min Goal',
                                                      'target_raise': 'Target Goal',
                                                      'max_raise': 'Max Goal'})

        return funding_pools.hvplot.table(title='Outputs Overview', width=450)


    @param.depends('action')
    def bounds_target_raise(self):
        if self.target_raise > self.max_raise:
            self.target_raise = self.max_raise
        elif self.target_raise < self.min_raise:
            self.target_raise = self.min_raise

    @param.depends('action')
    def trigger_target_cultural_build_tribute_too_high(self):
        if self.target_cultural_build_tribute > 100:
            return pn.pane.JPG('https://i.imgflip.com/540z6u.jpg')
        else:
            return pn.pane.Markdown('')



class ImpactHoursData(param.Parameterized):
    historic = pd.read_csv('data/IHPredictions.csv').query('Model=="Historic"')
    optimistic = pd.read_csv('data/IHPredictions.csv').query('Model=="Optimistic"')
    predicted_hours = param.Number(0.5, bounds=(-.5,1.5), step=0.05)
    #total_impact_hours = param.Integer(step=100)

    def __init__(self, **params):
        super(ImpactHoursData, self).__init__(**params)
        historic = self.historic.set_index('Round')
        optimistic = self.optimistic[self.optimistic["Actual / Predicted"] == "Predicted"].set_index('Round')
        predicted = optimistic.copy()
        predicted['Total IH'] = self.predicted_hours * historic[historic["Actual / Predicted"] == "Predicted"]['Total IH'] + (1 - self.predicted_hours) * optimistic['Total IH']
        predicted['Total Hours'] = self.predicted_hours * historic[historic["Actual / Predicted"] == "Predicted"]['Total Hours'] + (1 - self.predicted_hours) * optimistic['Total Hours']
        self.total_impact_hours = int(predicted['Total IH'].max())


    def impact_hours_accumulation(self):
        x = 'End Date'

        historic = self.historic.set_index('Round')
        historic = historic[historic['Total IH'] != self.optimistic.set_index('Round')['Total IH']]
        optimistic = self.optimistic[self.optimistic["Actual / Predicted"] == "Predicted"].set_index('Round')
        predicted = optimistic.copy()
        predicted['Total IH'] = self.predicted_hours * historic[historic["Actual / Predicted"] == "Predicted"]['Total IH'] + (1 - self.predicted_hours) * optimistic['Total IH']
        predicted['Total Hours'] = self.predicted_hours * historic[historic["Actual / Predicted"] == "Predicted"]['Total Hours'] + (1 - self.predicted_hours) * optimistic['Total Hours']

        historic_curve = historic.hvplot(x, 'Total IH', rot=45, title='Impact Hours Accumulation Curve ')
        historic_bar = historic.hvplot.bar(x, 'Total Hours', label='Historic')

        optimistic_curve = optimistic.hvplot(x, 'Total IH')
        optimistic_bar = optimistic.hvplot.bar(x, 'Total Hours', label='Optimistic')

        predicted_curve = predicted.hvplot(x, 'Total IH', rot=45, title='Impact Hours Accumulation Curve :)')
        predicted_bar = predicted.hvplot.bar(x, 'Total Hours', label='Predicted')

        self.total_impact_hours = int(predicted['Total IH'].max())

        #return pn.Column(historic_curve * historic_bar * predicted_curve * predicted_bar * optimistic_curve * optimistic_bar, f"<b>Predicted Impact Hours: </b>{round(self.total_impact_hours)}")
        return pn.Column(historic_curve * historic_bar * predicted_curve * predicted_bar * optimistic_curve * optimistic_bar)


class ImpactHoursFormula(param.Parameterized):
    """
    Sem's Formula 🌱 🐝 🍯
    This formula was a collaboration of Sem and Griff for the TEC hatch impact hours formula.
    https://forum.tecommons.org/t/impact-hour-rewards-deep-dive/90/5
    """
     #total_impact_hours = param.Number(step=100)
    target_raise = param.Number(500, bounds=(20,1000), step=1, label="Target Goal (wxDai)")
    maximum_raise = param.Number(1000, bounds=(150,1000), step=1, label="Maximum Goal (wxDai)")
    minimum_raise = param.Number(5, bounds=(1, 100), step=1, label="Minimum Goal (wxDai)")
    hour_slope = param.Number(0.012, bounds=(0,1), step=0.001, label="Impact Hour Slope (wxDai/IH)")
    maximum_impact_hour_rate = param.Number(0.01, bounds=(0,10), step=0.01, label="Maximum Impact Hour Rate (wxDai/IH)")
    hatch_tribute_percentage = param.Number(5, bounds=(0,100), step=1, label="Hatch Tribute (%)")
    #expected_impact_hour_rate = param.Number()
    target_impact_hour_rate = param.Number(label="Target Impact Hour Rate (wxDai/hour)", constant=True)
    target_cultural_build_tribute = param.Number(label="Target Cultural Build Tribute (%)", constant=True)

    def __init__(self, total_impact_hours, impact_hour_data, **params):
        super(ImpactHoursFormula, self).__init__(**params)
        self.total_impact_hours = total_impact_hours
        self.impact_hour_data = impact_hour_data
        # self.maximum_raise = self.total_impact_hours * self.hour_slope * 10
        # self.param['maximum_raise'].bounds = (self.maximum_raise / 10, self.maximum_raise * 10)
        # self.param['maximum_raise'].step = self.maximum_raise / 10

        # self.target_raise = self.maximum_raise / 2
        # self.param['target_raise'].bounds = (self.minimum_raise, self.maximum_raise)
        # self.param['target_raise'].step = self.maximum_raise / 10

    def payout_view(self):
        # self.impact_hour_data['Expected Payout (wXDAI)'] = self.impact_hour_data['Impact Hours'] * self.expected_impact_hour_rate
        self.impact_hour_data['Target Payout (wXDAI)'] = self.impact_hour_data['Impact Hours'] * self.target_impact_hour_rate
        return self.impact_hour_data.hvplot.table()

    def impact_hours_rewards(self):
        # self.param['maximum_raise'].bounds = (expected_raise, expected_raise * 10)
        # self.param['maximum_raise'].step = expected_raise / 10
        # if self.target_raise > self.maximum_raise:
        # self.target_raise = self.maximum_raise
        # self.param['target_raise'].bounds = (self.minimum_raise, self.maximum_raise)
        # self.param['target_raise'].step = self.maximum_raise / 100

        x = np.linspace(1, self.maximum_raise, num=1000)

        R = self.maximum_impact_hour_rate

        m = self.hour_slope

        H = self.total_impact_hours

        y = [R* (x / (x + m*H)) for x in x]

        df = pd.DataFrame([x,y]).T
        df.columns = ['Total XDAI Raised','Impact Hour Rate']
        y_fill_minimum = [y[i] for i, x in enumerate(x) if x <= self.minimum_raise]
        df_fill_minimum = pd.DataFrame(zip(x,y_fill_minimum))

        try:
            target_impact_hour_rate = df[df['Total XDAI Raised'] > self.target_raise].iloc[0]['Impact Hour Rate']
        except:
            target_impact_hour_rate = df['Impact Hour Rate'].max()
        impact_hours_plot = df.hvplot.area(title='Impact Hour Rate', x='Total XDAI Raised', xformatter='%.0f', yformatter='%.4f', hover=True, xlim=(0,1000)).opts(axiswise=True)
        minimum_raise_plot = df_fill_minimum.hvplot.area(x='0', y='1', xformatter='%.0f', yformatter='%.4f', color='red').opts(axiswise=True)

        # Enables the edition of constant params
        with param.edit_constant(self):
            self.target_impact_hour_rate = target_impact_hour_rate
            self.target_cultural_build_tribute = 100 * (self.total_impact_hours * self.target_impact_hour_rate)/self.target_raise

        #return impact_hours_plot * hv.VLine(expected_raise) * hv.HLine(expected_impact_hour_rate) * hv.VLine(self.target_raise) * hv.HLine(target_impact_hour_rate)
        return impact_hours_plot * minimum_raise_plot * hv.VLine(self.target_raise).opts(color='#E31212') * hv.HLine(target_impact_hour_rate).opts(color='#E31212')

    def output_scenarios(self):
        hatch_tribute = self.hatch_tribute_percentage / 100
        x = list(range(1,1001))

        R = self.maximum_impact_hour_rate

        m = self.hour_slope

        H = self.total_impact_hours

        y = [R* (x / (x + m*H)) for x in x]

        df_hatch_params = pd.DataFrame([x,y]).T
        df_hatch_params.columns = ['Total XDAI Raised','Impact Hour Rate']
        df_hatch_params['Cultural Build Tribute'] = (H * df_hatch_params['Impact Hour Rate'])/df_hatch_params['Total XDAI Raised']
        df_hatch_params['Hatch tribute'] = hatch_tribute
        df_hatch_params['Redeemable'] = (1 - df_hatch_params['Hatch tribute'])/(1 + df_hatch_params['Cultural Build Tribute'])
        df_hatch_params['label'] = ""

        # Add 'Min Raise' label case there is already a row with min_raise value
        df_hatch_params.loc[df_hatch_params['Total XDAI Raised'] == self.minimum_raise, 'label'] = "Min Raise"

        # Add a new row with min_raise vale case there is no row with its value
        if "Min Raise" not in df_hatch_params['label']:
            impact_hour_rate = R* (self.minimum_raise / (self.minimum_raise + m*H))
            cultural_build_tribute = (H * impact_hour_rate)/self.minimum_raise
            df_hatch_params = df_hatch_params.append({'Total XDAI Raised': self.minimum_raise, 'Impact Hour Rate':impact_hour_rate, 'Cultural Build Tribute':cultural_build_tribute, 'Hatch tribute':hatch_tribute, 'Redeemable':(1 - hatch_tribute)/(1 + cultural_build_tribute), 'label':'Min Raise'}, ignore_index=True)
            df_hatch_params = df_hatch_params.sort_values(['Total XDAI Raised'])

        df_min_raise = df_hatch_params.query("label == 'Min Raise'")
        if len(df_min_raise) > 1:
            df_hatch_params = df_hatch_params.drop(df_min_raise.first_valid_index())

        # Add 'Target Raise' label case there is already a row with target_raise value
        df_hatch_params.loc[df_hatch_params['Total XDAI Raised'] == self.target_raise, 'label'] = "Target Raise"

        # Add a new row with target_raise vale case there is no row with its value
        if "Target Raise" not in df_hatch_params['label']:
            impact_hour_rate = R* (self.target_raise / (self.target_raise + m*H))
            cultural_build_tribute = (H * impact_hour_rate)/self.target_raise
            df_hatch_params = df_hatch_params.append({'Total XDAI Raised': self.target_raise, 'Impact Hour Rate':impact_hour_rate, 'Cultural Build Tribute':cultural_build_tribute, 'Hatch tribute':hatch_tribute, 'Redeemable':(1 - hatch_tribute)/(1 + cultural_build_tribute), 'label':'Target Raise'}, ignore_index=True)
            df_hatch_params = df_hatch_params.sort_values(['Total XDAI Raised'])

        df_target_raise = df_hatch_params.query("label == 'Target Raise'")
        if len(df_target_raise) > 1:
            df_hatch_params = df_hatch_params.drop(df_target_raise.first_valid_index())

        # Add a new row with max_raise vale case there is no row with its value
        if "Max Raise" not in df_hatch_params['label']:
            impact_hour_rate = R* (self.maximum_raise / (self.maximum_raise + m*H))
            cultural_build_tribute = (H * impact_hour_rate)/self.maximum_raise
            df_hatch_params = df_hatch_params.append({'Total XDAI Raised': self.maximum_raise, 'Impact Hour Rate':impact_hour_rate, 'Cultural Build Tribute':cultural_build_tribute, 'Hatch tribute':hatch_tribute, 'Redeemable':(1 - hatch_tribute)/(1 + cultural_build_tribute), 'label':'Max Raise'}, ignore_index=True)
            df_hatch_params = df_hatch_params.sort_values(['Total XDAI Raised'])

        df_max_raise = df_hatch_params.query("label == 'Max Raise'")
        if len(df_max_raise) > 1:
            df_hatch_params = df_hatch_params.drop(df_max_raise.first_valid_index())

        # Send to zero the IH rate, cultural build tribute and hatch tribue of
        # amount raises smaller than the min target
        df_hatch_params.loc[df_hatch_params['Total XDAI Raised'] < self.minimum_raise, ['Impact Hour Rate','Cultural Build Tribute', 'Hatch tribute']] = 0
        df_hatch_params.loc[df_hatch_params['Total XDAI Raised'] < self.minimum_raise, 'Redeemable'] = 1
        df_hatch_params.loc[df_hatch_params['Total XDAI Raised'] > self.maximum_raise, ['Impact Hour Rate','Cultural Build Tribute', 'Hatch tribute', 'Redeemable']] = np.nan

        return df_hatch_params

    def output_scenarios_table(self):
    # Simply creates an hvplot table with the output scenarios DataFrame
        df_hatch_params = self.output_scenarios()
        df_hatch_params_table = df_hatch_params[df_hatch_params['label'].isin(["Min Raise", "Target Raise", "Max Raise"])]
        return df_hatch_params_table.hvplot.table()

    def output_scenarios_out_issue(self):
        x = [1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 75, 100, 150, 175, 200, 250,
        300, 350, 400, 500, 600, 700, 800, 900, 1000]
        df_hatch_params = self.output_scenarios()
        df_hatch_params = df_hatch_params[df_hatch_params['Total XDAI Raised'].isin(x) | df_hatch_params['label'].isin(["Min Raise", "Target Raise", "Max Raise"])]
        return df_hatch_params

    def redeemable(self):
        df_hatch_params = self.output_scenarios()
        # Drop NaN rows
        df_hatch_params_to_plot = df_hatch_params.dropna()
        df_hatch_params_to_plot['Redeemable'] = df_hatch_params_to_plot['Redeemable'] * 100
        redeemable_plot = df_hatch_params_to_plot.hvplot.area(title='Redeemable (%)', x='Total XDAI Raised', y='Redeemable', xformatter='%.0f', yformatter='%.1f', hover=True, ylim=(0, 100), xlim=(0,1000)).opts(axiswise=True)
        redeemable_target = 100 * df_hatch_params[df_hatch_params_to_plot['Total XDAI Raised'] ==self.target_raise].iloc[0]['Redeemable']
        return redeemable_plot * hv.VLine(self.target_raise).opts(color='#E31212') * hv.HLine(redeemable_target).opts(color='#E31212')

    def cultural_build_tribute(self):
        df_hatch_params = self.output_scenarios()
        # Drop NaN rows
        df_hatch_params_to_plot = df_hatch_params.dropna()
        df_hatch_params_to_plot['Redeemable'] = df_hatch_params_to_plot['Redeemable'] * 100
        df_hatch_params_to_plot['Cultural Build Tribute'] = df_hatch_params_to_plot['Cultural Build Tribute'] * 100
        cultural_build_tribute_plot = df_hatch_params_to_plot.hvplot.area(title='Cultural Build Tribute (%)', x='Total XDAI Raised', y='Cultural Build Tribute', xformatter='%.0f', yformatter='%.1f', hover=True, ylim=(0, 100), xlim=(0,1000)).opts(axiswise=True)
        cultural_build_tribute_target = 100 * df_hatch_params[df_hatch_params_to_plot['Total XDAI Raised'] ==self.target_raise].iloc[0]['Cultural Build Tribute']
        return cultural_build_tribute_plot * hv.VLine(self.target_raise).opts(color='#E31212') * hv.HLine(cultural_build_tribute_target).opts(color='#E31212')


class Hatch(param.Parameterized):
    # Min and Target Goals
    target_raise = param.Number(500, bounds=(20,1000), step=1, label="Target Goal (wxDai)")
    max_raise = param.Number(1000, bounds=(150,1000), step=1, label="Maximum Goal (wxDai)")
    min_raise = param.Number(5, bounds=(1, 20), step=1, label="Minimum Goal (wxDai)")

    # CSTK Ratio
    #total_cstk_tokens = param.Number()
    hatch_oracle_ratio = param.Number(0.005, bounds=(0.001, 1), step=0.001, label="Membership (wxDai/CSTK)")

    # Hatch params
    hatch_period_days = param.Integer(15, bounds=(5, 30), step=2, label="Hatch Period (days)")

    # Number of TESTTEC exchanged for 1 wxdai
    hatch_exchange_rate = param.Number(10000, bounds=(1,100000), step=1, label="Hatch Minting Rate (TESTTECH/wxDai)")
    hatch_tribute_percentage = param.Number(5, bounds=(0,100), step=1, label="Hatch Tribute (%)")

    total_target_tech_tokens = param.Number(precedence=-1, label="Total target tech tokens (TESTTECH)")

    def __init__(self, cstk_data: pd.DataFrame, target_raise, total_impact_hours, target_impact_hour_rate,**params):
        super(Hatch, self).__init__(**params)
        self.total_impact_hours = total_impact_hours
        self.target_impact_hour_rate = target_impact_hour_rate
        self.cstk_data = cstk_data
        self.total_cstk_tokens = cstk_data['CSTK Tokens Capped'].sum()
        self.target_raise = target_raise

    def min_goal(self):
        return self.min_raise

    def max_goal(self):
        return self.max_raise

    def hatch_tribute(self):
        return self.hatch_tribute_percentage/100

    def wxdai_range(self):
        return pn.Row(pn.Pane("Cap on wxdai staked: "), self.hatch_oracle_ratio * self.total_cstk_tokens)

    def hatch_raise_view(self):
        # Load CSTK data
        # cstk_data = pd.read_csv('CSTK_DATA.csv', header=None).reset_index().head(100)
        # cstk_data.columns = ['CSTK Token Holders', 'CSTK Tokens']
        # cstk_data['CSTK Tokens Capped'] = cstk_data['CSTK Tokens'].apply(lambda x: min(x, cstk_data['CSTK Tokens'].sum()/10))
        self.cstk_data['Cap Raise'] = self.cstk_data['CSTK Tokens Capped'] * self.hatch_oracle_ratio

        # cap_plot = self.cstk_data.hvplot.area(title="Raise Targets Per Hatcher", x='CSTK Token Holders', y='Cap raise', yformatter='%.0f', label="Cap Raise", ylabel="XDAI Staked")

        self.cstk_data['Max Goal'] = self.max_raise
        # max_plot = self.cstk_data.hvplot.area(x='CSTK Token Holders', y='max_goal', yformatter='%.0f', label="Max Raise")

        self.cstk_data['Min Goal'] = self.min_raise
        # min_plot = self.cstk_data.hvplot.area(x='CSTK Token Holders', y='min_goal', yformatter='%.0f', label="Min Raise")

        self.cstk_data['Target Goal'] = self.target_raise
        # target_plot = self.cstk_data.hvplot.line(x='CSTK Token Holders', y='target_goal', yformatter='%.0f', label="Target Raise")

        bar_data = pd.DataFrame(self.cstk_data.iloc[:,3:].sum().sort_values(), columns=['Total'])
        bar_data['Hatch Tribute'] = bar_data['Total'] * self.hatch_tribute()
        bar_data['Funding Pool'] = bar_data['Total'] * (1-self.hatch_tribute())
        # raise_bars = bar_data.hvplot.bar(yformatter='%.0f', title="Funding Pools", stacked=True, y=['Funding Pool', 'Hatch Tribute']).opts(color=hv.Cycle(['#0F2EEE', '#0b0a15', '#DEFB48']))

        colors = ['#0F2EEE', '#0b0a15', '#DEFB48']
        min_data = pd.Series(bar_data.T.iloc[1:]['Min Goal'])
        min_pchart = pie_chart(data=min_data, radius=0.2, title='Min Goal', colors=colors[:2])
        target_data = pd.Series(bar_data.T.iloc[1:]['Target Goal'])
        target_pchart = pie_chart(data=target_data, radius=0.2, title='Target Goal', colors=colors[:2])
        max_data = pd.Series(bar_data.T.iloc[1:]['Max Goal'])
        max_pchart = pie_chart(data=max_data, radius=0.2, title='Max Goal', colors=colors[:2])

        funding_pool = pn.Column('### Funding Pool', pn.Row(min_pchart, target_pchart), max_pchart)

        stats = pd.DataFrame(self.cstk_data.iloc[:,3:].sum(), columns=['Total XDAI Staked'])
        stats['GMean XDAI Co-vested Per Hatcher'] = gmean(self.cstk_data.iloc[:,3:])
        stats['XDAI Hatch Tribute'] = stats['Total XDAI Staked'] * self.hatch_tribute()
        stats['XDAI Funding Pool'] = stats['Total XDAI Staked'] * (1 - self.hatch_tribute())
        stats['Total TECH Tokens'] = stats['Total XDAI Staked'] * self.hatch_exchange_rate

        self.total_target_tech_tokens = int(stats.loc['Target Goal']['Total TECH Tokens'])

        hatch_oracle_ratio_data = pd.DataFrame(data={'CSTK balance':[2000]})
        hatch_oracle_ratio_data['Max wxDAI to be sent'] = self.hatch_oracle_ratio * hatch_oracle_ratio_data['CSTK balance']
        hatch_oracle_ratio_bars = hatch_oracle_ratio_data.hvplot.bar(yformatter='%.0f', title="Hatch Oracle Ratio", stacked=False, y=['CSTK balance','Max wxDAI to be sent']).opts(color=hv.Cycle(colors), axiswise=True)

        #return pn.Column(cap_plot * max_plot * min_plot * target_plot, raise_bars, stats.sort_values('Total XDAI Staked',ascending=False).apply(round).reset_index().hvplot.table())
        #return pn.Column(raise_bars, stats.sort_values('Total XDAI Staked',ascending=False).apply(round).reset_index().hvplot.table())

        return pn.Column(hatch_oracle_ratio_bars, funding_pool)


class DandelionVoting(param.Parameterized):
    #total_tokens = param.Number(17e6)
    support_required_percentage = param.Number(60, bounds=(50,90), step=1, label="Support Required (%)")
    minimum_accepted_quorum_percentage = param.Number(2, bounds=(1,100), step=1, label="Minimum Quorum (%)")
    vote_duration_days = param.Integer(3, label="Vote Duration (days)")
    vote_buffer_hours = param.Integer(8, label="Vote Proposal buffer (hours)")
    rage_quit_hours = param.Integer(24, label="Rage quit (hours)")
    tollgate_fee_xdai = param.Number(3, label="Tollgate fee (wxDai)")
    action = param.Action(lambda x: x.param.trigger('action'), label='Run simulation')

    def __init__(self, total_tokens, config, **params):
        super(DandelionVoting, self).__init__(**params, name="TEC Hatch DAO")
        self.total_tokens=total_tokens

        # Change the parameter bound according the config_bound argument
        self.param.support_required_percentage.bounds = config['support_required_percentage']['bounds']
        self.param.support_required_percentage.step = config['support_required_percentage']['step']
        self.support_required_percentage = config['support_required_percentage']['value']

        self.param.minimum_accepted_quorum_percentage.bounds = config['minimum_accepted_quorum_percentage']['bounds']
        self.param.minimum_accepted_quorum_percentage.step = config['minimum_accepted_quorum_percentage']['step']
        self.minimum_accepted_quorum_percentage = config['minimum_accepted_quorum_percentage']['value']

        #self.param.vote_duration_days.bounds = config['vote_duration_days']['bounds']
        #self.param.vote_duration_days.step = config['vote_duration_days']['step']
        self.vote_duration_days = config['vote_duration_days']['value']

        #self.param.vote_buffer_hours.bounds = config['vote_buffer_hours']['bounds']
        #self.param.vote_buffer_hours.step = config['vote_buffer_hours']['step']
        self.vote_buffer_hours = config['vote_buffer_hours']['value']

        #self.param.rage_quit_hours.bounds = config['rage_quit_hours']['bounds']
        #self.param.rage_quit_hours.step = config['rage_quit_hours']['step']
        self.rage_quit_hours = config['rage_quit_hours']['value']

        #self.param.tollgate_fee_xdai.bounds = config['tollgate_fee_xdai']['bounds']
        #self.param.tollgate_fee_xdai.step = config['tollgate_fee_xdai']['step']
        self.tollgate_fee_xdai = config['tollgate_fee_xdai']['value']

    def support_required(self):
        return self.support_required_percentage/100
    
    def minimum_accepted_quorum(self):
        return self.minimum_accepted_quorum_percentage/100

    @param.depends('action')
    def vote_pass_view(self):
        x = np.linspace(0, 100, num=100)
        y = [a*self.support_required() for a in x]
        df = pd.DataFrame(zip(x,y))
        y_fill = x.tolist()
        df_fill = pd.DataFrame(zip(x,y_fill))
        y_fill_quorum = [a for i, a in enumerate(x) if i < self.minimum_accepted_quorum()*len(x)]
        df_fill_q = pd.DataFrame(zip(x,y_fill_quorum))
        total_votes_plot = df_fill.hvplot.area(
                title = "Minimum Support and Quorum Accepted for Proposals to Pass",
                x='0', y='1', xformatter='%.0f', yformatter='%.0f', color='green',
                xlabel='Total Token Votes (%)', ylabel='Yes Token Votes (%)', label='Proposal Passes ✅')
        support_required_plot = df.hvplot.area(x='0', y='1', xformatter='%.0f', yformatter='%.0f', color='red', label='Proposal Fails 🚫')
        quorum_accepted_plot = df_fill_q.hvplot.area(x='0', y='1', xformatter='%.0f', yformatter='%.0f', color='yellow', label='Minimum quorum')
        return (total_votes_plot* support_required_plot * quorum_accepted_plot).opts(legend_position='top_left') 
