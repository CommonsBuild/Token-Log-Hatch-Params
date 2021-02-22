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
        os.path.join(APP_PATH, "data", "TEC Praise Quantification.xlsx"),
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
    cstk_data = pd.read_csv('CSTK_DATA.csv', header=None).reset_index().head(100)
    cstk_data.columns = ['CSTK Token Holders', 'CSTK Tokens']
    cstk_data['CSTK Tokens Capped'] = cstk_data['CSTK Tokens'].apply(lambda x: min(x, cstk_data['CSTK Tokens'].sum()/10))
    return cstk_data


class ImpactHoursData(param.Parameterized):
    historic = pd.read_csv('data/IHPredictions.csv').query('Model=="Historic"')
    optimistic =  pd.read_csv('data/IHPredictions.csv').query('Model=="Optimistic"')
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

        historic_curve = historic.hvplot(x, 'Total IH', rot=45, title='Impact Hours Accumulation Curve 🧼')
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
    This formala was a collaboration of Sem and Griff for the TEC hatch impact hours formula.
    https://forum.tecommons.org/t/impact-hour-rewards-deep-dive/90/5
    """
    #total_impact_hours = param.Number(step=100)
    target_raise = param.Number(500, bounds=(20,1000), step=1, label="Target raise (wxDai)")
    maximum_raise = param.Number(1000, bounds=(150,1000), step=1, label="Maximum raise (wxDai)")
    minimum_raise = param.Number(5, bounds=(1, 100), step=1, label="Minimum raise (wxDai)")
    hour_slope = param.Number(0.012, bounds=(0,1), step=0.001, label="Impact hour slope (wxDai/IH)")
    maximum_impact_hour_rate = param.Number(0.01, bounds=(0,10), step=0.01, label="Maximum impact hour rate (wxDai/IH)")
    hatch_tribute_percentage = param.Number(5, bounds=(0,100), step=1, label="Hatch tribute (%)")

    #expected_impact_hour_rate = param.Number()
    target_impact_hour_rate = param.Number(label="Target impact hour rate (wxDai/hour)", constant=True)
    target_cultural_build_tribute = param.Number(label="Target Cultural Build Tribute (%)", constant=True)

    def __init__(self, total_impact_hours, impact_hour_data, **params):
        super(ImpactHoursFormula, self).__init__(**params)
        self.total_impact_hours = total_impact_hours
        self.impact_hour_data = impact_hour_data
#         self.maximum_raise = self.total_impact_hours * self.hour_slope * 10
#         self.param['maximum_raise'].bounds =  (self.maximum_raise / 10, self.maximum_raise * 10)
#         self.param['maximum_raise'].step = self.maximum_raise / 10

#         self.target_raise = self.maximum_raise / 2
#         self.param['target_raise'].bounds =  (self.minimum_raise, self.maximum_raise)
#         self.param['target_raise'].step = self.maximum_raise / 10

    def payout_view(self):
#       self.impact_hour_data['Expected Payout (wXDAI)'] = self.impact_hour_data['Impact Hours'] * self.expected_impact_hour_rate
        self.impact_hour_data['Target Payout (wXDAI)'] = self.impact_hour_data['Impact Hours'] * self.target_impact_hour_rate
        return self.impact_hour_data.hvplot.table()

    def impact_hours_rewards(self):
        expected_raise = self.total_impact_hours * self.hour_slope
        if expected_raise > self.maximum_raise:
            expected_raise = self.maximum_raise
#         self.param['maximum_raise'].bounds =  (expected_raise, expected_raise * 10)
#         self.param['maximum_raise'].step = expected_raise / 10
#         if self.target_raise > self.maximum_raise:
#             self.target_raise = self.maximum_raise
#         self.param['target_raise'].bounds =  (self.minimum_raise, self.maximum_raise)
#         self.param['target_raise'].step = self.maximum_raise / 100

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
            expected_impact_hour_rate = df[df['Total XDAI Raised'] > expected_raise].iloc[0]['Impact Hour Rate']
        except:
            expected_impact_hour_rate = df['Impact Hour Rate'].max()
        try:
            target_impact_hour_rate = df[df['Total XDAI Raised'] > self.target_raise].iloc[0]['Impact Hour Rate']
        except:
            target_impact_hour_rate = df['Impact Hour Rate'].max()
        impact_hours_plot = df.hvplot.area(title='Impact Hour Rate', x='Total XDAI Raised',  xformatter='%.0f', yformatter='%.4f', hover=True, xlim=(0,1000)).opts(axiswise=True)
        minimum_raise_plot = df_fill_minimum.hvplot.area(x='0', y='1', xformatter='%.0f', yformatter='%.4f', color='red').opts(axiswise=True)
        
        # Enables the edition of constant params
        with param.edit_constant(self):
            self.target_impact_hour_rate = target_impact_hour_rate
            self.target_cultural_build_tribute = 100 * (self.total_impact_hours * self.target_impact_hour_rate)/self.target_raise

        #return impact_hours_plot * hv.VLine(expected_raise) * hv.HLine(expected_impact_hour_rate) * hv.VLine(self.target_raise) * hv.HLine(target_impact_hour_rate)
        return impact_hours_plot * minimum_raise_plot * hv.VLine(self.target_raise).opts(color='#E31212') * hv.HLine(target_impact_hour_rate).opts(color='#E31212')

    def funding_pools(self):
        hatch_tribute = self.hatch_tribute_percentage / 100
        x_table = [1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 75, 100, 125, 150, 175, 200, 250, 300, 350, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000, 2500, 3000]
        x = list(range(1500))

        R = self.maximum_impact_hour_rate

        m = self.hour_slope

        H = self.total_impact_hours

        y = [R* (x / (x + m*H)) for x in x]


        df_hatch_params = pd.DataFrame([x,y]).T
        df_hatch_params.columns = ['Total XDAI Raised','Impact Hour Rate']

        df_hatch_params['Cultural Build Tribute'] = (H * df_hatch_params['Impact Hour Rate'])/df_hatch_params['Total XDAI Raised']
        df_hatch_params['Hatch tribute'] = hatch_tribute
        df_hatch_params['Redeemable'] = df_hatch_params['Hatch tribute']/(1 + df_hatch_params['Cultural Build Tribute'])
        df_hatch_params['label'] = ""


        # Add 'Min Raise' label case there is already a row with min_raise value
        df_hatch_params.loc[df_hatch_params['Total XDAI Raised'] == self.minimum_raise, 'label'] = "Min Raise"

        # Add a new row with min_raise vale case there is no row with its value
        if "Min Raise" not in df_hatch_params['label']:
            impact_hour_rate = R* (self.minimum_raise / (self.minimum_raise + m*H))
            cultural_build_tribute = (H * impact_hour_rate)/self.minimum_raise
            df_hatch_params = df_hatch_params.append({'Total XDAI Raised': self.minimum_raise, 'Impact Hour Rate':impact_hour_rate, 'Cultural Build Tribute':cultural_build_tribute, 'Hatch tribute':hatch_tribute, 'Redeemable':hatch_tribute/(1 + cultural_build_tribute), 'label':'Min Raise'}, ignore_index=True)
            df_hatch_params = df_hatch_params.sort_values(['Total XDAI Raised'])

        df_min_raise = df_hatch_params.query("label == 'Min Raise'")
        if len(df_min_raise) > 1:
            df_hatch_params = df_hatch_params.drop(df_min_raise.first_valid_index())

        # Add 'Tarfet Raise' label case there is already a row with target_raise value
        df_hatch_params.loc[df_hatch_params['Total XDAI Raised'] == self.target_raise, 'label'] = "Target Raise"

        # Add a new row with target_raise vale case there is no row with its value
        if "Target Raise" not in df_hatch_params['label']:
            impact_hour_rate = R* (self.target_raise / (self.target_raise + m*H))
            cultural_build_tribute = (H * impact_hour_rate)/self.target_raise
            df_hatch_params = df_hatch_params.append({'Total XDAI Raised': self.target_raise, 'Impact Hour Rate':impact_hour_rate, 'Cultural Build Tribute':cultural_build_tribute, 'Hatch tribute':hatch_tribute, 'Redeemable':hatch_tribute/(1 + cultural_build_tribute), 'label':'Target Raise'}, ignore_index=True)
            df_hatch_params = df_hatch_params.sort_values(['Total XDAI Raised'])

        df_target_raise = df_hatch_params.query("label == 'Target Raise'")
        if len(df_target_raise) > 1:
            df_hatch_params = df_hatch_params.drop(df_target_raise.first_valid_index())

        # Add 'Max Raise' label case there is already a row with max_raise value
        df_hatch_params.loc[df_hatch_params['Total XDAI Raised'] == self.maximum_raise, 'label'] = "Max Raise"

        # Add a new row with max_raise vale case there is no row with its value
        if "Max Raise" not in df_hatch_params['label']:
            impact_hour_rate = R* (self.maximum_raise / (self.maximum_raise + m*H))
            cultural_build_tribute = (H * impact_hour_rate)/self.maximum_raise
            df_hatch_params = df_hatch_params.append({'Total XDAI Raised': self.maximum_raise, 'Impact Hour Rate':impact_hour_rate, 'Cultural Build Tribute':cultural_build_tribute, 'Hatch tribute':hatch_tribute, 'Redeemable':hatch_tribute/(1 + cultural_build_tribute), 'label':'Max Raise'}, ignore_index=True)
            df_hatch_params = df_hatch_params.sort_values(['Total XDAI Raised'])

        df_max_raise = df_hatch_params.query("label == 'Max Raise'")
        if len(df_max_raise) > 1:
            df_hatch_params = df_hatch_params.drop(df_target_raise.first_valid_index())



        # Send to zero the IH rate, cultural build tribute and hatch tribue of
        # amount raises smaller than the min target
        df_hatch_params.loc[df_hatch_params['Total XDAI Raised'] < self.minimum_raise, ['Impact Hour Rate','Cultural Build Tribute', 'Hatch tribute']] = 0
        df_hatch_params.loc[df_hatch_params['Total XDAI Raised'] < self.minimum_raise, 'Redeemable'] = 1
        df_hatch_params.loc[df_hatch_params['Total XDAI Raised'] > self.maximum_raise, ['Impact Hour Rate','Cultural Build Tribute', 'Hatch tribute', 'Redeemable']] = np.nan

        # Drop NaN rows
        df_hatch_params_to_plot = df_hatch_params.dropna()
        df_hatch_params_table = df_hatch_params[df_hatch_params['Total XDAI Raised'].isin(x_table) | df_hatch_params['label'].isin(["Min Raise", "Target Raise", "Max Raise"])]

        df_hatch_params_to_plot['Redeemable'] = df_hatch_params_to_plot['Redeemable'] * 100
        df_hatch_params_to_plot['Cultural Build Tribute'] = df_hatch_params_to_plot['Cultural Build Tribute'] * 100

        cultural_build_tribute_plot = df_hatch_params_to_plot.hvplot.area(title='Cultural Build Tribute (%)', x='Total XDAI Raised', y='Cultural Build Tribute',  xformatter='%.0f', yformatter='%.1f', hover=True, ylim=(0, 100), xlim=(0,1000)).opts(axiswise=True)
        redeemable_plot = df_hatch_params_to_plot.hvplot.area(title='Redeemable (%)', x='Total XDAI Raised', y='Redeemable',  xformatter='%.0f', yformatter='%.1f', hover=True, ylim=(0, 100), xlim=(0,1000)).opts(axiswise=True)

        #return df_hatch_params.hvplot.table()
        return pn.Row(pn.Column(redeemable_plot, cultural_build_tribute_plot), df_hatch_params_table.hvplot.table())


class Hatch(param.Parameterized):
    # Min and Target Goals
    target_raise = param.Number(500, bounds=(20,1000), step=1, label="Target raise (wxDai)")
    max_raise = param.Number(1000, bounds=(150,1000), step=1, label="Maximum raise (wxDai)")
    min_raise = param.Number(5, bounds=(1, 20), step=1, label="Minimum raise (wxDai)")

    # CSTK Ratio
    #total_cstk_tokens = param.Number()
    hatch_oracle_ratio = param.Number(0.005, bounds=(0.001, 1), step=0.001, label="Hatch oracle ratio (wxDai/CSTK)")

    # Hatch params
    hatch_period_days = param.Integer(15, bounds=(5, 30), step=2, label="Hatch period (days)")

    # Number of TESTTEC exchanged for 1 wxdai
    hatch_exchange_rate = param.Number(10000, bounds=(1,100000), step=1, label="Hatch exchange rate (TESTTECH/wxDai)")
    hatch_tribute_percentage = param.Number(5, bounds=(0,100), step=1, label="Hatch tribute (%)")

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
#         cstk_data = pd.read_csv('CSTK_DATA.csv', header=None).reset_index().head(100)
#         cstk_data.columns = ['CSTK Token Holders', 'CSTK Tokens']
#         cstk_data['CSTK Tokens Capped'] = cstk_data['CSTK Tokens'].apply(lambda x: min(x, cstk_data['CSTK Tokens'].sum()/10))
        self.cstk_data['Cap raise'] = self.cstk_data['CSTK Tokens Capped'] * self.hatch_oracle_ratio

        cap_plot = self.cstk_data.hvplot.area(title="Raise Targets Per Hatcher", x='CSTK Token Holders', y='Cap raise', yformatter='%.0f', label="Cap Raise", ylabel="XDAI Staked")

        self.cstk_data['max_goal'] = self.max_raise
        max_plot = self.cstk_data.hvplot.area(x='CSTK Token Holders', y='max_goal', yformatter='%.0f', label="Max Raise")

        self.cstk_data['min_goal'] = self.min_raise
        min_plot = self.cstk_data.hvplot.area(x='CSTK Token Holders', y='min_goal', yformatter='%.0f', label="Min Raise")

        self.cstk_data['target_goal'] = self.target_raise
        target_plot = self.cstk_data.hvplot.line(x='CSTK Token Holders', y='target_goal', yformatter='%.0f', label="Target Raise")
        
        bar_data = pd.DataFrame(self.cstk_data.iloc[:,3:].sum().sort_values(), columns=['Total'])
        bar_data['Hatch Tribute'] = bar_data['Total'] * self.hatch_tribute()
        bar_data['Funding Pool'] = bar_data['Total'] * (1-self.hatch_tribute())
        raise_bars = bar_data.hvplot.bar(yformatter='%.0f', title="Funding Pools", stacked=True, y=['Funding Pool', 'Hatch Tribute']).opts(color=hv.Cycle(['#0F2EEE', '#0b0a15', '#DEFB48']))

        stats = pd.DataFrame(self.cstk_data.iloc[:,3:].sum(), columns=['Total XDAI Staked'])
        stats['GMean XDAI Co-vested Per Hatcher'] = gmean(self.cstk_data.iloc[:,3:])
        stats['XDAI Hatch Tribute'] = stats['Total XDAI Staked'] * self.hatch_tribute()
        stats['XDAI Funding Pool'] = stats['Total XDAI Staked'] * (1 - self.hatch_tribute())
        stats['Total TECH Tokens'] = stats['Total XDAI Staked'] * self.hatch_exchange_rate

        self.total_target_tech_tokens = int(stats.loc['target_goal']['Total TECH Tokens'])

        hatch_oracle_ratio_data = pd.DataFrame(data={'CSTK balance':[2000]})
        hatch_oracle_ratio_data['Max wxDAI to be sent'] = self.hatch_oracle_ratio * hatch_oracle_ratio_data['CSTK balance']
        hatch_oracle_ratio_bars = hatch_oracle_ratio_data.hvplot.bar(yformatter='%.0f', title="Hatch Oracle Ratio", stacked=False, y=['CSTK balance','Max wxDAI to be sent']).opts(color=hv.Cycle(['#0F2EEE', '#0b0a15', '#DEFB48']), axiswise=True)

        #return pn.Column(cap_plot * max_plot * min_plot * target_plot, raise_bars, stats.sort_values('Total XDAI Staked',ascending=False).apply(round).reset_index().hvplot.table())
        #return pn.Column(raise_bars, stats.sort_values('Total XDAI Staked',ascending=False).apply(round).reset_index().hvplot.table())
        return pn.Column(hatch_oracle_ratio_bars, raise_bars)

class DandelionVoting(param.Parameterized):
    #total_tokens = param.Number(17e6)
    support_required_percentage = param.Number(60, bounds=(50,90), step=1, label="Support required (%)")
    minimum_accepted_quorum_percentage = param.Number(2, bounds=(1,100), step=1, label="Minimum accepted quorum (%)")
    vote_duration_days = param.Number(3, bounds=(1,14), step=1, label="Vote duration (days)")
    vote_buffer_hours = param.Number(8, bounds=(1,48), step=1, label="Vote buffer (hours)")
    rage_quit_hours = param.Number(24, bounds=(1, 48), step=1, label="Rage quit (hours)")
    tollgate_fee_xdai = param.Number(3, bounds=(1,100), step=1, label="Tollgate fee (wxDai)")

    def __init__(self, total_tokens, **params):
        super(DandelionVoting, self).__init__(**params)
        self.total_tokens=total_tokens

    def support_required(self):
        return self.support_required_percentage/100

    def minimum_accepted_quorum(self):
        return self.minimum_accepted_quorum_percentage/100

    def vote_pass_view(self):
        x = np.linspace(0, 100, num=100)
        y = [a*self.support_required() for a in x]
        df = pd.DataFrame(zip(x,y))
        y_fill = [a for a in x]
        df_fill = pd.DataFrame(zip(x,y_fill))
        y_fill_quorum = [a for i, a in enumerate(x) if i < self.minimum_accepted_quorum()*len(x)]
        df_fill_q = pd.DataFrame(zip(x,y_fill_quorum))
        total_votes_plot = df_fill.hvplot.area(
                title = "Minimum Support and Quorum Accepted for Proposals to Pass",
                x='0', y='1', xformatter='%.0f', yformatter='%.0f', color='green',
                xlabel='Total Token Votes (%)', ylabel='Yes Token Votes (%)')
        support_required_plot = df.hvplot.area(x='0', y='1', xformatter='%.0f', yformatter='%.0f', color='red')
        quorum_accepted_plot = df_fill_q.hvplot.area(x='0', y='1', xformatter='%.0f', yformatter='%.0f', color='#0F2EEE')
        return total_votes_plot * support_required_plot * quorum_accepted_plot
