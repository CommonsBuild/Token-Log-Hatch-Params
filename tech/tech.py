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

class TECH(param.Parameterized):
    min_max_raise = param.Range((5, 1000), bounds=(1,1000))
    target_raise = param.Number(500, bounds=(5,1000), step=1)
    impact_hour_slope = param.Number(0.012, bounds=(0,1), step=0.001)
    maximum_impact_hour_rate = param.Number(0.01, bounds=(0,1), step=0.01)
    hatch_oracle_ratio = param.Number(0.005, bounds=(0.001, 1), step=0.001)
    hatch_period_days = param.Integer(15, bounds=(5, 30), step=2)
    hatch_exchange_rate = param.Number(10000, bounds=(1,100000), step=1)
    hatch_tribute = param.Number(0.05, bounds=(0,1), step=0.01)
    target_impact_hour_rate = param.Number(precedence=-1)

    def __init__(self, total_impact_hours, impact_hour_data, total_cstk_tokens, **params):
        super(TECH, self).__init__(**params)
        self.total_impact_hours = total_impact_hours
        self.impact_hour_data = impact_hour_data
        self.total_cstk_tokens = total_cstk_tokens

    def payout_view(self):
        scenario_rates = self.get_rate_scenarios()
        self.impact_hour_data['Minimum Payout (wXDAI)'] = self.impact_hour_data['Impact Hours'] * scenario_rates['min_rate']
        self.impact_hour_data['Target Payout (wXDAI)'] = self.impact_hour_data['Impact Hours'] * self.target_impact_hour_rate
        self.impact_hour_data['Maximum Payout (wXDAI)'] = self.impact_hour_data['Impact Hours'] * scenario_rates['max_rate']
        return self.impact_hour_data.hvplot.table()

    def impact_hours_formula(self, minimum_raise, maximum_raise):
        x = np.linspace(minimum_raise, maximum_raise, num=500)
        R = self.maximum_impact_hour_rate
        m = self.impact_hour_slope
        H = self.total_impact_hours
        y = [R* (x / (x + m*H)) for x in x]
        df = pd.DataFrame([x,y]).T
        df.columns = ['Total XDAI Raised','Impact Hour Rate']
        return df


    def impact_hours_view(self):
        df = self.impact_hours_formula(0, self.min_max_raise[1])
        df['Passes Minimum'] = df['Total XDAI Raised'] >= self.min_max_raise[0] - 1
        feasible_raise = self.hatch_oracle_ratio * self.total_cstk_tokens
        df['Feasible Raise'] = df['Passes Minimum'] & (df['Total XDAI Raised'] <= feasible_raise + 1)
        
        try:
            self.target_impact_hour_rate = df[df['Total XDAI Raised'] > self.target_raise].iloc[0]['Impact Hour Rate']
        except:
            self.target_impact_hour_rate = df['Impact Hour Rate'].max()

        impact_hours_plot_fails = df[df['Passes Minimum']==False].hvplot.area(color='red', y=['Impact Hour Rate'], title='Impact Hour Rate', x='Total XDAI Raised',  xformatter='%.0f', hover=True)
        impact_hours_plot_passes = df[df['Passes Minimum']==True].hvplot.area(color='orange', y=['Impact Hour Rate'], title='Impact Hour Rate', x='Total XDAI Raised',  xformatter='%.0f', hover=True)
        impact_hours_plot_feasible = df[df['Feasible Raise']==True].hvplot.area(xlim=(0,self.param["min_max_raise"].bounds[1]), ylim=(0, self.param["maximum_impact_hour_rate"].bounds[1]), color='blue', y=['Impact Hour Rate'], title='Impact Hour Rate', x='Total XDAI Raised',  xformatter='%.0f', hover=True)
        impact_hours_plot = impact_hours_plot_passes * impact_hours_plot_feasible * impact_hours_plot_fails
        target_plot = hv.VLine(self.target_raise).opts(color='#E31212') * hv.HLine(self.target_impact_hour_rate).opts(color='#E31212')
        return impact_hours_plot * target_plot

    def get_impact_hour_rate(self, raise_amount):
        rates = self.impact_hours_formula(0, self.min_max_raise[1])
        try:
            rate = rates[rates['Total XDAI Raised'].gt(raise_amount)].iloc[0]['Impact Hour Rate']
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
            'min_raise' : self.min_max_raise[0],
            'target_raise' : self.target_raise,
            'max_raise' : min(self.min_max_raise[1], self.hatch_oracle_ratio * self.total_cstk_tokens),
        }
        return scenarios

    def get_funding_pool_data(self):
        scenarios = self.get_raise_scenarios()
        funding_pool_data = {}
        for scenario, raise_amount in scenarios.items():
            cultural_tribute = min(raise_amount, self.get_impact_hour_rate(raise_amount) * self.total_impact_hours)
            redeemable_reserve = (raise_amount-cultural_tribute) * (1 - self.hatch_tribute)
            non_redeemable_reserve = (raise_amount-cultural_tribute) * self.hatch_tribute
            funding_pool_data[scenario] = {
                'cultural_tribute': cultural_tribute,
                'non_redeemable_reserve': non_redeemable_reserve,
                'redeemable_reserve': redeemable_reserve,
                'total': raise_amount,
            }
        return pd.DataFrame(funding_pool_data).T

    def funding_pool_view(self):
        funding_pools = self.get_funding_pool_data()
        return funding_pools.hvplot.bar(title="Funding Pools", ylim=(0,self.param['hatch_oracle_ratio'].bounds[1]*self.param['min_max_raise'].bounds[1]), rot=45, yformatter='%.0f').opts(color=hv.Cycle(['#0F2EEE', '#0b0a15', '#DEFB48']))
        # raise_bars = bar_data.hvplot.bar(yformatter='%.0f', title="Funding Pools", stacked=True, y=['Funding Pool', 'Hatch Tribute']).opts(color=hv.Cycle(['#0F2EEE', '#0b0a15', '#DEFB48']))

    def funding_pool_data_view(self):
        funding_pools = self.get_funding_pool_data()
        return funding_pools.T.reset_index().hvplot.table(width=300)





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
    target_raise = param.Number(500, bounds=(20,1000), step=1)
    maximum_raise = param.Number(1000, bounds=(20,1000), step=1)
    minimum_raise = param.Number(5, bounds=(1, 100), step=1)
    hour_slope = param.Number(0.012, bounds=(0,1), step=0.001)
    maximum_impact_hour_rate = param.Number(0.01, bounds=(0,10), step=0.01)
    
    #expected_impact_hour_rate = param.Number()
    target_impact_hour_rate = param.Number()
    
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
        #self.impact_hour_data['Expected Payout (wXDAI)'] = self.impact_hour_data['Impact Hours'] * self.expected_impact_hour_rate
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
        
        x = np.linspace(self.minimum_raise, self.maximum_raise)

        R = self.maximum_impact_hour_rate

        m = self.hour_slope
        
        H = self.total_impact_hours

        y = [R* (x / (x + m*H)) for x in x]

        df = pd.DataFrame([x,y]).T
        df.columns = ['Total XDAI Raised','Impact Hour Rate']
        
       
        try:
            expected_impact_hour_rate = df[df['Total XDAI Raised'] > expected_raise].iloc[0]['Impact Hour Rate']
        except:
            expected_impact_hour_rate = df['Impact Hour Rate'].max()
        try:
            target_impact_hour_rate = df[df['Total XDAI Raised'] > self.target_raise].iloc[0]['Impact Hour Rate']
        except:
            target_impact_hour_rate = df['Impact Hour Rate'].max()
        impact_hours_plot = df.hvplot.area(title='Impact Hour Rate', x='Total XDAI Raised',  xformatter='%.0f', hover=True)
        
        #return impact_hours_plot * hv.VLine(expected_raise) * hv.HLine(expected_impact_hour_rate) * hv.VLine(self.target_raise) * hv.HLine(target_impact_hour_rate)
        return impact_hours_plot * hv.VLine(self.target_raise).opts(color='#E31212') * hv.HLine(target_impact_hour_rate).opts(color='#E31212')

    def funding_pools(self):
        x = np.linspace(self.minimum_raise, self.maximum_raise)

        R = self.maximum_impact_hour_rate

        m = self.hour_slope
        
        H = self.total_impact_hours

        y = [R* (x / (x + m*H)) for x in x]

        df = pd.DataFrame([x,y]).T
        df.columns = ['Total XDAI Raised','Impact Hour Rate']
        
        # Minimum Results
        minimum_raise = self.minimum_raise
        minimum_rate = df[df['Total XDAI Raised'] > minimum_raise].iloc[0]['Impact Hour Rate']
        minimum_cultural_tribute = self.total_impact_hours * minimum_rate
        
        # Expected Results
        expected_raise = self.total_impact_hours * self.hour_slope
        try:
            expected_rate = df[df['Total XDAI Raised'] > expected_raise].iloc[0]['Impact Hour Rate']
        except:
            expected_rate = df['Impact Hour Rate'].max()
        self.expected_impact_hour_rate = round(expected_rate, 5)
            
        expected_cultural_tribute = self.total_impact_hours * expected_rate

        # Target Results
        target_raise = self.target_raise
        try:
            target_rate = df[df['Total XDAI Raised'] > target_raise].iloc[0]['Impact Hour Rate']
        except:
            target_rate = df['Impact Hour Rate'].max()
        self.target_impact_hour_rate = round(target_rate, 5)
            
        target_cultural_tribute = self.total_impact_hours * target_rate

        # Funding Pools and Tribute
        funding = pd.DataFrame.from_dict({
            'Mimimum': [minimum_cultural_tribute, minimum_raise-minimum_cultural_tribute],
            #'Expected': [expected_cultural_tribute, expected_raise-expected_cultural_tribute],
            'Target': [target_cultural_tribute, target_raise-target_cultural_tribute]}, orient='index', columns=['Culture Tribute', 'Funding Pool'])
        funding_plot = funding.hvplot.bar(title="Target Funding Pools", stacked=True, ylim=(0,self.maximum_raise),  yformatter='%.0f').opts(color=hv.Cycle(['#0F2EEE', '#0b0a15', '#DEFB48']))

        return funding_plot
    
    
class Hatch(param.Parameterized):
    # Min and Target Goals
    target_raise = param.Number(500, bounds=(20,1000), step=1)
    max_raise = param.Number(1000, bounds=(20,1000), step=1)
    min_raise = param.Number(5, bounds=(1, 100), step=1)

    # CSTK Ratio
    #total_cstk_tokens = param.Number()
    hatch_oracle_ratio = param.Number(0.005, bounds=(0.001, 1), step=0.001)
       
    # Hatch params
    hatch_period_days = param.Integer(15, bounds=(5, 30), step=2)
    
    # Number of TESTTEC exchanged for 1 wxdai
    hatch_exchange_rate = param.Number(10000, bounds=(1,100000), step=1) 
    hatch_tribute = param.Number(0.05, bounds=(0,1), step=0.01)    
    
    total_target_tech_tokens = param.Number(precedence=-1)
    
    def __init__(self, cstk_data: pd.DataFrame, **params):
        super(Hatch, self).__init__(**params)
        self.cstk_data = cstk_data
        self.total_cstk_tokens = cstk_data['CSTK Tokens Capped'].sum()
    
    def min_goal(self):
        return self.min_raise
    
    def max_goal(self):
        return self.max_raise

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
        bar_data['Hatch Tribute'] = bar_data['Total'] * self.hatch_tribute
        bar_data['Funding Pool'] = bar_data['Total'] * (1-self.hatch_tribute)
        raise_bars = bar_data.hvplot.bar(yformatter='%.0f', title="Funding Pools", stacked=True, y=['Funding Pool', 'Hatch Tribute']).opts(color=hv.Cycle(['#0F2EEE', '#0b0a15', '#DEFB48']))
        
        stats = pd.DataFrame(self.cstk_data.iloc[:,3:].sum(), columns=['Total XDAI Staked'])
        stats['GMean XDAI Co-vested Per Hatcher'] = gmean(self.cstk_data.iloc[:,3:])
        stats['XDAI Hatch Tribute'] = stats['Total XDAI Staked'] * self.hatch_tribute
        stats['XDAI Funding Pool'] = stats['Total XDAI Staked'] * (1 - self.hatch_tribute)
        stats['Total TECH Tokens'] = stats['Total XDAI Staked'] * self.hatch_exchange_rate
        
        self.total_target_tech_tokens = int(stats.loc['target_goal']['Total TECH Tokens'])

        #return pn.Column(cap_plot * max_plot * min_plot * target_plot, raise_bars, stats.sort_values('Total XDAI Staked',ascending=False).apply(round).reset_index().hvplot.table())
        #return pn.Column(raise_bars, stats.sort_values('Total XDAI Staked',ascending=False).apply(round).reset_index().hvplot.table())
        return raise_bars
    
class DandelionVoting(param.Parameterized):
    #total_tokens = param.Number(17e6)
    support_required = param.Number(0.6, bounds=(0.5,0.9), step=0.01)
    minimum_accepted_quorum = param.Number(0.02, bounds=(0.01,1), step=0.01)
    vote_duration_days = param.Number(3, bounds=(1,14), step=1)
    vote_buffer_hours = param.Number(8, bounds=(1,48), step=1)
    rage_quit_hours = param.Number(24, bounds=(1, 48), step=1)
    tollgate_fee_xdai = param.Number(3, bounds=(1,100), step=1)
    
    def __init__(self, total_tokens, **params):
        super(DandelionVoting, self).__init__(**params)
        self.total_tokens=total_tokens
    
    def vote_pass_view(self):
        x = np.linspace(0, self.total_tokens, num=100)
        y = [a*self.support_required for a in x]
        df = pd.DataFrame(zip(x,y))
        y_fill = [a for a in x]
        df_fill = pd.DataFrame(zip(x,y_fill))
        y_fill_quorum = [a for i, a in enumerate(x) if i < self.minimum_accepted_quorum*len(x)]
        df_fill_q = pd.DataFrame(zip(x,y_fill_quorum))
        total_votes_plot = df_fill.hvplot.area(
                title = "Minimum Support and Quorum Accepted for Proposals to Pass", 
                x='0', y='1', xformatter='%.0f', yformatter='%.0f', color='green', 
                xlabel='Total Token Votes', ylabel='Yes Token Votes')
        support_required_plot = df.hvplot.area(x='0', y='1', xformatter='%.0f', yformatter='%.0f', color='red')
        quorum_accepted_plot = df_fill_q.hvplot.area(x='0', y='1', xformatter='%.0f', yformatter='%.0f', color='#0F2EEE')
        return total_votes_plot * support_required_plot * quorum_accepted_plot
