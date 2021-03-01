from bokeh.palettes import Category20c
from bokeh.plotting import figure
from bokeh.plotting.figure import Figure
from bokeh.transform import cumsum

from math import pi
import pandas as pd
from typing import List


def pie_chart(data: pd.Series, 
              colors: List[str] = None, 
              title: str = None, 
              plot_height: int = 250, 
              plot_width=None,
              radius: int = 0.1, 
              toolbar_location: str = 'right',
              x_range=None,
              show_legend=True
             ) -> Figure:
    data = data.reset_index(name='value').rename(columns={'index':'column'})
    data['angle'] = data['value']/data['value'].sum() * 2*pi
    data['perc'] = data['value']/data['value'].sum() * 100
    if colors:
        data['color'] = colors
    else:
        data['color'] = Category20c[len(data)] if len(data) > 2 else Category20c[3][:2]

    p = figure(plot_height=plot_height,
               title=title, toolbar_location=toolbar_location,
               tools="pan,save,hover", tooltips="@column: @value (@perc%)",
               x_range=x_range,
              )

    p.wedge(x=0, y=1, radius=radius,
            start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
            line_color="white", fill_color='color', legend_field='column', source=data)

    p.axis.axis_label = None
    p.axis.visible = False
    p.grid.grid_line_color = None
    p.legend.visible = show_legend
    
    if plot_width:
        p.width = plot_width
        
    return p

