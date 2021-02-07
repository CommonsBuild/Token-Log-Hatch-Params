import panel as pn
import holoviews as hv
import numpy as np


react = pn.template.ReactTemplate(title='React Template')

pn.config.sizing_mode = 'stretch_both'

xs = np.linspace(0, np.pi)
freq = pn.widgets.FloatSlider(name="Frequency", start=0, end=10, value=2)
phase = pn.widgets.FloatSlider(name="Phase", start=0, end=np.pi)

@pn.depends(freq=freq, phase=phase)
def sine(freq, phase):
    return hv.Curve((xs, np.sin(xs*freq+phase))).opts(
        responsive=True, min_height=400)

@pn.depends(freq=freq, phase=phase)
def cosine(freq, phase):
    return hv.Curve((xs, np.cos(xs*freq+phase))).opts(
        responsive=True, min_height=400)

react.sidebar.append(freq)
react.sidebar.append(phase)

# Unlike other templates the `ReactTemplate.main` area acts like a GridSpec 
react.main[:4, :6] = pn.Card(hv.DynamicMap(sine), title='Sine')
react.main[:4, 6:] = pn.Card(hv.DynamicMap(cosine), title='Cosine')

react.servable();
