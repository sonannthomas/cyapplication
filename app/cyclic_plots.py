import numpy as np
#import matplotlib.pyplot as plt

#import scipy
#import scipy.io
#import toy_example as toy
#import cyclic_analysis as cal

from bokeh.plotting import figure,  output_file, save
from bokeh.layouts import layout
from bokeh.palettes import (Blues9, BrBG9, BuGn9, BuPu9, GnBu9, Greens9,
                            Greys9, OrRd9, Oranges9, PRGn9, PiYG9, PuBu9,
                            PuBuGn9, PuOr9, PuRd9, Purples9, RdBu9, RdGy9,
                            RdPu9, RdYlBu9, RdYlGn9, Reds9, Spectral9, YlGn9,
                            YlGnBu9, YlOrBr9, YlOrRd9, Inferno9, Magma9,
                            Plasma9, Viridis9, Accent8, Dark2_8, Paired9,
                            Pastel1_9, Pastel2_8, Set1_9, Set2_8, Set3_9)

from bokeh.models import ColumnDataSource, Range1d, LabelSet, Label, HoverTool

#from abbreviate import Abbreviate


def plot_signals(lines,channel_names, phases, perm, lm, evals):
	hover = HoverTool(
				tooltips=[
					("index", "$index"),
					("(x,y)", "($x, $y)"),
					("desc", "@desc"),
				]
	)

	p0 = figure(title="Signal plot", width=750, plot_height=250, tools=['pan','box_zoom','wheel_zoom','save','reset',hover])
	p0.grid.grid_line_alpha = 0.3
	p0.xaxis.axis_label = 'n'
	p0.yaxis.axis_label = 'x[n]'



	palette = Set1_9

	for l in enumerate(lines):
		data = dict()
		data['x'] = np.arange(len(l[1]))
		data['y'] = l[1]
		data['desc']=[channel_names[l[0]]]*len(l[1])
		src = ColumnDataSource(data=data)
		p0.line('x','y',color=palette[l[0] % len(palette)] ,source=src)



	return p0


def plot_results(lines, channel_names, phases, perm, lm, evals):
	unit_circle = np.linspace(-np.pi,np.pi,100)

	angs = np.angle(phases[perm])
	rads = np.abs(phases[perm])


	p1 = figure(title="Component Phase Diagram",plot_width = 320, plot_height=320,tools=['pan','box_zoom','wheel_zoom','save','reset'])
	p1.grid.grid_line_alpha = 0.3
	p1.xaxis.axis_label = 'Real'
	p1.yaxis.axis_label = 'Imaginary'

	# Plot axis system (Real, imaginary, unit circle)
	dim = rads.max()
	p1.line([-dim,dim],[0,0], line_color="black", line_width=2)
	p1.line([0, 0], [-dim, dim], line_color="black", line_width=2)
	p1.line(dim*np.cos(unit_circle), dim*np.sin(unit_circle), line_dash=[4, 4], line_color="gray")


	sorted_channel_names = []
	for i in range(len(channel_names)):
		sorted_channel_names.append(channel_names[perm[i]])

	data = dict(x = rads*np.cos(angs),
				y = rads*np.sin(angs),
				desc=sorted_channel_names)

	src = ColumnDataSource(data=data)

	# Plot phase diagram
	p1.line('x','y', color='#A6CEE3',  line_width=2, source=src)
	g1=p1.circle('x','y', size=10, fill_color=None, line_color="olivedrab", source = src)

	p1.legend.location = "top_left"
	p1.xaxis.bounds = (-1, 1)
	p1.yaxis.bounds = (-1, 1)

	hover = HoverTool(
				renderers=[g1],
				tooltips=[
					("index", "$index"),
					("(x,y)", "($x, $y)"),
					("desc", "@desc"),
				]
	)

	p1.add_tools(hover)


	source = ColumnDataSource(data=dict(x=np.cos(angs), y= np.sin(angs), names=np.array(perm,dtype=np.str)))

	labels = LabelSet(x='x', y='y', text='names', level='glyph', x_offset=5, y_offset=5, source=source, render_mode='canvas')
	p1.add_layout(labels)

	p2 = figure(title="Eigenvalues (magnitude)",plot_width = 320, plot_height=320)
	p2.grid.grid_line_alpha = 0.3
	p2.xaxis.axis_label = 'Index'
	p2.yaxis.axis_label = 'Absolute Value'

	xs = np.arange(len(evals))
	p2.line(xs, np.abs(evals),color='#A6CEE3', line_width=2)
	p2.circle(xs, np.abs(evals), color='red', size=10, fill_color=None)


	unitscale = lambda t: (t-t.min())/(t.max()-t.min())
	normlm = unitscale(abs(lm))

	xname = []
	yname = []
	cxname = []
	cyname = []
	color = []
	alpha = []
	N = lm.shape[0]

	# abr = Abbreviate()
	# chno = [abr.abbreviate(elem,target_len=10) for i, elem in enumerate(channel_names)]
	chno = [elem[:4]+elem[-2:] for i, elem in enumerate(channel_names)]
    # chno = [elem[:4]+elem[4::3] for i, elem in enumerate(channel_names)]

    # Fixed problem of limiting to ten channels with line below - June 2018
	for i in range(N):
		for j in range(N):
			xname.append(chno[i])
			yname.append(chno[j])
			cxname.append(channel_names[j])
			cyname.append(channel_names[i])
			alpha.append(min(normlm[i,j], 0.9) + 0.1)

			if (lm[i,j]<=0):
				color.append('red')
			else:
				color.append('blue')

	source = ColumnDataSource(data=dict(
		xname=xname,
		yname=yname,
		cxname=cxname,
		cyname=cyname,
		colors=color,
		alphas=alpha,
		lm=lm.flatten(),
		))


	p3 = figure(title = 'Matrix footprint',plot_width = 640, plot_height=640,
			x_axis_location="above", tools="hover,save",
           x_range=chno, y_range=list(reversed(chno)))

	p3.select_one(HoverTool).tooltips = [
		('Channels:', '@cyname, @cxname'),
		('Lead Matrix', '@lm'),
	]

	p3.grid.grid_line_color = None
	p3.axis.axis_line_color = None
	p3.axis.major_tick_line_color = None
	p3.axis.major_label_text_font_size = "8pt"
	p3.axis.major_label_standoff = 0
	p3.xaxis.major_label_orientation = np.pi/3

	p3.rect('xname', 'yname', 0.9, 0.9, source=source,
      color='colors', alpha='alphas', line_color=None,
      hover_line_color='black', hover_color='colors')


	lout = layout([[p1,p2]])

	return lout, p3
