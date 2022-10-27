import os

from flask import render_template, flash, request, redirect, url_for, Response
from . import app
from werkzeug.utils import secure_filename

from flask_wtf import Form
from wtforms import StringField, SelectField, FileField
from wtforms.validators import InputRequired

import numpy as np
from bokeh.plotting import figure
from bokeh.resources import CDN
from bokeh.embed import components
from .cyclic_plots import plot_signals, plot_results
from .cyclic_analysis import cyclic_analysis, quad_norm, tv_norm, remove_bias, remove_linear

import csv

#from flask.ext.uploads import UploadSet, DATA

norms = {'none': ('Leave Intact',lambda t:t),
        'sqr':('Unit Squares', quad_norm),
        'tv':('Unit Total Variation', tv_norm)}

trend_removals = {'none': ('None', lambda t:t),
                  'bias': ('Remove Bias', remove_bias),
                  'linear': ('Remove Linear Trend', remove_linear)}

class UploadForm(Form):
    file = FileField(label='Select CSV File for Upload',validators = [InputRequired()])
    norm_type = SelectField(label='Normalization type',  choices=[(a,norms[a][0]) for a in norms])
    trend_removal = SelectField(label='Trend Removal',  choices=[(a,trend_removals[a][0]) for a in trend_removals])


UPLOAD_FOLDER = '/tmp/'
ALLOWED_EXTENSIONS = set(['csv', 'txt', 'mat', 'npy', 'npz'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'Shukey'
#data = UploadSet('data', DATA)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def validate_file(file, col_header):
    #dialect = csv.Sniffer().sniff(file.readline(), [',',';'])
    #csvfile.seek(0)
    #data = csv.reader(file, dialect)

    if col_header:  # Each column is a channel, and the first row has names
        try:
            text = np.genfromtxt(file.stream,delimiter=',',names=True,filling_values=0)
            A = np.array([list(l) for l in text]).T
            channel_names = list(text.dtype.names)
        except (RuntimeError, TypeError, NameError):
            return False, "File decoding failed",[' ']


    else:# A simple file...
        try:
            A = np.genfromtxt(file.stream,delimiter=',',filling_values=0)
        except (RuntimeError, TypeError, NameError):
            return False, "File decoding failed",[' ']

        if (A.shape[0]>A.shape[1]):
            A = A.T # Transpose if needed
        channel_names = ['channel' +str(k).zfill(2) for k in range(A.shape[0])]  #Added zfill for pad 0 in front - Jun 2018

    if A.shape[0]>30:
        return False, "Too many channels", ' '
    elif A.shape[1]<30:
        return False, "Not enough time samples", ' '

    return True, A, channel_names


def gen_report(A, channel_names, norm_type, tr_removal):
    ret ,normed_data= cyclic_analysis(A,p=1,normalize = norms[norm_type][1], trend_removal = trend_removals[tr_removal][1])
    lm, phases, perm, sorted_lm, evals = ret

    LM_str = " \n".join([",".join(map('{0:.2f}'.format, line)) for line in lm])
    perm_str = " ,".join(map('{0:d}'.format, perm))

    text = "# Phase order \n" + perm_str +"\n#Lead matrix \n"+LM_str
    return Response(
        text,
        mimetype="text/txt",
        headers={"Content-disposition":
                 "attachment; filename=report.txt"})

def plot_page(A, channel_names, norm_type, tr_removal):
    #rep_file = open(,'w+')

    ret ,normed_data= cyclic_analysis(A,p=1,normalize = norms[norm_type][1], trend_removal = trend_removals[tr_removal][1])
    lm, phases, perm, sorted_lm, evals = ret

    p_signal = plot_signals(normed_data, channel_names, phases, perm, lm, evals)
    p_phase, p_results = plot_results(A, channel_names, phases, perm, lm, evals)

    script_signal, div_signal = components(p_signal, CDN)
    script_results, div_results = components(p_results, CDN)
    script_phase, div_phase = components(p_phase, CDN)

    # Fixed lead matrix overflow when displaying as tex - June 2018
    if lm.shape[0] > 10:
        LM_str = '\\textrm{Lead matrix is too big to be displayed as text. Try downloading report.}'
    else:
        LM_str = " \\\\\n".join([" & ".join(map('{0:.2f}'.format, line)) for line in lm])

    eig_str = "  & ".join(map('{0:.3f}'.format, evals))
    perm_str = "  & ".join(map('{0:d}'.format, perm))

    js_resources = CDN.render_js()
    css_resources = CDN.render_css()

    data = {"analysis_script": script_results, "analysis_div": div_results,
                "data_script": script_signal, "data_div": div_signal,
                "phase_script": script_phase, "phase_div": div_phase,
                'norm_type': norms[norm_type][0],
                'matrix_output': LM_str,
                'eig_output': eig_str,
                'perm_output': perm_str,
                'channel_perm':[channel_names[p] for p in perm],}

    html = render_template('plots.html',data=data,js_resources=js_resources,css_resources=css_resources)
    return html


@app.route('/')
@app.route('/index')
def index():
    return redirect("/upload")

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    form = UploadForm()
    form.modal = " "
    error = None
    # <p><input type=file name=file class="mybutton">

    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files[form.file.name]
        print(request)
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            col_header = request.form.getlist('colheader')==['True']

            res,A,channel_names = validate_file(file,col_header)

            if not res:
                error=A
                flash("Error:")
                return render_template('upform.html',form=form, error=error)
            else:
                norm_type = request.form.get('norm_type')
                tr_removal = request.form.get('trend_removal')
                if 'analyze' in request.form:
                    return plot_page(A,channel_names,norm_type, tr_removal)
                elif 'report' in request.form:
                    return gen_report(A,channel_names,norm_type, tr_removal)

    return render_template('upform.html',form=form)
