# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: ExecuteTime,autoscroll,heading_collapsed,hidden,slideshow,title,tags,jupyter,pycharm,-hide_ouput,-code_folding
#     cell_metadata_json: true
#     formats: ipynb,py:percent
#     notebook_metadata_filter: all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.8.8
#   latex_envs:
#     LaTeX_envs_menu_present: true
#     autoclose: false
#     autocomplete: false
#     bibliofile: biblio.bib
#     cite_by: apalike
#     current_citInitial: 1
#     eqLabelWithNumbers: true
#     eqNumInitial: 1
#     hotkeys:
#       equation: Ctrl-E
#       itemize: Ctrl-I
#     labels_anchors: false
#     latex_user_defs: false
#     report_style_numbering: false
#     user_envs_cfg: false
# ---

# %% [markdown]
# # KohKW Second Year Project
#

# %% [markdown]
# <a id='interactive-dashboard'></a>
#
# [This notebook](https://econ-ark.org/BufferStockTheory/#launch) uses the [Econ-ARK/HARK](https://github.com/econ-ark/HARK) toolkit to reproduce and illustrate key results of the paper [Theoretical Foundations of Buffer Stock Saving](http://econ-ark.github.io/BufferStockTheory/BufferStockTheory).
#
# An [interactive dashboard](https://econ-ark.org/BufferStockStockTheory/#Dashboard) allows you to modify parameters to see how the figures change.
#
# - JupyterLab, click on the $\bullet$$\bullet$$\bullet$ patterns to expose the runnable code
# - in either a Jupyter notebook or JupyterLab, click a double triangle to execute the code and generate the figures

# %% [markdown]
# `# Setup Python Below`

# %% {"jupyter": {"source_hidden": true}, "tags": []}
# This cell does some setup

# Import required python packages
import warnings

# Ignore some harmless but alarming warning messages
warnings.filterwarnings("ignore")

# Plotting tools
import pandas as pd
import statsmodels.discrete.discrete_model as smdm
import statsmodels.api as sm
import numpy as np
from patsy import dmatrices
import os
filepath = os.path.abspath(os.getcwd())
filepathmain = os.path.dirname(os.path.dirname(filepath))


OwnData = pd.read_csv("RegA2.csv")
OwnData = OwnData.dropna(subset = ['sw_p', 'mov_past'])

y, X = dmatrices('sw_p ~ log_contributions_FIRE + bill_complexity + tight', data=OwnData, return_type='dataframe')
OLSmodel = sm.OLS(y,X)
results_0 = OLSmodel.fit()
results_0.summary()

OwnData['mov_contr_int'] = OwnData.apply(lambda row: row.mov_past * row.log_contributions_FIRE, axis = 1)

y, X = dmatrices('sw_p ~ log_contributions_FIRE + mov_past + mov_contr_int + bill_complexity + tight', data=OwnData, return_type='dataframe')
OLSmodel = sm.OLS(y,X)
results_1 = OLSmodel.fit()
results_1.summary()

OwnData['congru_contr_int'] = OwnData.apply(lambda row: row.congruence_dc * row.log_contributions_FIRE, axis = 1)

y, X = dmatrices('sw_p ~ log_contributions_FIRE + congruence_dc + congru_contr_int + bill_complexity + tight', data=OwnData, return_type='dataframe')
OLSmodel = sm.OLS(y,X)
results_2 = OLSmodel.fit()
results_2.summary()


with open('results_0.tex','w') as file:
	file.write(results_0.summary().as_latex())
os.replace(filepath + "\\results_0.tex", filepathmain + "\\Tables\\results_0.tex")

with open('results_1.tex','w') as file:
	file.write(results_1.summary().as_latex())
os.replace(filepath + "\\results_1.tex", filepathmain + "\\Tables\\results_1.tex")

with open('results_2.tex','w') as file:
	file.write(results_2.summary().as_latex())

os.replace(filepath + "\\results_2.tex", filepathmain + "\\Tables\\results_2.tex")
