#!/usr/bin/env python3

import os
import sys
import inspect

# Needed for using packages in child directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from Dataloader import Param, M_freq, Aggr_style, YearObj, DataObj
import numpy as np


param_names = {'g10m':(49,[Aggr_style.MAX]),
               'swvl1':(39, [Aggr_style.MIN, Aggr_style.MAX]),
               'slhf':(147,[Aggr_style.MIN, Aggr_style.MAX, Aggr_style.AVG]),
               'sshf':(146,[Aggr_style.MIN, Aggr_style.MAX, Aggr_style.AVG]),
               'ssrd':(169,[Aggr_style.MIN, Aggr_style.MAX, Aggr_style.AVG]),
               'u10m':(165,[Aggr_style.AVG]),
               'v10m':(166,[Aggr_style.AVG]),
               'blh':(159,[Aggr_style.MAX]),
               'd2m':(168,[Aggr_style.MIN, Aggr_style.MAX, Aggr_style.AVG]),
               'lsp':(142,[Aggr_style.SUM]),
               'skt':(235,[Aggr_style.MIN, Aggr_style.MAX, Aggr_style.AVG]),
               'src':(198,[Aggr_style.MIN, Aggr_style.AVG]),
               'ssr':(176,[Aggr_style.MAX, Aggr_style.AVG]),
               't2m':(167,[Aggr_style.MIN, Aggr_style.MAX, Aggr_style.AVG]),
               'ci':(31,[Aggr_style.AVG]),
               'cp':(143,[Aggr_style.MAX, Aggr_style.SUM]),
               'sd':(141,[Aggr_style.MIN, Aggr_style.MAX]),
               'sf':(144,[Aggr_style.AVG, Aggr_style.MAX])}

print('Starting script')

params = [[Param(p_name, param_id=param_details[0], aggr_style=aggr_style) for aggr_style in param_details[1]]for p_name, param_details in param_names.items()]
# p1 = Param('lsp', param_id=142, aggr_style=Aggr_style.SUM)
# p2 = Param('t2m', param_id=167, aggr_style=Aggr_style.AVG)
# p3 = Param('swvl1', param_id=39, aggr_style=Aggr_style.AVG)

# make a 1-d list
params = sum(params, [])

full_df = DataObj(params, m_freq=M_freq.week)

full_df.load_years(range(2000, 2021))

full_df.dump_ds()