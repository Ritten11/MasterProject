#!/usr/bin/env python3

import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from Dataloader import Param, M_freq, Aggr_style, YearObj, DataObj
import numpy as np


param_names = ['g10m', 'swvl1', 'ewss', 'g10m', 'nsss', 'slhf', 'sshf', 'ssrd', 'strd', 'u10m', 'v10m', 'blh', 'd2m', 'lsp', 'skt', 'src', 'ssr', 'sst', 'str', 't2m', 'ci', 'cp', 'sd', 'sf']
print('Starting script')

params = [Param(p_name, param_id=i, aggr_style=Aggr_style.AVG) for i, p_name in enumerate(param_names)]
# p1 = Param('lsp', param_id=142, aggr_style=Aggr_style.s)
# p2 = Param('t2m', param_id=167, aggr_style=Aggr_style.a)
# p3 = Param('swvl1', param_id=39, aggr_style=Aggr_style.a)

full_df = DataObj(params, m_freq=M_freq.week)

full_df.check_data(range(2000,2021), True)
# full_df.dump()
print('\n\nFINISHED CHECKING --> COMPLETE DATASETS:')
incomplete = np.unique([obj.param.name for obj in full_df.incomplete_years])

complete = [p.name for p in params if p.name not in incomplete]

print(complete)

print('Done!')