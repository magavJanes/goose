# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 19:47:32 2021

@author: magav
"""
from kaggle_environments import make
env = make("hungry_geese", debug=True)

env.reset()
env.run(['submission.py', 'greedy'])
env.render(mode="ipython", width=800, height=700)