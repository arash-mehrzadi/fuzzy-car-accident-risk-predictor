import src.Funcs as fin

config = {'name':'Arash'}

fuzzy_clc = fin.fuzzy_inference(config)
#create speed membership functions
fuzzy_clc.speed()
#create slippery membership functions
fuzzy_clc.slippery()
#create distance membership functions
fuzzy_clc.distance()
#create risk membership functions
fuzzy_clc.risk()
#create inference space
fuzzy_clc.create_inference_space()
#find membership values according to each input
fuzzy_clc.membership_finder([0,0.4,0])
#Calculate risk
fuzzy_clc.risk_calculator()
#create surfaces
fuzzy_clc.surface()
