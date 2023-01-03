import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
import pandas as pd
import math
class fuzzy_inference:
    def __init__(self, config):
        self.name = config['name']

    def speed(self, step=0.001):
        self.Fs = []
        self.speed_range = []
        self.speed_range_sl = []
        self.speed_range_dt = []
        self.speed_range_rv = []
        self.Fh = []
        self.Fm = []
        for i in np.arange(0.0, 1.001, step):
            self.speed_range.append(i)
            self.Fs.append((1-i)**4)
            self.Fh.append((1-(1-(i)**2)**2))
            self.Fm.append(-((((2*i)**2)-1)**2)+1 if -((((2*i)**2)-1)**2)+1>=0 else 0)
        plt.plot(self.speed_range, self.Fs, 'k--')
        plt.plot(self.speed_range, self.Fh, 'k--')
        plt.plot(self.speed_range, self.Fm, 'k--')
        plt.fill_between(self.speed_range, self.Fs, color='#539ecd', alpha=0.5)
        plt.fill_between(self.speed_range, self.Fh, color='#FBB6A8', alpha=0.5)
        plt.fill_between(self.speed_range, self.Fm, color='#B4F1B6', alpha=0.5)
        plt.grid()
        plt.xlabel('speed')
        plt.ylabel('membership value')
        plt.show()
        print('done!')

    def slippery(self, step=0.001):
        self.Fsl = []
        self.Fsm = []
        self.Fsh = []

        for i in np.arange(0.0, 1.001, step):
            self.speed_range_sl.append(i)
            self.Fsl.append(math.cos(3*i) if math.cos(3*i) >= 0 else 0)
            self.Fsm.append(((-3)*abs(i-0.5))+1 if ((-3)*abs(i-0.5))+1 >= 0 else 0)
            self.Fsh.append(i**3)
        plt.plot(self.speed_range_sl, self.Fsl, 'k--')
        plt.plot(self.speed_range_sl, self.Fsh, 'k--')
        plt.plot(self.speed_range_sl, self.Fsm, 'k--')
        plt.fill_between(self.speed_range_sl, self.Fsl, color='#539ecd', alpha=0.5)
        plt.fill_between(self.speed_range_sl, self.Fsh, color='#FBB6A8', alpha=0.5)
        plt.fill_between(self.speed_range_sl, self.Fsm, color='#B4F1B6', alpha=0.5)
        plt.grid()
        plt.xlabel('slippery level')
        plt.ylabel('membership value')
        plt.show()
        print('done!')
    def distance(self, step=0.001):
        self.Ffa = []
        self.Fmd = []
        self.Fc = []

        for i in np.arange(0.0, 1.001, step):
            self.speed_range_dt.append(i)
            self.Ffa.append((math.cos((3*i)-3)) if (math.cos((3*i)-3)) >= 0 else 0)
            self.Fmd.append((math.sin(5*i-1.2)) if (math.sin(5*i-1.2)) >= 0 else 0)
            self.Fc.append(-((2*i-0.1)**3)+1 if -((2*i-0.1)**3)+1 >= 0 else 0)
        plt.plot(self.speed_range_dt, self.Ffa, 'k--')
        plt.plot(self.speed_range_dt, self.Fmd, 'k--')
        plt.plot(self.speed_range_dt, self.Fc, 'k--')
        plt.fill_between(self.speed_range_dt, self.Ffa, color='#539ecd', alpha=0.5)
        plt.fill_between(self.speed_range_dt, self.Fmd, color='#FBB6A8', alpha=0.5)
        plt.fill_between(self.speed_range_dt, self.Fc, color='#B4F1B6', alpha=0.5)
        plt.grid()
        plt.xlabel('distance level')
        plt.ylabel('membership value')
        plt.show()
        print('done!')
    def risk(self, step=0.001):
        self.Frl = []
        self.Frm = []
        self.Frh = []

        for i in np.arange(0.0, 1.001, step):
            self.speed_range_rv.append(i)
            self.Frl.append((-abs(10*i-1)+1) if (-abs(10*i-1)+1) >= 0 else 0)
            self.Frm.append((-((5*i-2)**2)+1) if (-((5*i-2)**2)+1) >= 0 else 0)
            self.Frh.append((-abs(4*i-3)+1) if (-abs(4*i-3)+1) >= 0 else 0)
        plt.plot(self.speed_range_rv, self.Frl, 'k--')
        plt.plot(self.speed_range_rv, self.Frm, 'k--')
        plt.plot(self.speed_range_rv, self.Frh, 'k--')
        plt.fill_between(self.speed_range_rv, self.Frl, color='#539ecd', alpha=0.5)
        plt.fill_between(self.speed_range_rv, self.Frm, color='#FBB6A8', alpha=0.5)
        plt.fill_between(self.speed_range_rv, self.Frh, color='#B4F1B6', alpha=0.5)
        plt.grid()
        plt.xlabel('risk level')
        plt.ylabel('membership value')
        plt.show()
        print('done!')
    def create_inference_space(self):
        self.Inferencespace={'Range': self.speed_range,
                             'speed_low': self.Fs,
                             'speed_medium': self.Fm,
                             'speed_high': self.Fh,
                             'slippery_low': self.Fsl,
                             'slippery_medium': self.Fsm,
                             'slippery_high': self.Fsh,
                             'distance_close': self.Fc,
                             'distance_medium': self.Fmd,
                             'distance_far': self.Ffa,
                             'risk_low': self.Frl,
                             'risk_medium': self.Frm,
                             'risk_high': self.Frh}
        self.inference_df = pd.DataFrame(self.Inferencespace)
        self.inference_df.to_csv('./exp/Inference_space.csv', index=False)


    def membership_finder(self, input):   #speed,distance,slippery
        self.speed_fuzzy_space = self.inference_df.loc[self.inference_df['Range'] == input[0]]
        self.speed_Mu = self.speed_fuzzy_space[['slippery_low', 'slippery_medium', 'slippery_high']].values.flatten().tolist()
        self.distnace_fuzzy_space = self.inference_df.loc[self.inference_df['Range'] == input[1]]
        self.distnace_Mu = self.distnace_fuzzy_space[['slippery_low', 'slippery_medium', 'slippery_high']].values.flatten().tolist()
        self.slippery_fuzzy_space = self.inference_df.loc[self.inference_df['Range'] == input[2]]
        self.slippery_Mu = self.distnace_fuzzy_space[['distance_close', 'distance_medium', 'distance_far']].values.flatten().tolist()
        print('speed_MU: {}\nDistance: {}\nSlippery: {}'.format(self.speed_Mu,self.distnace_Mu,self.slippery_Mu))

    def risk_calculator(self):
        self.rules()
        self.inference_df['risk_low'].where(self.inference_df['risk_low'] <= max(self.lowrisk),
                                            max(self.lowrisk), inplace=True)
        self.inference_df['risk_medium'].where(self.inference_df['risk_medium'] <= max(self.mediumrisk),
                                               max(self.mediumrisk), inplace=True)
        self.inference_df['risk_high'].where(self.inference_df['risk_high'] <= max(self.highrisk),
                                             max(self.highrisk), inplace=True)
        self.inference_area= self.inference_df[['risk_low','risk_medium','risk_high']].max(axis=1)
        a = 0
        Mu =0
        for i in range(len(self.speed_range_rv)):
            a += self.speed_range_rv[i]*self.inference_area[i]
            Mu += self.inference_area[i]
        self.RoG = a/Mu


        # plt.plot(self.speed_range_rv, self.inference_area, 'k--')
        # plt.fill_between(self.speed_range_rv, self.inference_area, color='#539ecd', alpha=0.5)
        # plt.grid()
        # plt.xlabel('risk level')
        # plt.ylabel('membership value')
        # plt.ylim([0,1])
        # plt.show()
        print('done!')
        print('ROG: {}'.format(self.RoG))

    def rules(self):
        self.highrisk = []
        self.mediumrisk = []
        self.lowrisk = []
        self.highrisk.append(max(self.speed_Mu[2], self.slippery_Mu[2], self.distnace_Mu[2]))
        self.mediumrisk.append(max(self.speed_Mu[0], self.slippery_Mu[2], self.distnace_Mu[1]))
        self.mediumrisk.append(max(self.speed_Mu[0], self.slippery_Mu[1], self.distnace_Mu[2]))
        self.lowrisk.append(max(self.speed_Mu[1], self.slippery_Mu[0], self.distnace_Mu[0]))

    def surface(self):
        self.speed(step=0.1)
        self.slippery(step=0.1)
        self.distance(step=0.1)
        self.risk(step=0.1)
        self.a_input = self.b_input = self.c_input = self.speed_range
        self.create_inference_space()
        self.surface_dct = {'speed':[] , 'slippery':[], 'distance':[], 'risk':[]}
        for i in self.a_input:
            for j in self.b_input:
                for k in self.c_input:
                    self.surface_dct['speed'].append(i)
                    self.surface_dct['distance'].append(j)
                    self.surface_dct['slippery'].append(k)

                    self.create_inference_space()
                    self.membership_finder([i, j, k])
                    self.risk_calculator()
                    self.surface_dct['risk'].append(self.RoG)

        self.surface_df = pd.DataFrame(self.surface_dct)
        self.surface_df.to_csv('./exp/surface_Inference_space.csv', index=False)
        self.plot_surface()
        print('surface Done!')
    def plot_surface(self):
        self.modes={'speed-distance':['speed', 'distance'],
                    'speed-slippery':['speed', 'slippery'],
                    'distance-slippery':['distance', 'slippery']}
        for i in self.modes.keys():
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            ax.plot_trisurf(self.surface_df[self.modes[i][0]],
                            self.surface_df[self.modes[i][1]],
                            self.surface_df.risk, cmap=cm.jet, linewidth=0.2)
            plt.xlabel(self.modes[i][0])
            plt.ylabel(self.modes[i][1])
            ax.set_zlabel('risk')
            plt.show()









