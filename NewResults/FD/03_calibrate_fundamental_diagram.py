# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.metrics import mean_squared_error, r2_score
plt.rcParams.update({'figure.max_open_warning': 0})
plt.rc('font',family='Times New Roman')
plt.rcParams['mathtext.fontset']='stix'


class fundamental_diagram_model():
    
    def __init__(self, observed_flow, observed_density, observed_speed):
        self.observed_flow = observed_flow
        self.observed_density = observed_density
        self.observed_speed = observed_speed
    
    def S3(self, beta):
        vf, kc, foc = beta
        estimated_speed = vf/np.power(1 + np.power((self.observed_density/kc), foc), 2/foc)
        f_obj = np.mean(np.power(estimated_speed - self.observed_speed, 2))
        return f_obj
    
    def OVM(self, beta):
        vf, veh_length, form_factor, transition_width = beta
        estimated_speed = vf*(np.tanh((1-self.observed_density*veh_length)/(self.observed_density*transition_width))+np.tanh(form_factor))/(1+np.tanh(form_factor))
        f_obj = np.mean(np.power(estimated_speed - self.observed_speed, 2))
        return f_obj
    
    def METANET(self, beta):
        vf, kc, m = beta
        estimated_speed = vf*np.exp(-1/m*np.power(self.observed_density/kc, m))
        f_obj = np.mean(np.power(estimated_speed - self.observed_speed, 2))
        return f_obj
    
    def MayKeller(self, beta):
        vf, kj, m, n = beta
        estimated_speed = vf*np.power(1 - np.power(self.observed_density/kj, m), n)
        f_obj = np.mean(np.power(estimated_speed - self.observed_speed, 2))
        return f_obj

class first_order_derivative():
    
    def __init__(self, observed_flow, observed_density, observed_speed):
        self.observed_flow = observed_flow
        self.observed_density = observed_density
        self.observed_speed = observed_speed
        
    def S3(self, beta):
        vf, kc, foc = beta
        intermediate_variable = np.power(self.observed_density/kc, foc)
        first_order_derivative_1 = 2*np.mean((vf/np.power(1 + intermediate_variable, 2/foc) - self.observed_speed) / np.power(1 + intermediate_variable, 2/foc))
        first_order_derivative_2 = 2*np.mean((vf/np.power(1 + intermediate_variable, 2/foc) - self.observed_speed) * 2 * vf * intermediate_variable / kc / np.power(1 + intermediate_variable, (foc+2)/foc))
        first_order_derivative_3 = 2*np.mean((vf/np.power(1 + intermediate_variable, 2/foc) - self.observed_speed) * 2 * vf * ((1 + intermediate_variable)*np.log(1 + intermediate_variable) - foc * intermediate_variable * np.log(intermediate_variable)) / np.power(foc, 2) / np.power(1 + intermediate_variable, (foc+2)/foc))
        first_order_derivative = np.asarray([first_order_derivative_1, first_order_derivative_2, first_order_derivative_3])
        return first_order_derivative
    
    def OVM(self, beta):
        vf, veh_length, form_factor, transition_width = beta
        intermediate_variable = (1-self.observed_density*veh_length)/(self.observed_density*transition_width)
        first_order_derivative_1 = 2*np.mean((vf*(np.tanh((1-self.observed_density*veh_length)/(self.observed_density*transition_width))+np.tanh(form_factor))/(1+np.tanh(form_factor)) - self.observed_speed) * (np.tanh(intermediate_variable) + np.tanh(form_factor))/(1+np.tanh(form_factor)))
        first_order_derivative_2 = 2*np.mean((vf*(np.tanh((1-self.observed_density*veh_length)/(self.observed_density*transition_width))+np.tanh(form_factor))/(1+np.tanh(form_factor)) - self.observed_speed) * (-1) * vf * np.power(1/np.cosh(intermediate_variable), 2)/(transition_width*np.tanh(form_factor) + transition_width))
        first_order_derivative_3 = 2*np.mean((vf*(np.tanh((1-self.observed_density*veh_length)/(self.observed_density*transition_width))+np.tanh(form_factor))/(1+np.tanh(form_factor)) - self.observed_speed) * vf * (self.observed_density*veh_length-1)*np.power(1/np.cosh(intermediate_variable), 2)/np.power(self.observed_density, 2)/np.power(transition_width, 2)/(1+np.tanh(form_factor)))
        first_order_derivative_4 = 2*np.mean((vf*(np.tanh((1-self.observed_density*veh_length)/(self.observed_density*transition_width))+np.tanh(form_factor))/(1+np.tanh(form_factor)) - self.observed_speed) * (-1) * vf * np.power(1/np.cosh(form_factor), 2) * (np.tanh(intermediate_variable) - 1)/np.power(1 + np.tanh(form_factor), 2))
        first_order_derivative = np.asarray([first_order_derivative_1, first_order_derivative_2, first_order_derivative_3, first_order_derivative_4])
        return first_order_derivative
    
    def METANET(self, beta):
        vf, kc, m = beta
        intermediate_variable = np.exp(-1/m*np.power(self.observed_density/kc, m))
        first_order_derivative_1 = 2*np.mean((vf*intermediate_variable - self.observed_speed)*intermediate_variable)
        first_order_derivative_2 = 2*np.mean((vf*intermediate_variable - self.observed_speed)*(vf*intermediate_variable*np.power(self.observed_density/kc, m)/kc))
        first_order_derivative_3 = 2*np.mean((vf*intermediate_variable - self.observed_speed)*(vf*intermediate_variable*np.power(self.observed_density/kc, m)*(m*np.log(self.observed_density/kc)-1)/m/m))
        first_order_derivative = np.asarray([first_order_derivative_1, first_order_derivative_2, first_order_derivative_3])
        return first_order_derivative
    
    def MayKeller(self, beta):
        vf, kj, m, n = beta
        intermediate_variable = np.power(self.observed_density/kj, m)
        first_order_derivative_1 = 2*np.mean((vf*np.power(1-intermediate_variable,n) - self.observed_speed)*(np.power(1-intermediate_variable,n)))
        first_order_derivative_2 = 2*np.mean((vf*np.power(1-intermediate_variable,n) - self.observed_speed)*(m*n*vf*intermediate_variable*np.power(1-intermediate_variable,n-1)/kj))
        first_order_derivative_3 = 2*np.mean((vf*np.power(1-intermediate_variable,n) - self.observed_speed)*(-n*vf*intermediate_variable*np.log(self.observed_density/kj)*np.power(1-intermediate_variable,n-1)))
        first_order_derivative_4 = 2*np.mean((vf*np.power(1-intermediate_variable,n) - self.observed_speed)*(vf*np.power(1-intermediate_variable,n)*np.log(1-intermediate_variable)))
        first_order_derivative = np.asarray([first_order_derivative_1, first_order_derivative_2, first_order_derivative_3, first_order_derivative_4])
        return first_order_derivative

class estimated_value():
    
    def __init__(self, observed_flow, observed_density, observed_speed):
        self.observed_flow = observed_flow
        self.observed_density = observed_density
        self.observed_speed = observed_speed
        
    def S3(self, beta):
        vf, kc, foc = beta
        estimated_speed = vf/np.power(1 + np.power((self.observed_density/kc), foc), 2/foc)
        estimated_flow = self.observed_density*estimated_speed
        return estimated_speed, estimated_flow
    
    def OVM(self, beta):
        vf, veh_length, form_factor, transition_width = beta
        estimated_speed = vf*(np.tanh((1-self.observed_density*veh_length)/(self.observed_density*transition_width))+np.tanh(form_factor))/(1+np.tanh(form_factor))
        estimated_flow = self.observed_density*estimated_speed
        return estimated_speed, estimated_flow
    
    def METANET(self, beta):
        vf, kc, m = beta
        estimated_speed = vf*np.exp(-1/m*np.power(self.observed_density/kc, m))
        estimated_flow = self.observed_density*estimated_speed
        return estimated_speed, estimated_flow
    
    def MayKeller(self, beta):
        vf, kj, m, n = beta
        estimated_speed = vf*np.power(1 - np.power(self.observed_density/kj, m), n)
        estimated_flow = self.observed_density*estimated_speed
        return estimated_speed, estimated_flow

class theoretical_value():
    
    def __init__(self, density, speed):
        self.density = density
        self.speed = speed
        
    def S3(self, beta):
        vf, kc, foc = beta
        theoretical_speed = vf/np.power(1 + np.power((self.density/kc), foc), 2/foc)
        theoretical_flow = self.density*theoretical_speed
        return theoretical_speed, theoretical_flow
    
    def OVM(self, beta):
        vf, veh_length, form_factor, transition_width = beta
        theoretical_speed = vf*(np.tanh((1-self.density*veh_length)/(self.density*transition_width))+np.tanh(form_factor))/(1+np.tanh(form_factor))
        theoretical_flow = self.density*theoretical_speed
        return theoretical_speed, theoretical_flow
    
    def IDM(self, beta):
        delta = 4
        vf, g0, T0, L = beta
        theoretical_density = 1000*np.sqrt(1-np.power(self.speed/vf, delta))/(g0+self.speed*T0/3.6+L*np.sqrt(1-np.power(self.speed/vf, delta)))
        theoretical_flow = theoretical_density*self.speed
        return theoretical_density, theoretical_flow
    
    def METANET(self, beta):
        vf, kc, m = beta
        theoretical_speed = vf*np.exp(-1/m*np.power(self.density/kc, m))
        theoretical_flow = self.density*theoretical_speed
        return theoretical_speed, theoretical_flow
    
    def MayKeller(self, beta):
        vf, kj, m, n = beta
        theoretical_speed = vf*np.power(1 - np.power(self.density/kj, m), n)
        theoretical_flow = self.density*theoretical_speed
        return theoretical_speed, theoretical_flow
    
class Adam_optimization():
    
    def __init__(self, objective, first_order_derivative, bounds, x0):
        self.n_iter = 2000
        self.alpha = 0.01
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        self.objective = objective
        self.first_order_derivative = first_order_derivative
        self.bounds = bounds
        self.x0 = x0
    
    def adam(self):
        # keep track of solutions and scores
        solutions = list()
        scores = list()
        # generate an initial point
        x = list(self.x0)
        score = self.objective(x)
        # initialize first and second moments
        m = [0.0 for _ in range(self.bounds.shape[0])]
        v = [0.0 for _ in range(self.bounds.shape[0])]
        # run the gradient descent updates
        for t in range(1, self.n_iter):
            # calculate gradient g(t)
            g = self.first_order_derivative(x)
            # build a solution one variable at a time
            for i in range(self.bounds.shape[0]):
                # m(t) = beta1 * m(t-1) + (1 - beta1) * g(t)
                m[i] = self.beta1 * m[i] + (1.0 - self.beta1) * g[i]
                # v(t) = beta2 * v(t-1) + (1 - beta2) * g(t)^2
                v[i] = self.beta2 * v[i] + (1.0 - self.beta2) * g[i]**2
                # mhat(t) = m(t) / (1 - beta1(t))
                mhat = m[i] / (1.0 - self.beta1**(t+1))
                # vhat(t) = v(t) / (1 - beta2(t))
                vhat = v[i] / (1.0 - self.beta2**(t+1))
                # x(t) = x(t-1) - alpha * mhat(t) / (sqrt(vhat(t)) + eps)
                x[i] = x[i] - self.alpha * mhat / (sqrt(vhat) + self.eps)
            # evaluate candidate point
            score = self.objective(x)
            # keep track of solutions and scores
            solutions.append(x.copy())
            scores.append(score)
            # report progress
        # print('Solution: %s, \nOptimal function value: %.5f' %(solutions[np.argmin(scores)], min(scores)))
        return solutions, scores
    
    def plot_iteration_process_adam(self, solutions):
        # sample input range uniformly at 0.1 increments
        xaxis = np.arange(self.bounds[0,0], self.bounds[0,1], 0.1)
        yaxis = np.arange(self.bounds[1,0], self.bounds[1,1], 0.1)
        x, y = np.meshgrid(xaxis, yaxis)
        results = self.objective(x, y)
        solutions = np.asarray(solutions)
        fig, ax = plt.subplots(figsize=(10,6))
        cs = ax.contourf(x, y, results, levels=50, cmap='jet')
        ax.set_xlim(self.bounds[0,0], self.bounds[0,1])
        ax.set_ylim(self.bounds[1,0], self.bounds[1,1])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.xaxis.label.set_size(18)
        ax.yaxis.label.set_size(18)
        plt.tick_params(labelsize=14)
        plt.plot(solutions[:, 0], solutions[:, 1], '.-', color='k')
        plt.colorbar(cs)
        plt.title('Iteration process')
        fig.savefig('../Figures/Case 1/Iteration process.png', dpi=300, bbox_inches='tight')

class plot_calibration_results():
    
    def __init__(self, observed_flow, observed_density, observed_speed, calibrated_paras):
        
        self.observed_flow = observed_flow
        self.observed_density = observed_density
        self.observed_speed = observed_speed
        self.calibrated_paras_S3 = calibrated_paras["S3"]      # Calibrated from fundamental diagram model, vf, kc, foc
        self.calibrated_paras_OVM = calibrated_paras["OVM"]    # Calibrated from fundamental diagram model, vf, veh_length, form_factor, transition_width
        self.calibrated_paras_IDM = [75, 2, 1.7, 5]              # Calibrated from microscopic car-following model, vf, g0, T0, L
        self.calibrated_paras_METANET = calibrated_paras["METANET"]
        self.calibrated_paras_MayKeller = calibrated_paras["MayKeller"]
        self.k = np.linspace(0.000001,140,70)
        self.v = np.linspace(0.000001,self.calibrated_paras_IDM[0],70)
        self.theoretical_value = theoretical_value(self.k, self.v)
        self.theoretical_speed_S3, self.theoretical_flow_S3 = self.theoretical_value.S3(self.calibrated_paras_S3)
        self.theoretical_speed_OVM, self.theoretical_flow_OVM = self.theoretical_value.OVM(self.calibrated_paras_OVM)
        self.theoretical_density_IDM, self.theoretical_flow_IDM = self.theoretical_value.IDM(self.calibrated_paras_IDM)
        self.theoretical_speed_METANET, self.theoretical_flow_METANET = self.theoretical_value.METANET(self.calibrated_paras_METANET)
        self.theoretical_speed_MayKeller, self.theoretical_flow_MayKeller = self.theoretical_value.MayKeller(self.calibrated_paras_MayKeller)
    
    def plot_qk(self):
        
        fig = plt.figure(figsize=(7,5))
        plt.scatter(self.observed_density, self.observed_flow, s = 4, marker='o', c='r', edgecolors='r', label = 'Observation')
        plt.plot(self.k, self.theoretical_flow_S3, 'b-', linewidth=4, label = "S3")
        plt.plot(self.k, self.theoretical_flow_OVM, 'g--', linewidth=4, label = "OVM")
        plt.plot(self.theoretical_density_IDM, self.theoretical_flow_IDM, 'k-.', linewidth=4, label = "IDM")
        plt.plot(self.k, self.theoretical_flow_METANET, 'm:', linewidth=4, label = "METANET")
        plt.plot(self.k, self.theoretical_flow_MayKeller, 'kx:', linewidth=4, label = "MayKeller")
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel('Density (veh/km)', fontsize=16)
        plt.ylabel('Flow (veh/h)', fontsize=16)
        plt.xlim((0, 120))
        plt.ylim((0, 2100))
        plt.legend(loc='upper right', fontsize=14)
        plt.title('Flow vs. density', fontsize=20)
        fig.savefig('Figures\\flow vs density.png', dpi=400, bbox_inches='tight')
    
    def plot_vk(self):
        
        fig = plt.figure(figsize=(7,5))
        plt.scatter(self.observed_density, self.observed_speed, s = 4, marker='o', c='r', edgecolors='r', label = 'Observation')
        plt.plot(self.k, self.theoretical_speed_S3, 'b-', linewidth=4, label = "S3")
        plt.plot(self.k, self.theoretical_speed_OVM, 'g--', linewidth=4, label = "OVM")
        plt.plot(self.theoretical_density_IDM, self.v, 'k-.', linewidth=4, label = "IDM")
        plt.plot(self.k, self.theoretical_speed_METANET, 'm:', linewidth=4, label = "METANET")
        plt.plot(self.k, self.theoretical_speed_MayKeller, 'kx:', linewidth=4, label = "MayKeller")
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel('Density (veh/km)', fontsize=16)
        plt.ylabel('Speed (km/h)', fontsize=16)
        plt.xlim((5, 120))
        plt.ylim((0, 100))
        plt.legend(loc='upper right', fontsize=14)
        plt.title('Speed vs. density', fontsize=20)
        fig.savefig('Figures\\speed vs density.png', dpi=400, bbox_inches='tight')
        
    def plot_vq(self):
        
        fig = plt.figure(figsize=(7,5))
        plt.scatter(self.observed_flow, self.observed_speed, s = 4, marker='o', c='r', edgecolors='r', label = 'Observation')
        plt.plot(self.theoretical_flow_S3, self.theoretical_speed_S3, 'b-', linewidth=4, label = "S3")
        plt.plot(self.theoretical_flow_OVM, self.theoretical_speed_OVM, 'g--', linewidth=4, label = "OVM")
        plt.plot(self.theoretical_flow_IDM, self.v, 'k-.', linewidth=4, label = "IDM")
        plt.plot(self.theoretical_flow_METANET, self.theoretical_speed_METANET, 'm:', linewidth=4, label = "METANET")
        plt.plot(self.theoretical_flow_MayKeller, self.theoretical_speed_MayKeller, 'kx:', linewidth=4, label = "MayKeller")
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel('Flow (veh/h)', fontsize=16)
        plt.ylabel('Speed (km/h)', fontsize=16)
        plt.xlim((400,2100))
        plt.ylim((0, 100))
        plt.legend(loc='upper right', fontsize=14)
        plt.title('Speed vs. flow', fontsize=20)
        fig.savefig('Figures\\speed vs flow.png', dpi=400, bbox_inches='tight')

class getMetrics():
    
    def __init__(self, observed_flow, observed_density, observed_speed):
        self.observed_flow = observed_flow
        self.observed_density = observed_density
        self.observed_speed = observed_speed
        self.estimated_value = estimated_value(observed_flow, observed_density, observed_speed)
        
    def S3_RMSE_Overall(self, paras):
        estimated_speed, estimated_flow = self.estimated_value.S3(paras)
        rmse_speed = mean_squared_error(self.observed_speed, estimated_speed, squared=False)
        rmse_flow = mean_squared_error(self.observed_flow, estimated_flow, squared=False)
        r2_speed = r2_score(self.observed_speed, estimated_speed)
        r2_flow = r2_score(self.observed_flow, estimated_flow)
        return rmse_speed, rmse_flow, r2_speed, r2_flow
    
    def OVM_RMSE_Overall(self, paras):
        estimated_speed, estimated_flow = self.estimated_value.OVM(paras)
        rmse_speed = mean_squared_error(self.observed_speed, estimated_speed, squared=False)
        rmse_flow = mean_squared_error(self.observed_flow, estimated_flow, squared=False)
        r2_speed = r2_score(self.observed_speed, estimated_speed)
        r2_flow = r2_score(self.observed_flow, estimated_flow)
        return rmse_speed, rmse_flow, r2_speed, r2_flow
    
    def METANET_RMSE_Overall(self, paras):
        estimated_speed, estimated_flow = self.estimated_value.METANET(paras)
        rmse_speed = mean_squared_error(self.observed_speed, estimated_speed, squared=False)
        rmse_flow = mean_squared_error(self.observed_flow, estimated_flow, squared=False)
        r2_speed = r2_score(self.observed_speed, estimated_speed)
        r2_flow = r2_score(self.observed_flow, estimated_flow)
        return rmse_speed, rmse_flow, r2_speed, r2_flow
    
    def MayKeller_RMSE_Overall(self, paras):
        estimated_speed, estimated_flow = self.estimated_value.MayKeller(paras)
        rmse_speed = mean_squared_error(self.observed_speed, estimated_speed, squared=False)
        rmse_flow = mean_squared_error(self.observed_flow, estimated_flow, squared=False)
        r2_speed = r2_score(self.observed_speed, estimated_speed)
        r2_flow = r2_score(self.observed_flow, estimated_flow)
        return rmse_speed, rmse_flow, r2_speed, r2_flow
    
    def S3_RMSE_Small_Range(self, paras, interval=10):
        estimated_speed, estimated_flow = self.estimated_value.S3(paras)
        rmse_speed_small_range = []
        rmse_flow_small_range = []
        for i in range(0,10):
            temp_index = np.where((self.observed_density>=10*i) & (self.observed_density<10*(i+1)))
            observed_speed_i = self.observed_speed[temp_index]
            estimated_speed_i = estimated_speed[temp_index]
            observed_flow_i = self.observed_flow[temp_index]
            estimated_flow_i = estimated_flow[temp_index]
            rmse_speed_small_range.append(mean_squared_error(observed_speed_i, estimated_speed_i, squared=False))
            rmse_flow_small_range.append(mean_squared_error(observed_flow_i, estimated_flow_i, squared=False))
        observed_speed_last = self.observed_speed[np.where((self.observed_density>=100))]
        estimated_speed_last = estimated_speed[np.where((self.observed_density>=100))]
        observed_flow_last = self.observed_flow[np.where((self.observed_density>=100))]
        estimated_flow_last = estimated_flow[np.where((self.observed_density>=100))]
        rmse_speed_small_range.append(mean_squared_error(observed_speed_last, estimated_speed_last, squared=False))
        rmse_flow_small_range.append(mean_squared_error(observed_flow_last, estimated_flow_last, squared=False))
        return rmse_speed_small_range, rmse_flow_small_range
    
    def OVM_RMSE_Small_Range(self, paras, interval=10):
        estimated_speed, estimated_flow = self.estimated_value.OVM(paras)
        rmse_speed_small_range = []
        rmse_flow_small_range = []
        for i in range(0,10):
            temp_index = np.where((self.observed_density>=10*i) & (self.observed_density<10*(i+1)))
            observed_speed_i = self.observed_speed[temp_index]
            estimated_speed_i = estimated_speed[temp_index]
            observed_flow_i = self.observed_flow[temp_index]
            estimated_flow_i = estimated_flow[temp_index]
            rmse_speed_small_range.append(mean_squared_error(observed_speed_i, estimated_speed_i, squared=False))
            rmse_flow_small_range.append(mean_squared_error(observed_flow_i, estimated_flow_i, squared=False))
        observed_speed_last = self.observed_speed[np.where((self.observed_density>=100))]
        estimated_speed_last = estimated_speed[np.where((self.observed_density>=100))]
        observed_flow_last = self.observed_flow[np.where((self.observed_density>=100))]
        estimated_flow_last = estimated_flow[np.where((self.observed_density>=100))]
        rmse_speed_small_range.append(mean_squared_error(observed_speed_last, estimated_speed_last, squared=False))
        rmse_flow_small_range.append(mean_squared_error(observed_flow_last, estimated_flow_last, squared=False))
        return rmse_speed_small_range, rmse_flow_small_range
    
    def METANET_RMSE_Small_Range(self, paras, interval=10):
        estimated_speed, estimated_flow = self.estimated_value.METANET(paras)
        rmse_speed_small_range = []
        rmse_flow_small_range = []
        for i in range(0,10):
            temp_index = np.where((self.observed_density>=10*i) & (self.observed_density<10*(i+1)))
            observed_speed_i = self.observed_speed[temp_index]
            estimated_speed_i = estimated_speed[temp_index]
            observed_flow_i = self.observed_flow[temp_index]
            estimated_flow_i = estimated_flow[temp_index]
            rmse_speed_small_range.append(mean_squared_error(observed_speed_i, estimated_speed_i, squared=False))
            rmse_flow_small_range.append(mean_squared_error(observed_flow_i, estimated_flow_i, squared=False))
        observed_speed_last = self.observed_speed[np.where((self.observed_density>=100))]
        estimated_speed_last = estimated_speed[np.where((self.observed_density>=100))]
        observed_flow_last = self.observed_flow[np.where((self.observed_density>=100))]
        estimated_flow_last = estimated_flow[np.where((self.observed_density>=100))]
        rmse_speed_small_range.append(mean_squared_error(observed_speed_last, estimated_speed_last, squared=False))
        rmse_flow_small_range.append(mean_squared_error(observed_flow_last, estimated_flow_last, squared=False))
        return rmse_speed_small_range, rmse_flow_small_range
    
    def MayKeller_RMSE_Small_Range(self, paras, interval=10):
        estimated_speed, estimated_flow = self.estimated_value.MayKeller(paras)
        rmse_speed_small_range = []
        rmse_flow_small_range = []
        for i in range(0,10):
            temp_index = np.where((self.observed_density>=10*i) & (self.observed_density<10*(i+1)))
            observed_speed_i = self.observed_speed[temp_index]
            estimated_speed_i = estimated_speed[temp_index]
            observed_flow_i = self.observed_flow[temp_index]
            estimated_flow_i = estimated_flow[temp_index]
            rmse_speed_small_range.append(mean_squared_error(observed_speed_i, estimated_speed_i, squared=False))
            rmse_flow_small_range.append(mean_squared_error(observed_flow_i, estimated_flow_i, squared=False))
        observed_speed_last = self.observed_speed[np.where((self.observed_density>=100))]
        estimated_speed_last = estimated_speed[np.where((self.observed_density>=100))]
        observed_flow_last = self.observed_flow[np.where((self.observed_density>=100))]
        estimated_flow_last = estimated_flow[np.where((self.observed_density>=100))]
        rmse_speed_small_range.append(mean_squared_error(observed_speed_last, estimated_speed_last, squared=False))
        rmse_flow_small_range.append(mean_squared_error(observed_flow_last, estimated_flow_last, squared=False))
        return rmse_speed_small_range, rmse_flow_small_range

class calibrate():
    
    def __init__(self, flow, density, speed):
        self.flow = flow
        self.density = density
        self.speed = speed
        self.init_model_dict()
        
    def init_model_dict(self):
        self.model = fundamental_diagram_model(self.flow, self.density, self.speed)
        self.first_order_derivative = first_order_derivative(self.flow, self.density, self.speed)
        self.model_dict = {"S3":self.model.S3,
                           "OVM":self.model.OVM,
                           "METANET":self.model.METANET,
                           "MayKeller":self.model.MayKeller,
                           }
        self.derivative = {"S3": self.first_order_derivative.S3,
                           "OVM": self.first_order_derivative.OVM,
                           "METANET": self.first_order_derivative.METANET,
                           "MayKeller": self.first_order_derivative.MayKeller,
                           }
        self.bounds = {"S3": np.asarray([[70,80], [20,60], [1,8]]),     # vf, kc, foc
                       "OVM": np.asarray([[70,80], [0.003,0.007], [1,2], [0.010,0.030]]),    # vf, veh_length, form_factor, transition_width
                       "METANET": np.asarray([[70,80], [20,60], [1,8]]),     # vf, kc, m
                       "MayKeller": np.asarray([[70,80], [140,220], [0,8], [0,8]]),
                       }
        self.x0 = {"S3": np.asarray([75, 35, 3.6]),
                   "OVM": np.asarray([75, 0.005, 1.5, 0.015]),
                   "METANET": np.asarray([78, 36, 1.8]),
                   "MayKeller": np.asarray([75, 170, 2, 2]),
                   }
    
    def getSolution(self, model_str):
        # calibration
        objective = self.model_dict[model_str]
        derivative = self.derivative[model_str]
        bounds = self.bounds[model_str]
        x0 = self.x0[model_str]
        Adam = Adam_optimization(objective, derivative, bounds, x0)
        solutions, scores = Adam.adam()
        parameters = solutions[np.argmin(scores)]
        # obj = min(scores)
        return parameters


if __name__ == '__main__':
    
    # load q-k-v data
    filepath = './Data/'
    flow = np.array(pd.read_csv(filepath+'flow.csv', header=None)).flatten()
    density = np.array(pd.read_csv(filepath+'density.csv', header=None)).flatten()
    speed = np.array(pd.read_csv(filepath+'speed.csv', header=None)).flatten()
    
    solver = calibrate(flow, density, speed)
    result = {"S3":solver.getSolution("S3"),
              "OVM":solver.getSolution("OVM"),
              "METANET":solver.getSolution("METANET"),
              "MayKeller":solver.getSolution("MayKeller"),
              }
    
    plot_results = plot_calibration_results(flow, density, speed, result)
    plot_results.plot_qk(), plot_results.plot_vk(), plot_results.plot_vq()
    
    metrics = getMetrics(flow, density, speed)
    S3_RMSE_SPEED_Overall, S3_RMSE_FLOW_Overall, S3_R2_SPEED_Overall, S3_R2_FLOW_Overall = metrics.S3_RMSE_Overall(result["S3"])
    OVM_RMSE_SPEED_Overall, OVM_RMSE_FLOW_Overall, OVM_R2_SPEED_Overall, OVM_R2_FLOW_Overall = metrics.OVM_RMSE_Overall(result["OVM"])
    METANET_RMSE_SPEED_Overall, METANET_RMSE_FLOW_Overall, METANET_R2_SPEED_Overall, METANET_R2_FLOW_Overall = metrics.METANET_RMSE_Overall(result["METANET"])
    MayKeller_RMSE_SPEED_Overall, MayKeller_RMSE_FLOW_Overall, MayKeller_R2_SPEED_Overall, MayKeller_R2_FLOW_Overall = metrics.MayKeller_RMSE_Overall(result["MayKeller"])
    S3_RMSE_Small_Range = metrics.S3_RMSE_Small_Range(result["S3"])
    OVM_RMSE_Small_Range = metrics.OVM_RMSE_Small_Range(result["OVM"])
    METANET_RMSE_Small_Range = metrics.METANET_RMSE_Small_Range(result["METANET"])
    MayKeller_RMSE_Small_Range = metrics.MayKeller_RMSE_Small_Range(result["MayKeller"])
    
    