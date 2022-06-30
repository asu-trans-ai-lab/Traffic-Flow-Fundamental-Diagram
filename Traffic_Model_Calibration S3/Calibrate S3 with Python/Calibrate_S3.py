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

class theoretical_value():
    
    def __init__(self, density):
        self.density = density
        
    def S3(self, beta):
        vf, kc, foc = beta
        theoretical_speed = vf/np.power(1 + np.power((self.density/kc), foc), 2/foc)
        theoretical_flow = self.density*theoretical_speed
        return theoretical_speed, theoretical_flow

class Adam_optimization():
    
    def __init__(self, objective, first_order_derivative, bounds, x0):
        self.n_iter = 5000
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

class plot_calibration_results():
    
    def __init__(self, observed_flow, observed_density, observed_speed, calibrated_paras):
        
        self.observed_flow = observed_flow
        self.observed_density = observed_density
        self.observed_speed = observed_speed
        self.calibrated_paras_S3 = calibrated_paras["S3"]      # Calibrated from fundamental diagram model, vf, kc, foc
        self.k = np.linspace(0.000001,150,70)
        self.theoretical_value = theoretical_value(self.k)
        self.theoretical_speed_S3, self.theoretical_flow_S3 = self.theoretical_value.S3(self.calibrated_paras_S3)
        
    def plot_qk(self):
        
        fig = plt.figure(figsize=(7,5))
        plt.scatter(self.observed_density, self.observed_flow, s = 4, marker='o', c='r', edgecolors='r', label = 'Observation')
        plt.plot(self.k, self.theoretical_flow_S3, 'b-', linewidth=4, label = "S3")
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel('Density (veh/mi/ln)', fontsize=16)
        plt.ylabel('Flow (veh/h/ln)', fontsize=16)
        plt.xlim((0, 150))
        plt.ylim((0, 2100))
        plt.legend(loc='upper right', fontsize=14)
        plt.title('Flow vs. density', fontsize=20)
        fig.savefig('./Figures/flow vs density.png', dpi=400, bbox_inches='tight')
    
    def plot_vk(self):
        
        fig = plt.figure(figsize=(7,5))
        plt.scatter(self.observed_density, self.observed_speed, s = 4, marker='o', c='r', edgecolors='r', label = 'Observation')
        plt.plot(self.k, self.theoretical_speed_S3, 'b-', linewidth=4, label = "S3")
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel('Density (veh/mi/ln)', fontsize=16)
        plt.ylabel('Speed (km/h)', fontsize=16)
        plt.xlim((0, 150))
        plt.ylim((0, 90))
        plt.legend(loc='upper right', fontsize=14)
        plt.title('Speed vs. density', fontsize=20)
        fig.savefig('./Figures/speed vs density.png', dpi=400, bbox_inches='tight')
        
    def plot_vq(self):
        
        fig = plt.figure(figsize=(7,5))
        plt.scatter(self.observed_flow, self.observed_speed, s = 4, marker='o', c='r', edgecolors='r', label = 'Observation')
        plt.plot(self.theoretical_flow_S3, self.theoretical_speed_S3, 'b-', linewidth=4, label = "S3")
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel('Flow (veh/h/ln)', fontsize=16)
        plt.ylabel('Speed (km/h)', fontsize=16)
        plt.xlim((0,2100))
        plt.ylim((0, 90))
        plt.legend(loc='upper right', fontsize=14)
        plt.title('Speed vs. flow', fontsize=20)
        fig.savefig('./Figures/speed vs flow.png', dpi=400, bbox_inches='tight')

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

class calibrate():
    
    def __init__(self, flow, density, speed):
        self.flow = flow
        self.density = density
        self.speed = speed
        self.init_model_dict()
        
    def init_model_dict(self):
        self.model = fundamental_diagram_model(self.flow, self.density, self.speed)
        self.first_order_derivative = first_order_derivative(self.flow, self.density, self.speed)
        self.model_dict = {"S3":self.model.S3,}
        self.derivative = {"S3": self.first_order_derivative.S3,}
        self.bounds = {"S3": np.asarray([[70,80], [20,60], [1,8]]),}
        self.x0 = {"S3": np.asarray([75, 35, 3.6]),}
    
    def getSolution(self, model_str):
        # calibration
        objective = self.model_dict[model_str]
        derivative = self.derivative[model_str]
        bounds = self.bounds[model_str]
        x0 = self.x0[model_str]
        Adam = Adam_optimization(objective, derivative, bounds, x0)
        solutions, scores = Adam.adam()
        parameters = solutions[np.argmin(scores)]
        return parameters

if __name__ == '__main__':
    
    # load q-k-v data
    filepath = './'
    data = pd.read_csv(filepath+'input_data.csv', header=0, index_col=None)
    flow = np.array(data.Flow)
    density = np.array(data.Density)
    speed = np.array(data.Speed)
    
    solver = calibrate(flow, density, speed)
    result = {"S3":solver.getSolution("S3"),}
    print('Calibration results:\n'+'vf =', format(result['S3'][0], '.2f') + ' mi/h')
    print('kc =', format(result['S3'][1], '.2f') + ' veh/mi/ln')
    print('m =', format(result['S3'][2], '.2f'))
    
    plot_results = plot_calibration_results(flow, density, speed, result)
    plot_results.plot_qk(), plot_results.plot_vk(), plot_results.plot_vq()
    
    metrics = getMetrics(flow, density, speed)
    S3_RMSE_SPEED_Overall, S3_RMSE_FLOW_Overall, S3_R2_SPEED_Overall, S3_R2_FLOW_Overall = metrics.S3_RMSE_Overall(result["S3"])
    