import os
import time
import pandas as pd
import numpy as np
# from matplotlib import pyplot as plt
from scipy import stats
import pickle


class DatasetDistributions:
    def __init__(self, data_set):
        self.data_set = data_set
        self.dict_filename = 'surgery_steps_dict.pkl'

    def __call__(self, data_sheet):
        return self.generate_surgery_steps_dict(data_sheet)

    def save_dictionary(self, dictionary):  # save time by saving all parameters in dictionary
        with open(self.dict_filename, 'wb') as all_parameters:
            pickle.dump(dictionary, all_parameters)

    def load_dictionary(self):
        if os.path.exists(self.dict_filename):
            with open(self.dict_filename, 'rb') as all_parameters:
                data = pickle.load(all_parameters)
                # for key, value in data.items():
                #     print(f"Key: {key}")
                #     print(f"Value: {value}")
                return data
        return None

    def generate_surgery_steps_dict(self, sheet):
        start_time = time.time()
        surgery_dataset = pd.read_excel(self.data_set, sheet_name=sheet)
        surgery_steps_dict = {}  # all steps per surgery with {ops: [distr,param1,param2]}

        for index, row in surgery_dataset.iterrows():
            ops = row['(main) ops code']
            distr = row['distribution type']
            param1 = row['parameter1']
            param2 = row['parameter2']

            if ops in surgery_steps_dict:
                surgery_steps_dict[ops].append([distr, param1, param2])
            else:
                surgery_steps_dict[ops] = [[distr, param1, param2]]

        end_time = time.time()
        self.save_dictionary(surgery_steps_dict)
        # print(f"Time to generate and save dict: {end_time - start_time}")
        return surgery_steps_dict

    # def surgery_durations(self, sheet):
    #     start_time = time.time()
    #     surgery_steps_dict = self.load_dictionary()
    #     if surgery_steps_dict is None:
    #         surgery_steps_dict = self.generate_surgery_steps_dict(sheet)
    #     # else:
    #     #  print("loaded dict from file")  # as dict loaded, no need to generate, just grad saved file
    #     end_time = time.time()
    #     # print(f"Time to load dict: {end_time - start_time}")
    #     total_procedure_times = {}  # {ops:[x1000 durations]
    #     number_of_samples = 1000
    #
    #     for procedure_key, procedure_steps in surgery_steps_dict.items():
    #         total_time_per_procedure = np.zeros(number_of_samples)
    #
    #         for steps in procedure_steps:
    #             distribution, parameter1, parameter2 = steps
    #             if distribution == 'lognormal':
    #                 random_samples = np.random.lognormal(mean=parameter1, sigma=parameter2, size=number_of_samples)
    #             elif distribution == 'gamma':
    #                 random_samples = np.random.gamma(parameter1, parameter2, size=number_of_samples)
    #             else:
    #                 random_samples = np.random.weibull(parameter1, size=number_of_samples) * parameter2
    #
    #             total_time_per_procedure += random_samples
    #
    #         total_procedure_times[procedure_key] = total_time_per_procedure
    #
    #     return total_procedure_times

    # def total_procedures_parameters(self, sheet, distr):
    #     all_procedure_times = self.surgery_durations(sheet)
    #     parameters_per_procedure = {}
    #     for surgery, data in all_procedure_times.items():
    #
    #         # KS Test for best distribution:
    #         # gamma_params = stats.gamma.fit(data)
    #         # log_test = stats.kstest(data, stats.lognorm.cdf, args=lognorm_params)
    #         # gamma_test = stats.kstest(data, stats.gamma.cdf, args=gamma_params)
    #         # best_dist = min(log_test.statistic, gamma_test.statistic)
    #
    #         if distr == "normal":
    #             norm_params = stats.norm.fit(data)  # (mu,sigma)
    #             parameters_per_procedure[surgery] = norm_params
    #         else:
    #             lognorm_params = stats.lognorm.fit(data)
    #             s, loc, scale = lognorm_params
    #             exp = np.exp(loc + 0.5 * (s ** 2))
    #             sd = np.sqrt((np.exp(s ** 2) - 1) * np.exp(2 * loc + s ** 2))
    #             all_lognorm_params = lognorm_params + (exp, sd)
    #             parameters_per_procedure[surgery] = all_lognorm_params
    #
    #         # else:  # possibly select also gamma (37 out of 182 better fit with gamma)
    #         #     shape, loc, scale = gamma_params
    #         #     exp = shape * scale
    #         #     sd = np.sqrt(shape * (scale**2))
    #         #     all_gamma_params = gamma_params + (exp, sd)
    #         #     parameters_per_procedure[surgery] = all_gamma_params
    #
    #     return parameters_per_procedure


dataset_class = DatasetDistributions('filtered_surgical_times_germany_2019.xlsx')
parameters_per_procedure_general = dataset_class('specialized_general')
# print(parameters_per_procedure_general)
# print(dataset_class.surgery_durations('specialized_general')["1-653"])
# print(parameters_per_procedure["1-653"])

#### Testing one surgery (fitted distribution, KS test and histogram)
# try1 = surgery_durations('specialized_general')
# # Show first surgery in histogram
# first_key = next(iter(try1))
# first_surgery = try1[first_key]

# all_durations = dataset_class.surgery_durations('specialized_general')
# first_surgery = all_durations["5-916.a0"]
# params_first = parameters_per_procedure["5-916.a0"]
# print("parameters for 5-916.a0", params_first)
# #
#### Fit distribution to the first_surgery
# params_log = stats.lognorm.fit(first_surgery)
# params_gamma = stats.gamma.fit(first_surgery)
# params_normal = stats.norm.fit(first_surgery)
# params_weibull = stats.weibull_min.fit(first_surgery)
#
#
###### Evaluate distributions by KS test
# log_test = stats.kstest(first_surgery, stats.lognorm.cdf, args=params_log)
# gamma_test = stats.kstest(first_surgery, stats.gamma.cdf, args=params_gamma)
# normal_test = stats.kstest(first_surgery, stats.norm.cdf, args=params_normal)
# weibull_test = stats.kstest(first_surgery, stats.weibull_min.cdf, args=params_weibull)
#
###### Testing best distribution by Kolmogorov-Smirnov Test
# best_dist = min(log_test.statistic, gamma_test.statistic, normal_test.statistic,weibull_test.statistic)
# if best_dist == log_test.statistic:
#     print("By KS test, lognormal distribution is the best fit, with parameters:", params_log)
# elif best_dist == gamma_test.statistic:
#     print("By KS test, gamma distribution is the best fit, with parameters:", params_gamma)
# elif best_dist == normal_test.statistic:
#     print("By KS test, normal distribution is the best fit, with parameters:", params_gamma)
# else:
#     print("By KS test, Weibull distribution is the best fit, with parameters:", params_weibull)
#
###### Checking largest p-value from KS Test:
# max_pval = max(log_test.pvalue, gamma_test.pvalue, normal_test.pvalue,weibull_test.pvalue)
# if max_pval == log_test.pvalue:
#     print("Highest p-value from lognormal:", max_pval)
# elif max_pval == gamma_test.pvalue:
#     print("Highest p-value from gamma:", max_pval)
# elif max_pval == normal_test.pvalue:
#     print("Highest p-value from normal:", max_pval)
# else:
#     print("Highest p-value from Weibull:", max_pval)
#
#
###### Generate histogram of first surgery with corresponding fitted distribution
# xmin, xmax = min(first_surgery), max(first_surgery)
# x = np.linspace(xmin, xmax, 100)
# s, loc, scale, ex, sd = params_first
# pdf = stats.lognorm.pdf(x, s, loc, scale)
# plt.hist(first_surgery, bins=20, density=True, alpha=0.7, label='Histogram')
# #
# ##### Plot fitted distributions
# plt.plot(x, pdf, 'r-', label='Fitted lognormal distribution')
# # plt.plot(x, stats.norm.pdf(x, *params_normal), 'y-', label="Fitted normal distribution")
# # plt.plot(x, stats.gamma.pdf(x, *params_gamma), 'g-', label="Fitted gamma distribution")
# # plt.plot(x, stats.weibull_min.pdf(x, *params_weibull), label='Fitted weibull_min distribution')
#
# plt.title('Histogram with Fitted Distributions')
# plt.xlabel('Data')
# plt.ylabel('Density')
# plt.legend()
# plt.show()

#
##### Check if all surgeries fit a lognormal distribution using KS test
# count_log_ks = 0
# count_gamma_ks = 0
#
# for surgery, data in try1.items():
#     lognorm_params = stats.lognorm.fit(data)
#     kstest_log = stats.kstest(data, stats.lognorm.cdf, args=lognorm_params)
#
#     gamma_params = stats.gamma.fit(data)
#     kstest_gamma = stats.kstest(data, stats.gamma.cdf, args=gamma_params)
#
#     if kstest_log.statistic < kstest_gamma.statistic:
#         best_fit = 'lognormal'
#         count_log_ks += 1
#     else:
#         best_fit = 'gamma'
#         count_gamma_ks += 1
#
# print("Log surgeries", count_log_ks)
# print("Gamma surgeries", count_gamma_ks)
