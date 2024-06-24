from SurgeryParamsDict import DatasetDistributions
import numpy as np
from scipy.stats import lognorm
import matplotlib.pyplot as plt
from scipy.stats import norm
import os
import pickle

weights_dictionary = pickle.load(open('weights_for_mixture.pkl', 'rb'))


class GroupedSurgeries:
    def __init__(self):
        self.dataset = DatasetDistributions('filtered_surgical_times_germany_2019.xlsx')
        self.dict_filename = 'grouped_surgeries_dict.pkl'

    def save_dictionary(self, dictionary):
        with open(self.dict_filename, 'wb') as grouped_surgeries:
            pickle.dump(dictionary, grouped_surgeries)

    def load_dictionary(self):
        if os.path.exists(self.dict_filename):
            with open(self.dict_filename, 'rb') as grouped_surgeries:
                return pickle.load(grouped_surgeries)
        return None

    def generate_grouped_surgeries_dict(self, sheet):
        all_surgery_parameters = self.dataset(sheet)
        grouped_surgeries = {}  # dict with surgeries as keys (until .) and possible procedures listed

        for surgery, value in all_surgery_parameters.items():
            general_code = surgery.split('.')[0]
            if general_code not in grouped_surgeries:
                grouped_surgeries[general_code] = []
            grouped_surgeries[general_code].append(surgery)

        self.save_dictionary(grouped_surgeries)

        return grouped_surgeries

    # def parameters_grouped_surgeries(self, sheet, distribution):
    #     grouped_surgeries = self.load_dictionary()
    #     all_surgery_parameters = self.dataset(sheet)
    #     if grouped_surgeries is None:
    #         grouped_surgeries = self.generate_grouped_surgeries_dict(sheet)
    #
    #     parameters_per_group = {}
    #
    #     for general_code, surgeries in grouped_surgeries.items():
    #         parameters_per_group[general_code] = []
    #         for surgery in surgeries:
    #             parameters_per_group[general_code].append(all_surgery_parameters[surgery])
    #
    #     return parameters_per_group
    #
    # def expected_durations_per_surgery_group(self, sheet, distribution):
    #     expectations_general_code = {}
    #     variances_general_code = {}
    #     sds_general_code = {}
    #     parameters_per_general_code = self.parameters_grouped_surgeries(sheet, distribution)
    #     # print("within expected function:", parameters_per_general_code)
    #     if distribution == 'normal':
    #         for general, params_list in parameters_per_general_code.items():
    #             means = [params[0] for params in params_list] # all mus per general code
    #             weights = weights_dictionary[general]
    #             expected_value_mixture = sum(mean * weight for mean, weight in zip(means, weights))
    #             expectations_general_code[general] = expected_value_mixture
    #
    #             sds = [params[1] for params in params_list]
    #             variances_1 = sum((weight * (sd ** 2 + mean ** 2) for weight, sd, mean in zip(weights, sds, means)))
    #             variance_mixture = variances_1 - (expected_value_mixture ** 2)
    #             if variance_mixture < 0:
    #                 print(f"Here negative variance: {variance_mixture}")
    #             variances_general_code[general] = variance_mixture
    #             sds_general_code[general] = np.sqrt(variance_mixture)
    #
    #     else:  # lognormal
    #         pass
    #
    #     return expectations_general_code, sds_general_code


mixture_surgeries_class = GroupedSurgeries()
# expected_dur, variances_dur, sds_dur = mixture_surgeries_class.expected_durations_per_surgery_group('specialized_general', 'normal')
# print("expected duration of mixture", expected_dur)
# print("variances:", variances_dur)
# print("sds:", sds_dur)

###### testing on one group only
#
# parameters_per_group_test = mixture_surgeries_class.parameters_grouped_surgeries("specialized_general", 'normal')
# print(parameters_per_group_test["5-916"])
# expected_dur, variances_dur, sds_dur = mixture_surgeries_class.expected_durations_per_surgery_group('specialized_general', 'normal')
# print("expected duration of mixture", expected_dur["5-916"])
# print("variances:", variances_dur["5-916"])
# print("sd:", sds_dur["5-916"])
#
# # Grab parameters corresponding to the key "5-916"
# parameters_5_916 = parameters_per_group_test.get("5-916", [])
#
# # Generate data points for plotting the PDF
# x = np.linspace(0, 200, 1000)
#
# # Plot each normal distribution using the parameters
# for i, params in enumerate(parameters_5_916):
#     mean, sd = params
#     plt.plot(x, norm.pdf(x, loc=mean, scale=sd), label=f"Plot of curve a{i}")
#
# mixture_pdf = np.zeros_like(x)
# n = len(parameters_5_916)
# for params in parameters_5_916:
#     mean, sd = params
#     mixture_pdf += (1/n) * norm.pdf(x, loc=mean, scale=sd)
# plt.plot(x, mixture_pdf, label='Mixture Distribution', linestyle='--', color='black')
#
# # Add labels and legend
# plt.xlabel('Duration')
# plt.ylabel('Probability Density')
# plt.title('PDF of normal distributions and mixture distribution')
# plt.legend()
# plt.show()



# # generate 1000 random samples to set min and max values for range of x (for pdf)
# all_random_samples = []
# for procedure in parameters_one_group:
#     mean, s = procedure
#     random_sampl = np.random.normal(mean, s, size=50)
#     all_random_samples.extend(random_sampl)
#
# minval, maxval = min(all_random_samples), max(all_random_samples)
# print("min value:", minval)
# print("max value:", maxval)
#
# x = np.linspace(minval, maxval, 1000)
# pdfs = []
#
# for i, (mu, sigma) in enumerate(parameters_one_group):
#     pdf = norm.pdf(x, mu, sigma)
#     plt.plot(x, pdf, label=f'PDF {i}')
#     pdfs.append(pdf)
#     # expected_values.append(exp)
#     # print(f'expected values for procedure {i}:', exp)
#
# # Calculate the mixture PDF
# n = len(parameters_one_group)
# mixture_pdf = sum([(1/n) * pdf for pdf in pdfs])
#
# # Calculate expected value of mixture PDF
# # expected_val_procedures = []
# # for i, procedure in enumerate(parameters):
# #     s2, loc2, scale2 = procedure
# #     expected_val = np.exp(loc2 + (s2**2)/2)
# #     print(f'expected values per procedure {i}:', expected_val)
# #     expected_val_procedures.append(expected_val)
# #     sd = np.sqrt((np.exp(s2 ** 2) - 1) * np.exp(2 * loc2 + s2 ** 2))
# #     print(f'sd per procedure {i}:', sd)
#
# mixture_expectation = np.sum([(1/n) * ex for ex in expectations])
# print("mixture distr by w*e(x):", mixture_expectation)
# mix_exp = np.sum(x*pdfs)
# print("mixture exp by pdf:", mix_exp)
#
#
# # Plot the mixture PDF example
# plt.plot(x, mixture_pdf, label='Mixture PDF', linestyle='--', color='black')
# plt.xlabel('x')
# plt.ylabel('Probability Density')
# plt.title('Mixture and Individual Log-Normal Distributions')
# plt.legend()
# plt.grid(True)
# plt.show()
#
# print(weights_dictionary)