import numpy as np
import pickle
from scipy import stats
import matplotlib.pyplot as plt
from scipy.stats import norm

surgery_steps_dict = pickle.load(open('surgery_steps_dict.pkl', 'rb'))
weights_dictionary = pickle.load(open('weights_for_mixture.pkl', 'rb'))

def generate_params_total_surgeries(prefix):
    total_procedure_times = {}  # {ops:[x1000 durations]}
    number_of_samples = 1000

    for procedure_key, procedure_steps in surgery_steps_dict.items():
        if not procedure_key.startswith(prefix):  # run only for given surgery
            continue

        total_time_per_procedure = np.zeros(number_of_samples)

        for steps in procedure_steps:
            distribution, parameter1, parameter2 = steps
            if distribution == 'lognormal':
                random_samples = np.random.lognormal(mean=parameter1, sigma=parameter2, size=number_of_samples)
            elif distribution == 'gamma':
                random_samples = np.random.gamma(parameter1, parameter2, size=number_of_samples)
            else:
                random_samples = np.random.weibull(parameter1, size=number_of_samples) * parameter2

            total_time_per_procedure += random_samples

        total_procedure_times[procedure_key] = total_time_per_procedure

    return total_procedure_times


def total_procedures_parameters(distr, prefix):
    all_procedure_times = generate_params_total_surgeries(prefix)
    parameters_per_procedure = {}

    for surgery, data in all_procedure_times.items():
        if distr == "normal":
            norm_params = stats.norm.fit(data)  # (mu, sigma)
            parameters_per_procedure[surgery] = norm_params  # {"ops":[mu,sigma]}
        else:  # distr == "lognormal"
            lognorm_params = stats.lognorm.fit(data)
            s, loc, scale = lognorm_params
            exp = np.exp(loc + 0.5 * (s ** 2))
            sd = np.sqrt((np.exp(s ** 2) - 1) * np.exp(2 * loc + s ** 2))
            all_lognorm_params = lognorm_params + (exp, sd)
            parameters_per_procedure[surgery] = all_lognorm_params

    return parameters_per_procedure


def total_procedure_parameters_multimodal(distr, prefix):
    all_procedure_times = generate_params_total_surgeries(prefix)
    parameters_per_procedure = {}
    separation_factor = 5  # Factor to control the separation between means
    sd_variation_factor = 0.5
    overall_std = np.std([time for times in all_procedure_times.values() for time in times])
    # weights = [0.6, 0.2, 0.1, 0.1]
    procedures = list(all_procedure_times.items())
    n = len(procedures)
    mu = 0
    sigma = 0
    for idx, (surgery, data) in enumerate(all_procedure_times.items()):
        if distr == "normal":
            mu, sigma = stats.norm.fit(data)

            mu += idx * overall_std
            # sigma = sigma * sd_variation_factor
            parameters_per_procedure[surgery] = (mu, sigma)

        else:  # distr == "lognormal"
            lognorm_params = stats.lognorm.fit(data)
            s, loc, scale = lognorm_params
            exp = np.exp(loc + 0.5 * (s ** 2))
            sd = np.sqrt((np.exp(s ** 2) - 1) * np.exp(2 * loc + s ** 2))
            all_lognorm_params = lognorm_params + (exp, sd)
            parameters_per_procedure[surgery] = all_lognorm_params

    return parameters_per_procedure


def mixture_distribution_multimodal(distr, prefix):
    parameters_surgeries = total_procedure_parameters_multimodal(distr, prefix)
    means = []
    sds = []
    exp_sd_mixture = []

    if distr == 'normal':
        for params in parameters_surgeries.values():
            means.append(params[0])
            sds.append(params[1])

        weights = weights_dictionary[prefix]
        # weights = [0.4,0.3,0.3]
        # weights = [0.4, 0.3, 0.3]
        # print(weights)
        expected_value_mixture = sum(mean * weight for mean, weight in zip(means, weights))
        exp_sd_mixture.append(expected_value_mixture)

        variance_mixture = sum((weight * (sd ** 2 + mean ** 2) for weight, sd, mean in zip(weights, sds, means))) - (
                    expected_value_mixture ** 2)
        if variance_mixture < 0:
            print(f"Here negative variance: {variance_mixture}")
        exp_sd_mixture.append(np.sqrt(variance_mixture))

    else:  # lognormal
        pass

    return parameters_surgeries, exp_sd_mixture


def mixture_distribution(distr, prefix):
    parameters_surgeries = total_procedures_parameters(distr, prefix)
    means = []
    sds = []
    exp_sd_mixture = []

    if distr == 'normal':
        for code, params in parameters_surgeries.items():
            means.append(params[0])
            sds.append(params[1])

        weights = weights_dictionary[prefix]
        # weights = [0.6, 0.2, 0.1, 0.1]
        # weights = [0.4, 0.3, 0.3]
        expected_value_mixture = sum(mean * weight for mean, weight in zip(means, weights))
        exp_sd_mixture.append(expected_value_mixture)

        variance_mixture = sum((weight * (sd ** 2 + mean ** 2) for weight, sd, mean in zip(weights, sds, means))) - (expected_value_mixture ** 2)
        if variance_mixture < 0:
            print(f"Here negative variance: {variance_mixture}")
        exp_sd_mixture.append(np.sqrt(variance_mixture))

    else:  # lognormal
        pass

    return parameters_surgeries, exp_sd_mixture  # dict of individual surgeries, list of [mu_mix, sigma_mix]

# NOW USING MULTIMODAL FUNCTION
def dictionary_all_params(distr, prefix):
    all_params = mixture_distribution_multimodal(distr, prefix)
    data_surgeries = {prefix: [tuple(all_params[1])]}
    mixture_params = tuple(all_params[1])
    for key, values in all_params[0].items():
        if key not in data_surgeries:
            data_surgeries[key] = []
        data_surgeries[key].append(values)
        # data_surgeries = {"general code":[mu_mix, sigma_mix], "procedure1":[mu1,sigma1],...}
        # mixture_params = (mu_mix, sigma_mix)
    return data_surgeries, mixture_params


grouped_surgeries_dict = pickle.load(open('grouped_surgeries_dict.pkl', 'rb'))
probabilities_per_general = pickle.load(open('probabilities_per_general.pkl', 'rb'))
list_individual_surgeries = pickle.load(open("list_individual_surgeries.pkl", 'rb'))
probabilities_per_individual = pickle.load(open('probabilities_per_surgery.pkl', 'rb'))


def waiting_list(size, distr):  # each key is a patient id
    surgeries_list = list(grouped_surgeries_dict.keys())  # list of patient ids
    generated_waiting_list = np.random.choice(surgeries_list, size=size, p=probabilities_per_general)
    # print(generated_waiting_list)
    # generated_waiting_list is a list of surgeries
    # print(generated_waiting_list)
    # dict_parameters_mixture_and_individuals = {}
    dict_parameters_individuals = {}
    dict_mixture = {}

# Dict with keys being the general code:
# for surgery in generated_waiting_list:
#     surgery_params = dictionary_all_params(distr, surgery)
#     individual = surgery_params[0]
#     mixture = surgery_params[1]
#     combined_list = []
#     for values in individual.values():
#         combined_list.extend(values)
#     if surgery in dict_parameters_individuals:
#         # dict_parameters_mixture_and_individuals[surgery].append(combined_list)
#         dict_parameters_individuals[surgery].append(combined_list[1:])
#         dict_mixture[surgery].append(mixture)
#     else:
#         # dict_parameters_mixture_and_individuals[surgery] = [combined_list]
#         dict_parameters_individuals[surgery] = [combined_list[1:]]
#         dict_mixture[surgery] = [mixture]

    for i, surgery in enumerate(generated_waiting_list, start=1):
        surgery_params = dictionary_all_params(distr, surgery)
        individual = surgery_params[0]
        mixture = surgery_params[1]
        combined_list = []
        for values in individual.values():
            combined_list.extend(values)  # [(procedure1), (procedure2 params), ...]
        dict_parameters_individuals[str(i)] = [combined_list[1:]]
        dict_mixture[str(i)] = [surgery, mixture]

    return dict_mixture, dict_parameters_individuals


def waiting_list_individual_surgeries(size):
    generated_waiting_list = np.random.choice(list_individual_surgeries, size=size, p=probabilities_per_individual)
    individual_waiting_list_dict = {}

    # TODO: generate a waiting list of individual surgeries

    return individual_waiting_list_dict

# test = waiting_list(50, "normal")
# print("waiting list", test[0])
# print("individual parameters", test[1])


# Example usage
prefix = "5-455"  # Adjust this as per your data
distr = "normal"

parameters_surgeries, mixture_params = mixture_distribution(distr, prefix)
# print(parameters_surgeries)

# Extract means and standard deviations of individual distributions
means = [params[0] for params in parameters_surgeries.values()]
sds = [params[1] for params in parameters_surgeries.values()]

# Extract mean and standard deviation of the mixture distribution
mixture_mean = mixture_params[0]
mixture_sd = mixture_params[1]

# Generate x values for plotting
x = np.linspace(min(means) - 3 * max(sds), max(means) + 3 * max(sds), 1000)

# Parameters of the individual distributions
# means = [30, 50, 100]
# std_devs = [5, 10, 20]
weights = weights_dictionary[prefix]
# weights = [0.4, 0.3, 0.3]

# Create the mixture distribution
y_mixture = np.zeros_like(x)
for weight, mean, std_dev in zip(weights, means, sds):
    y_mixture += weight * norm.pdf(x, mean, std_dev)

# Plot the mixture distribution

# plt.legend()
# plt.show()
# Plot individual normal distributions
# plt.figure(figsize=(10, 6))

for mean, sd in zip(means, sds):
    y = norm.pdf(x, mean, sd)
    plt.plot(x, y, label=f'Individual Normal (μ={mean:.2f}, σ={sd:.2f})')

plt.plot(x, y_mixture, label=f'Mixture Normal (μ={mixture_mean:.2f}, σ={mixture_sd:.2f})', linewidth=2, linestyle='--')
# Plot mixture distribution
# y_mixture = norm.pdf(x, mixture_mean, mixture_sd)
# plt.plot(x, y_mixture, label=f'Mixture Normal (μ={mixture_mean:.2f}, σ={mixture_sd:.2f})', linewidth=2, linestyle='--')

# Add labels and legend
plt.title(f'Normal Distributions for {prefix} Surgeries')
plt.xlabel('Duration')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.show()
