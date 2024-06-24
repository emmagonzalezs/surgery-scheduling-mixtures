import pickle
import numpy as np

# weights_per_surgery = pickle.load(open('total_weights.pkl', 'rb'))
#
# class WaitingList:
#     def __init__(self, grouped_surgeries_dict):
#         self.grouped_surgeries_dict = grouped_surgeries_dict
#
#     def calculate_probabilities(self):
#         total_weight = sum(weights_per_surgery)
#         probabilities = [surgery / total_weight for surgery in weights_per_surgery]
#         return probabilities
#
# # Load the necessary data
# grouped_surgeries_dict = pickle.load(open('grouped_surgeries_dict.pkl', 'rb'))
# weights_per_surgery = pickle.load(open('total_weights.pkl', 'rb'))
#
# # Initialize the class instance
# surgery_probabilities = WaitingList(grouped_surgeries_dict)
#
# # Calculate probabilities
# probabilities = surgery_probabilities.calculate_probabilities(weights_per_surgery)
#
# # Print or use probabilities as needed
# print(probabilities)
#
#
# # grouped_surgeries_dict = pickle.load(open('grouped_surgeries_dict.pkl', 'rb'))
#
# #
# # keys_grouped = grouped_surgeries_dict.keys()
# # surgeries_list = list(keys_grouped)
# #
# # total_weight = sum(weights_per_surgery)
# # probabilities = []
# #
# # for surgery in weights_per_surgery:
# #     prob = surgery / total_weight
# #     probabilities.append(prob)
#
# random_waiting_list = np.random.choice(surgeries_list, size=10, p=probabilities)


class WaitingList:
    def __init__(self, surgeries, weights):
        self.surgeries_list = list(surgeries.keys())
        self.probabilities = self.calculate_probabilities(weights)

    def __call__(self, size):
        return self.generate_waiting_list(size)

    @staticmethod
    def calculate_probabilities(weights):
        total_weight = sum(weights)
        probabilities = [surgery / total_weight for surgery in weights]
        return probabilities

    def generate_waiting_list(self, size):
        return np.random.choice(self.surgeries_list, size=size, p=self.probabilities)


grouped_surgeries_dict = pickle.load(open('grouped_surgeries_dict.pkl', 'rb'))
weights_per_surgery = pickle.load(open('total_weights.pkl', 'rb'))
waiting_list = WaitingList(grouped_surgeries_dict, weights_per_surgery)

# Generate random waiting list
random_waiting_list = waiting_list(size=10)


