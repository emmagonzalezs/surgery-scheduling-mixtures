import os
import pandas as pd
import pickle

# Load the grouped surgeries dictionary
grouped_surgeries_dict = pickle.load(open('grouped_surgeries_dict.pkl', 'rb'))


class SurgeryWeights:

    def __init__(self, data_set):
        self.data_set = data_set
        self.dict_filename = 'weights_for_mixture.pkl'
        self.totals_filename = 'probabilities_per_general.pkl'
        self.probs_per_surgery_filename = 'probabilities_per_surgery.pkl'  # per-surgery probabilities
        self.probs_per_surgery_keys_filename = 'list_individual_surgeries.pkl'  # Keys individual surgeries

    def save_dictionary(self, dictionary):
        with open(self.dict_filename, 'wb') as weights_dict:
            pickle.dump(dictionary, weights_dict)

    def save_probs(self, totals_list):
        with open(self.totals_filename, 'wb') as probs_file:
            pickle.dump(totals_list, probs_file)

    def save_probs_per_surgery(self, probs_per_surgery):
        with open(self.probs_per_surgery_filename, 'wb') as probs_file:
            pickle.dump(probs_per_surgery, probs_file)

    def save_probs_per_surgery_keys(self, probs_per_surgery_keys):
        with open(self.probs_per_surgery_keys_filename, 'wb') as keys_file:
            pickle.dump(probs_per_surgery_keys, keys_file)

    def load_dictionary(self):
        if os.path.exists(self.dict_filename):
            with open(self.dict_filename, 'rb') as weights_dict:
                return pickle.load(weights_dict)
        return None

    def load_probs(self):
        if os.path.exists(self.totals_filename):
            with open(self.totals_filename, "rb") as totals_file:
                return pickle.load(totals_file)
        return None

    def load_probs_per_surgery(self):
        if os.path.exists(self.probs_per_surgery_filename):
            with open(self.probs_per_surgery_filename, "rb") as probs_file:
                return pickle.load(probs_file)
        return None

    def load_probs_per_surgery_keys(self):
        if os.path.exists(self.probs_per_surgery_keys_filename):
            with open(self.probs_per_surgery_keys_filename, "rb") as keys_file:
                return pickle.load(keys_file)
        return None

    def all_class_size_per_surgery(self, sheet):
        class_size_dataset = pd.read_excel(self.data_set, sheet_name=sheet)
        class_sizes_dict = {}

        for index, row in class_size_dataset.iterrows():
            ops = row['(main) OPS code']
            class_size = row['Class size (number of observations)']

            class_sizes_dict[ops] = class_size

        return class_sizes_dict

    def weights_for_mixture(self, sheet):
        weights_dict = {}
        totals_list = []
        dict_class_size = self.all_class_size_per_surgery(sheet)
        probs_per_surgery = {}  # New dictionary for probabilities per surgery

        # Calculate the total number of surgeries
        total_surgeries = sum(dict_class_size.values())

        for general, procedures in grouped_surgeries_dict.items():
            weights_per_general = []
            for procedure in procedures:
                weights_per_general.append(dict_class_size[procedure])

            total_per_general = sum(weights_per_general)
            totals_list.append(total_per_general)
            weights_dict[general] = [(1 / total_per_general) * weight for weight in weights_per_general]

        # Calculate and store the overall probability for each surgery
        for surgery, procedure in dict_class_size.items():
            probs_per_surgery[surgery] = dict_class_size[surgery] / total_surgeries

        # Calculate and save the overall probabilities per general category
        all_surgeries_total = sum(totals_list)
        probabilities_list = [(1 / all_surgeries_total) * totals for totals in totals_list]

        self.save_dictionary(weights_dict)
        self.save_probs(probabilities_list)
        self.save_probs_per_surgery(probs_per_surgery)  # Save the per-surgery probabilities
        self.save_probs_per_surgery_keys(list(probs_per_surgery.keys()))

        return weights_dict


# Example usage
class_weights = SurgeryWeights('filtered_surgery_weights_germany_2019.xlsx')
# print(class_weights.all_class_size_per_surgery(sheet='specialized_general'))
# weights_dict = class_weights.weights_for_mixture(sheet='specialized_general')
# probs_per_surgery = class_weights.load_probs_per_surgery()
# print(probs_per_surgery)

