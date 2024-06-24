from WaitingList import waiting_list
from scipy.stats import norm
from scipy.optimize import fsolve
import numpy as np
import pickle
import random
import itertools
import time
from multiprocessing import Pool, cpu_count


class SurgeryScheduling:
    def __init__(self, number_surgeries, distr, number_of_ors, alpha):
        self.number_surgeries = number_surgeries
        self.distr = distr
        self.number_of_ors = number_of_ors
        # Initialize waiting list to schedule in ORs
        self.waiting = waiting_list(number_surgeries, distr)
        # print(waiting[0])
        # self.waiting_list_mixture_surgeries = self.waiting[0]  # {'1': ["general code", (mu,sigma)]}
        # self.waiting_list_mixture_surgeries = pickle.load(open('waiting_list_5_surgeries.pickle', 'rb'))
        # self.waiting_list_mixture_surgeries = pickle.load(open('waiting_list_300_surgeries.pickle', 'rb'))
        # self.waiting_list_mixture_surgeries = pickle.load(open('waiting_list_500_surgeries.pickle', 'rb'))
        self.waiting_list_mixture_surgeries = pickle.load(open('waiting_list_400_surgeries_multi.pickle', 'rb'))

        self.waiting_list_mixture_surgeries_list = [[key, *value] for key, value in
                                                    self.waiting_list_mixture_surgeries.items()]
        # self.waiting_list_individual_surgeries = self.waiting[1]  # {'1': [(mu11,sigma11), (mu12, sigma12)]}
        # self.waiting_list_individual_surgeries = pickle.load(open('waiting_individual_300_surgeries.pickle', 'rb'))
        # self.waiting_list_individual_surgeries = pickle.load(open('individual_params_5_surgeries.pickle', 'rb'))
        # self.waiting_list_individual_surgeries = pickle.load(open('waiting_individual_500_surgeries.pickle', 'rb'))
        self.waiting_list_individual_surgeries = pickle.load(open('waiting_individual_400_surgeries_multi.pickle', 'rb'))

        # Weights per surgery in waiting list, for delta calculations
        self.weights_dictionary = pickle.load(open('weights_for_mixture.pkl', 'rb'))  # {"general code": [w11,w12,w13]}
        # self.weights_surgeries_waiting_list = [self.weights_dictionary[key] for key in
        #                                        self.waiting_list_mixture_surgeries if key in self.weights_dictionary]
        self.weights_surgeries_waiting_list = [self.weights_dictionary[value[0]] for key, value in
                                               self.waiting_list_mixture_surgeries.items() if
                                               value[0] in self.weights_dictionary]  # surgery in position 0
        # weights_surgeries_waiting_list = [[w11,w12,w13], [w21,w22], ...]

        # self.ors = {i: [] for i in range(1, number_of_ors + 1)}  # {"1":[], "2":[]}
        self.number_working_days = 60  # approx. 3 months = 60
        self.ors = {(k, t): [] for t in range(1, self.number_working_days + 1) for k in range(1, number_of_ors + 1)}
        # self.ors = {(OR, day):} for 1 year, append here ['surgery'
        self.max_capacity = 450  # per OR in minutes (7.5h)
        self.list_of_ors = [i for i in range(1, number_of_ors + 1)]  # [1,2,..., number_of_ors]
        self.list_of_ors = [(k, t) for t in range(1, self.number_working_days + 1) for k in range(1, number_of_ors + 1)]
        self.overtime_prob = 0.3
        self.alpha = alpha
        self.slack_stored_original = {}
        self.slack_stored_alternate = {}

    def sort_waiting_list(self):
        ordered_waiting_list_dict = dict(sorted(self.waiting_list_mixture_surgeries.items(), key=lambda item:
        item[1][1][0], reverse=True))
        ordered_waiting_list_list = [[k, *v] for k, v in ordered_waiting_list_dict.items()]

        sorted_surgeries = list(ordered_waiting_list_dict.keys())

        return ordered_waiting_list_dict, ordered_waiting_list_list, sorted_surgeries

    def total_surgery_time(self):
        total_time = sum(surgery[2][0] for surgery in self.waiting_list_mixture_surgeries_list)
        return total_time

    def first_fit(self, slack_method):  # surgeries not ordered
        current_schedule_ff = {or_plan: [] for or_plan in self.ors}
        remaining_surgeries = len(self.waiting_list_mixture_surgeries_list)
        if slack_method == "original":
            for surgery_info in self.waiting_list_mixture_surgeries_list:
                assigned = False
                for or_plan, surgeries in current_schedule_ff.items():
                    # Calculate total surgery time and slack time for the OR
                    if surgeries:
                        total_time = sum(t[2][0] for t in surgeries) + surgery_info[2][0] \
                                     + self.slack_time(surgeries + [surgery_info])
                        if total_time <= self.max_capacity:
                            current_schedule_ff[or_plan].append(surgery_info)
                            assigned = True
                            break
                    else:
                        current_schedule_ff[or_plan].append(surgery_info)
                        assigned = True
                        break

                if not assigned:
                    # print(current_schedule_ff)
                    min_or_used = min(current_schedule_ff, key=lambda k: sum(x[2][0] for x in current_schedule_ff[k])
                                                                         + surgery_info[2][0]
                                                                         + self.slack_time(
                        current_schedule_ff[k] + [surgery_info]) - self.max_capacity)
                    current_schedule_ff[min_or_used].append(surgery_info)

                remaining_surgeries -= 1
                print(f"FF: Iteration {len(self.waiting_list_mixture_surgeries_list) - remaining_surgeries}: "
                      f"{remaining_surgeries} surgeries remaining")

            total_overtime = 0
            for or_id, or_surgeries in current_schedule_ff.items():
                total_time = sum(t[2][0] for t in or_surgeries) + self.slack_time(or_surgeries)
                overtime = max(total_time - self.max_capacity, 0)
                total_overtime += overtime

            # for or_plan, surgeries in current_schedule_ff.items():
            #     total_time = sum(surgery[2][0] for surgery in surgeries)
            #     total_slack = self.slack_time(surgeries)
            #     print(f"OR {or_plan} scheduled time: {total_time} hours, slack: {total_slack} hours")

        else:
            for surgery_info in self.waiting_list_mixture_surgeries_list:
                assigned = False
                for or_plan, surgeries in current_schedule_ff.items():
                    # Calculate total surgery time and slack time for the OR
                    if surgeries:
                        total_time = sum(t[2][0] for t in surgeries) + surgery_info[2][0] \
                                     + self.slack_time_alternate(surgeries + [surgery_info])
                        if total_time <= self.max_capacity:
                            current_schedule_ff[or_plan].append(surgery_info)
                            assigned = True
                            break
                    else:
                        current_schedule_ff[or_plan].append(surgery_info)
                        assigned = True
                        break

                if not assigned:
                    # print(current_schedule_ff)
                    min_or_used = min(current_schedule_ff, key=lambda k: sum(x[2][0] for x in current_schedule_ff[k])
                                                                         + surgery_info[2][0]
                                                                         + self.slack_time_alternate(
                        current_schedule_ff[k] + [surgery_info]) - self.max_capacity)
                    current_schedule_ff[min_or_used].append(surgery_info)

                remaining_surgeries -= 1
                print(f"FF: Iteration {len(self.waiting_list_mixture_surgeries_list) - remaining_surgeries}: "
                      f"{remaining_surgeries} surgeries remaining")

            total_overtime = 0
            for or_id, or_surgeries in current_schedule_ff.items():
                total_time = sum(t[2][0] for t in or_surgeries) + self.slack_time_alternate(or_surgeries)
                overtime = max(total_time - self.max_capacity, 0)
                total_overtime += overtime

            # for or_plan, surgeries in current_schedule_ff.items():
            #     total_time = sum(surgery[2][0] for surgery in surgeries)
            #     total_slack = self.slack_time_alternate(surgeries)
            #     print(f"OR {or_plan} scheduled time: {total_time} hours, slack: {total_slack} hours")

        return current_schedule_ff, total_overtime

    def lpt(self, slack_method):
        _, ordered_waiting_list, _ = self.sort_waiting_list()
        current_schedule_lpt = {or_plan: [] for or_plan in self.ors}
        remaining_surgeries = len(ordered_waiting_list)

        if slack_method == "original":

            for surgery_info in ordered_waiting_list:
                # possible_ors = [or_id for or_id, surgeries in current_schedule_lpt.items() if not surgeries]
                # already stores which ORs are empty
                surgery_planned = False
                for or_plan, surgeries_in_or in current_schedule_lpt.items():

                    if not surgeries_in_or:
                        current_schedule_lpt[or_plan].append(surgery_info)
                        surgery_planned = True
                        break
                    else:
                        surgeries_or_with_surgery = surgeries_in_or + [surgery_info]  # find slack for all together
                        slack = self.slack_time(surgeries_or_with_surgery)
                        # print("calculated slack when nonempty OR", slack)
                        if sum(t[2][0] for t in current_schedule_lpt[or_plan]) + surgery_info[2][0] + \
                                slack <= self.max_capacity:
                            # possible_ors.append(or_plan)
                            current_schedule_lpt[or_plan].append(surgery_info)
                            surgery_planned = True
                            break

                if not surgery_planned:
                    min_or_used = min(current_schedule_lpt, key=lambda k: sum(x[2][0] for x in current_schedule_lpt[k])
                                                                          + surgery_info[2][0]
                                                                          + self.slack_time(
                        current_schedule_lpt[k] + [surgery_info]) - self.max_capacity)
                    current_schedule_lpt[min_or_used].append(surgery_info)

                remaining_surgeries -= 1
                print(f"LPT: Iteration {len(self.waiting_list_mixture_surgeries_list) - remaining_surgeries}: "
                      f"{remaining_surgeries} surgeries remaining")

            total_overtime = 0
            total_free_time = 0
            for or_id, or_surgeries in current_schedule_lpt.items():
                # Calculate total surgery time and slack time for the OR
                total_time = sum(t[2][0] for t in or_surgeries) + self.slack_time(current_schedule_lpt[or_id])
                overtime = max(total_time - self.max_capacity, 0)
                free_time = max(self.max_capacity - total_time, 0)
                total_overtime += overtime
                total_free_time += free_time
            #     print(f'Overtime in OR {or_id} using LPT = {overtime}')
            #     print(f'Slack for OR {or_id} using LPT = {self.slack_time(or_surgeries)}')
            # print("total overtime", total_overtime)

            # for or_plan, surgeries in current_schedule_lpt.items():
            #     total_time = sum(surgery[2][0] for surgery in surgeries)
            #     total_slack = self.slack_time(surgeries)
            #     print(f"LPT: OR {or_plan} scheduled time: {total_time} hours, slack: {total_slack} hours")

        else:
            for surgery_info in ordered_waiting_list:
                # possible_ors = [or_id for or_id, surgeries in current_schedule_lpt.items() if not surgeries]
                # already stores which ORs are empty
                surgery_planned = False
                for or_plan, surgeries_in_or in current_schedule_lpt.items():

                    if not surgeries_in_or:
                        current_schedule_lpt[or_plan].append(surgery_info)
                        surgery_planned = True
                        break
                    else:
                        surgeries_or_with_surgery = surgeries_in_or + [surgery_info]  # find slack for all together
                        slack = self.slack_time_alternate(surgeries_or_with_surgery)
                        # print("calculated slack when nonempty OR", slack)
                        if sum(t[2][0] for t in current_schedule_lpt[or_plan]) + surgery_info[2][0] + \
                                slack <= self.max_capacity:
                            # possible_ors.append(or_plan)
                            current_schedule_lpt[or_plan].append(surgery_info)
                            surgery_planned = True
                            break

                if not surgery_planned:
                    min_or_used = min(current_schedule_lpt, key=lambda k: sum(x[2][0] for x in current_schedule_lpt[k])
                                                                          + surgery_info[2][0]
                                                                          + self.slack_time_alternate(
                        current_schedule_lpt[k] + [surgery_info]) - self.max_capacity)
                    current_schedule_lpt[min_or_used].append(surgery_info)

                remaining_surgeries -= 1
                print(f"LPT: Iteration {len(self.waiting_list_mixture_surgeries_list) - remaining_surgeries}: "
                      f"{remaining_surgeries} surgeries remaining")

            total_overtime = 0
            total_free_time = 0
            for or_id, or_surgeries in current_schedule_lpt.items():
                # Calculate total surgery time and slack time for the OR
                total_time = sum(t[2][0] for t in or_surgeries) + self.slack_time_alternate(current_schedule_lpt[or_id])
                overtime = max(total_time - self.max_capacity, 0)
                free_time = max(self.max_capacity - total_time, 0)
                total_overtime += overtime
                total_free_time += free_time
            #     print(f'Overtime in OR {or_id} using LPT = {overtime}')
            #     print(f'Slack for OR {or_id} using LPT = {self.slack_time(or_surgeries)}')
            # print("total overtime", total_overtime)

            # for or_plan, surgeries in current_schedule_lpt.items():
            #     total_time = sum(surgery[2][0] for surgery in surgeries)
            #     total_slack = self.slack_time_alternate(surgeries)
            #     print(f"LPT: OR {or_plan} scheduled time: {total_time} hours, slack: {total_slack} hours")

        return current_schedule_lpt, total_overtime, total_free_time

    def slack_time(self, surgeries_for_slack):
        if not surgeries_for_slack:  # If empty, no slack
            return 0

        if isinstance(surgeries_for_slack[0], str):
            surgeries_for_slack = [surgeries_for_slack]
        else:
            pass

        tuple_map = tuple(tuple(sublist) for sublist in surgeries_for_slack)
        if tuple_map in self.slack_stored_original.keys():
            return self.slack_stored_original[tuple_map]

        # Calculate total mean and total standard deviation
        total_sigma_squared = sum((surgeries[2][1]) ** 2 for surgeries in surgeries_for_slack)
        total_sigma = np.sqrt(total_sigma_squared)

        # Calculate beta using the overtime probability
        beta = norm.ppf(1 - self.overtime_prob)  # This gets the B value for the normal distribution

        # Calculate the slack as beta * sqrt(sum(sigmas^2))
        slack = beta * total_sigma
        # print(f"beta: {beta}")
        # print(f"slack: {slack}")

        # Store the result in the cache
        self.slack_stored_original[tuple_map] = slack

        return slack

    def calculate_priority(self, current_surgery, possible_ors, schedule, slack_method):  # current_surgery: [k,surgery,(mu,sigma)]
        diff_per_or = []
        omega_per_or = {}

        if slack_method == "original":
            for ors in possible_ors:
                surgeries_with_current_surgery = []
                current_surgeries_in_or = schedule[ors]  # list of surgeries in specified OR

                if current_surgeries_in_or:
                    surgeries_with_current_surgery.extend(current_surgeries_in_or)
                    surgeries_with_current_surgery.append(current_surgery)
                    tuple_map1 = tuple(tuple(sublist) for sublist in surgeries_with_current_surgery)
                    tuple_map2 = tuple(tuple(sublist) for sublist in current_surgeries_in_or)
                    if tuple_map1 in self.slack_stored_original:
                        slack_with_surgery = self.slack_stored_original[tuple_map1]
                    else:
                        slack_with_surgery = self.slack_time(surgeries_with_current_surgery)
                        self.slack_stored_original[tuple_map1] = slack_with_surgery

                    if tuple_map2 in self.slack_stored_original:
                        slack_or = self.slack_stored_original[tuple_map1]
                    else:
                        slack_or = self.slack_time(current_surgeries_in_or)
                        self.slack_stored_original[tuple_map2] = slack_or

                    diff_or = slack_with_surgery - slack_or
                    diff_per_or.append(diff_or)
                    tuple_single_surgery = tuple(tuple(sublist) for sublist in current_surgery)
                    if tuple_single_surgery in self.slack_stored_original:
                        slack_single_surgery = self.slack_stored_original[tuple_single_surgery]
                    else:
                        slack_single_surgery = self.slack_time(current_surgery)
                        self.slack_stored_original[tuple_single_surgery] = slack_single_surgery

                    omega_or = slack_single_surgery - diff_or
                else:
                    omega_or = 0

                omega_per_or[ors] = omega_or

            best_or = max(omega_per_or, key=omega_per_or.get)
            priority_current_surgery = omega_per_or[best_or]

        else:
            for ors in possible_ors:
                surgeries_with_current_surgery = []
                current_surgeries_in_or = schedule[ors]  # list of surgeries in specified OR

                if current_surgeries_in_or:
                    surgeries_with_current_surgery.extend(current_surgeries_in_or)
                    surgeries_with_current_surgery.append(current_surgery)
                    tuple_map1 = tuple(tuple(sublist) for sublist in surgeries_with_current_surgery)
                    tuple_map2 = tuple(tuple(sublist) for sublist in current_surgeries_in_or)
                    if tuple_map1 in self.slack_stored_alternate:
                        slack_with_surgery = self.slack_stored_alternate[tuple_map1]
                    else:
                        slack_with_surgery = self.slack_time_alternate(surgeries_with_current_surgery)
                        self.slack_stored_alternate[tuple_map1] = slack_with_surgery

                    if tuple_map2 in self.slack_stored_original:
                        slack_or = self.slack_stored_alternate[tuple_map1]
                    else:
                        slack_or = self.slack_time_alternate(current_surgeries_in_or)
                        self.slack_stored_alternate[tuple_map2] = slack_or

                    diff_or = slack_with_surgery - slack_or
                    diff_per_or.append(diff_or)
                    tuple_single_surgery = tuple(tuple(sublist) for sublist in current_surgery)
                    if tuple_single_surgery in self.slack_stored_alternate:
                        slack_single_surgery = self.slack_stored_alternate[tuple_single_surgery]
                    else:
                        slack_single_surgery = self.slack_time_alternate(current_surgery)
                        self.slack_stored_alternate[tuple_single_surgery] = slack_single_surgery

                    omega_or = slack_single_surgery - diff_or

                else:
                    omega_or = 0

                omega_per_or[ors] = omega_or

            best_or = max(omega_per_or, key=omega_per_or.get)
            priority_current_surgery = omega_per_or[best_or]

        return current_surgery, priority_current_surgery, best_or

    def drawing_probabilities(self, priorities_list):
        all_priorities = [surgeries[1] for surgeries in priorities_list]
        all_regrets = [(surgeries[1] - min(all_priorities)) for surgeries in priorities_list]

        total = np.sum([(1 + w_i) ** self.alpha for w_i in all_regrets])
        probabilities = [(1 + regret) ** self.alpha / total for regret in all_regrets]

        return probabilities

    def regret_based_sampling(self, z, samples, slack_method):
        sorted_waiting_dict, sorted_waiting_list, sorted_ops = self.sort_waiting_list()
        best_schedule = None
        best_schedule_overtime = float("inf")
        final_slack = 0
        if slack_method == "alternate":
            for i in range(samples):
                print(f"Regret-based on sample: {i+1} (alternate slack)")
                current_schedule = {or_plan: [] for or_plan in self.ors}
                remaining_surgeries = sorted_waiting_list[:]  # create a copy of the sorted waiting list
                while remaining_surgeries:
                    print("REGRET: this surgeries left to schedule (alternate)", len(remaining_surgeries))
                    z_surgeries = remaining_surgeries[:min(z, len(remaining_surgeries))]  # take the next z surgeries

                    priorities_z_surgeries = []  # initialize priorities list for z surgeries
                    surgeries_to_draw = []
                    for surgery in z_surgeries:  # surgery = ["id", surgery, (mu,sigma)]
                        possible_ors = [or_id for or_id, surgeries in current_schedule.items() if not surgeries]
                        for or_plan, value in current_schedule.items():
                            surgeries_in_or = current_schedule[or_plan]
                            # if the OR is not empty:
                            if surgeries_in_or:
                                surgeries_or_with_surgery = surgeries_in_or + [surgery]  # find slack for all together
                                slack = self.slack_time_alternate(surgeries_or_with_surgery)
                                # print("calculated slack when nonempty OR", slack)
                                if sum(t[2][0] for t in current_schedule[or_plan]) + surgery[2][0] + \
                                        slack <= self.max_capacity:
                                    possible_ors.append(or_plan)

                        if possible_ors:
                            priority_surgery_i = self.calculate_priority(surgery, possible_ors, current_schedule,
                                                                         slack_method)
                            # priority_surgery_i = ["id",surgery,(mu,sigma)], priority, best_or
                            priorities_z_surgeries.append(priority_surgery_i)
                            surgeries_to_draw.append(priority_surgery_i[0])  # whole surgery info, ["id",surgery,(mu,sigma)]

                        else:
                            min_or_used = min(current_schedule.keys(), key=lambda k: sum(t[2][0] for t in
                                              current_schedule[k]) + surgery[2][0] + self.slack_time_alternate(
                                              current_schedule[k] + [surgery]) - self.max_capacity)
                            # print("calculated slack when no possible ORs without overtime",
                            #       self.slack_time(current_schedule[min_or_used] + [surgery]))
                            current_schedule[min_or_used].append(surgery)
                            remaining_surgeries.remove(surgery)  # remove the surgery from the remaining list
                            # print(f'Surgery {surgery} did not fit without overtime, so scheduled in {min_or_used}')

                    # print(priorities_z_surgeries)
                    if priorities_z_surgeries:  # if >1 surgery to consider, draw probability
                        probabilities = self.drawing_probabilities(priorities_z_surgeries)
                        drawn_surgery = random.choices(surgeries_to_draw, probabilities, k=1)[0]
                        # drawn_surgery: list of info of the surgery drawn
                        for i, (surgery, priority, best_or) in enumerate(priorities_z_surgeries):
                            if surgery == drawn_surgery:  # patient info (list)
                                current_schedule[best_or].append(surgery)
                                # print("drawn surgery", drawn_surgery_info)
                                # print(remaining_surgeries)
                                remaining_surgeries.remove(surgery)  # remove the scheduled surgery

                                break
                    # else:
                    #     current_schedule[priorities_z_surgeries[0][2]] = [priorities_z_surgeries[0][0]]
                    #     remaining_surgeries.remove(priorities_z_surgeries[0][0])

                total_overtime = 0
                total_slack = 0

                for or_id, or_surgeries in current_schedule.items():
                    overtime = max(sum(t[2][0] for t in or_surgeries) + self.slack_time_alternate(or_surgeries) -
                                   self.max_capacity, 0)

                    total_overtime += overtime
                    total_slack += self.slack_time_alternate(or_surgeries)
                    # print(f'Overtime in OR {or_id} using regret-based = {overtime}')
                    # print(f'Slack for OR {or_id} using regret-based = {self.slack_time(or_surgeries)}')
                # print("total overtime", total_overtime)

                # Update best schedule if overtime is less
                if total_overtime < best_schedule_overtime:
                    best_schedule = current_schedule
                    best_schedule_overtime = total_overtime
                    final_slack = total_slack
                #
                for or_plan, surgeries in best_schedule.items():
                    total_time = sum(surgery[2][0] for surgery in surgeries)
                    total_slack = self.slack_time_alternate(surgeries)
                    # print(f"OR {or_plan} scheduled time: {total_time} hours, slack: {total_slack} hours")
                #
        else:  # original
            for i in range(samples):
                print(f"Regret-based on sample: {i+1} (original slack)")
                current_schedule = {or_plan: [] for or_plan in self.ors}
                remaining_surgeries = sorted_waiting_list[:]  # create a copy of the sorted waiting list
                while remaining_surgeries:
                    print("REGRET: this surgeries left to schedule (original)", len(remaining_surgeries))
                    z_surgeries = remaining_surgeries[:min(z, len(remaining_surgeries))]  # take the next z surgeries

                    priorities_z_surgeries = []  # initialize priorities list for z surgeries
                    surgeries_to_draw = []
                    for surgery in z_surgeries:  # surgery = ["id", surgery, (mu,sigma)]
                        possible_ors = [or_id for or_id, surgeries in current_schedule.items() if not surgeries]
                        for or_plan, value in current_schedule.items():
                            surgeries_in_or = current_schedule[or_plan]
                            # if the OR is not empty:
                            if surgeries_in_or:
                                surgeries_or_with_surgery = surgeries_in_or + [surgery]  # find slack for all together
                                slack = self.slack_time(surgeries_or_with_surgery)
                                # print("calculated slack when nonempty OR", slack)
                                if sum(t[2][0] for t in current_schedule[or_plan]) + surgery[2][0] + \
                                        slack <= self.max_capacity:
                                    possible_ors.append(or_plan)

                        if possible_ors:
                            priority_surgery_i = self.calculate_priority(surgery, possible_ors, current_schedule,
                                                                         slack_method)
                            # priority_surgery_i = ["id",surgery,(mu,sigma)], priority, best_or
                            priorities_z_surgeries.append(priority_surgery_i)
                            surgeries_to_draw.append(
                                priority_surgery_i[0])  # whole surgery info, ["id",surgery,(mu,sigma)]

                        else:
                            min_or_used = min(current_schedule.keys(), key=lambda k: sum(t[2][0] for t in
                                                                                         current_schedule[k]) +
                                                                                     surgery[2][
                                                                                         0] + self.slack_time(
                                current_schedule[k]
                                + [surgery]) - self.max_capacity)
                            # print("calculated slack when no possible ORs without overtime",
                            #       self.slack_time(current_schedule[min_or_used] + [surgery]))
                            current_schedule[min_or_used].append(surgery)
                            remaining_surgeries.remove(surgery)  # remove the surgery from the remaining list
                            # print(f'Surgery {surgery} did not fit without overtime, so scheduled in {min_or_used}')

                    # print(priorities_z_surgeries)
                    if priorities_z_surgeries:  # if >1 surgery to consider, draw probability
                        probabilities = self.drawing_probabilities(priorities_z_surgeries)
                        drawn_surgery = random.choices(surgeries_to_draw, probabilities, k=1)[0]
                        # drawn_surgery: list of info of the surgery drawn
                        for i, (surgery, priority, best_or) in enumerate(priorities_z_surgeries):
                            if surgery == drawn_surgery:  # patient info (list)
                                current_schedule[best_or].append(surgery)
                                # print("drawn surgery", drawn_surgery_info)
                                # print(remaining_surgeries)
                                remaining_surgeries.remove(surgery)  # remove the scheduled surgery

                                break
                    # else:
                    #     current_schedule[priorities_z_surgeries[0][2]] = [priorities_z_surgeries[0][0]]
                    #     remaining_surgeries.remove(priorities_z_surgeries[0][0])

                total_overtime = 0
                total_slack = 0

                for or_id, or_surgeries in current_schedule.items():
                    overtime = max(
                        sum(t[2][0] for t in or_surgeries) + self.slack_time(or_surgeries) - self.max_capacity,
                        0)

                    total_overtime += overtime
                    # total_slack += self.slack_time(or_surgeries)
                    # print(f'Overtime in OR {or_id} using regret-based = {overtime}')
                    # print(f'Slack for OR {or_id} using regret-based = {self.slack_time(or_surgeries)}')
                # print("total overtime", total_overtime)

                # Update best schedule if overtime is less
                if total_overtime < best_schedule_overtime:
                    best_schedule = current_schedule
                    best_schedule_overtime = total_overtime
                    final_slack = total_slack

                # for or_plan, surgeries in best_schedule.items():
                #     total_time = sum(surgery[2][0] for surgery in surgeries)
                #     total_slack = self.slack_time(surgeries)
                #     print(f"OR {or_plan} scheduled time: {total_time} hours, slack: {total_slack} hours")

        return best_schedule, best_schedule_overtime

    def delta_expression(self, surgeries_for_slack, delta):
        if isinstance(surgeries_for_slack[0], str):
            surgeries_for_slack = [surgeries_for_slack]
        # print("surgeries for slack", surgeries_for_slack)
        weights_for_slack = [self.weights_dictionary[surgery[1]] for surgery in surgeries_for_slack]
        patient_id = [patient_info[0] for patient_info in surgeries_for_slack]
        # print("weights for slack:",weights_for_slack)
        total_mu = sum(self.waiting_list_mixture_surgeries[patient_info[0]][1][0] for patient_info in surgeries_for_slack)
        # print("total mu:", total_mu)
        mu_list, sigma_list = zip(*[zip(*self.waiting_list_individual_surgeries[ids][0]) for ids in patient_id])
        # print("mu_list", mu_list)
        # print("sigma_list", sigma_list)
        weights = np.array(weights_for_slack, dtype=object)
        mus = np.array(mu_list, dtype=object)
        sigmas = np.array(sigma_list, dtype=object)
        num_elements = [len(sublist) for sublist in weights_for_slack]
        # print("number of elements", num_elements)

        index_combinations = list(itertools.product(*[range(n) for n in num_elements]))
        # print("index combinations", index_combinations)
        selected_weights = [[weight[i] for i, weight in zip(indices, weights)] for indices in index_combinations]
        # print("selected weights", selected_weights)
        selected_mus = [[mu[i] for i, mu in zip(indices, mus)] for indices in index_combinations]
        # print("selected mus", selected_mus)
        selected_sigmas = [[sigma[i] for i, sigma in zip(indices, sigmas)] for indices in index_combinations]
        # print("selected sigmas", selected_sigmas)
        weight_products = np.prod(np.array(selected_weights), axis=1)
        # print("weight products", weight_products)
        mu_sums = np.sum(np.array(selected_mus), axis=1)
        # print("sum mus", mu_sums)
        sigma_sums = np.sum(np.array(selected_sigmas), axis=1)
        # print("sum sigmas", sigma_sums)
        cdf_arguments = (total_mu + delta - mu_sums) / sigma_sums
        cdf_values = norm.cdf(cdf_arguments)
        result = np.sum(weight_products * cdf_values)
        return result

    def to_solve(self, delta, surgeries_for_slack):
        return self.delta_expression(surgeries_for_slack, delta) - (1 - self.overtime_prob)

    def slack_time_alternate(self, surgeries_for_slack):
        if not surgeries_for_slack:
            return 0

        tuple_map = tuple(tuple(sublist) for sublist in surgeries_for_slack)
        if tuple_map in self.slack_stored_alternate:
            return self.slack_stored_alternate[tuple_map]

        solution_slack = fsolve(self.to_solve, x0=np.array([20]), args=surgeries_for_slack)
        if solution_slack[0] < 0:
            print("Slack is negative here")
        self.slack_stored_alternate[tuple_map] = abs(solution_slack[0])
        return abs(solution_slack[0])


def calculate_total_slack(schedule, surgery_scheduling, slack_method):
    total_slack = 0
    slack_to_add = 0
    for or_id, or_surgeries in schedule.items():
        if slack_method == 'original':
            slack_to_add = surgery_scheduling.slack_time(or_surgeries)
        elif slack_method == 'alternate':
            slack_to_add = surgery_scheduling.slack_time_alternate(or_surgeries)
        total_slack = total_slack + slack_to_add
    return total_slack


def calculate_total_free_time(schedule, surgery_scheduling, slack_method):
    total_free_time = 0
    total_overtime = 0
    total_slack_time = 0
    count_overtime = 0
    for or_id, or_surgeries in schedule.items():
        total_scheduled_time = sum(surgery[2][0] for surgery in or_surgeries)
        if slack_method == 'original':
            total_slack_time = surgery_scheduling.slack_time(or_surgeries)
        elif slack_method == 'alternate':
            total_slack_time = surgery_scheduling.slack_time_alternate(or_surgeries)
        total_or_time = total_scheduled_time + total_slack_time
        total_free_time += max(0, surgery_scheduling.max_capacity - total_or_time)
        if total_or_time > surgery_scheduling.max_capacity:
            total_overtime += (total_or_time - surgery_scheduling.max_capacity)
            count_overtime += 1
    return total_free_time, total_overtime, count_overtime


def calculate_free_days(schedule):
    total_free_days = 0
    for or_plan, surgeries in schedule.items():
        if not surgeries:
            total_free_days += 1
    return total_free_days


def run_first_fit(surgery_scheduling, slack_method):
    start_time3 = time.time()
    ff_schedule, ff_overtime = surgery_scheduling.first_fit(slack_method)
    end_time3 = time.time()
    ff_time = end_time3 - start_time3
    print("ff finished, took:", ff_time, "minutes")
    total_slack_ff = calculate_total_slack(ff_schedule, surgery_scheduling, slack_method)
    total_free_time_ff = calculate_total_free_time(ff_schedule, surgery_scheduling, slack_method)
    total_free_days_ff = calculate_free_days(ff_schedule)
    return ff_schedule, ff_overtime, ff_time, total_slack_ff, total_free_time_ff, total_free_days_ff


def run_lpt(surgery_scheduling, slack_method):
    start_time2 = time.time()
    lpt_schedule, overtime_lpt, free_time_lpt = surgery_scheduling.lpt(slack_method)
    end_time2 = time.time()
    lpt_time = end_time2 - start_time2
    print("lpt finished, took:", lpt_time, "minutes")
    total_slack_lpt = calculate_total_slack(lpt_schedule, surgery_scheduling, slack_method)
    total_free_time_lpt = calculate_total_free_time(lpt_schedule, surgery_scheduling, slack_method)
    total_free_days_lpt = calculate_free_days(lpt_schedule)
    return lpt_schedule, overtime_lpt, lpt_time, total_slack_lpt, total_free_time_lpt, free_time_lpt, total_free_days_lpt


def run_regret_based(surgery_scheduling, z, samples, slack_method):
    start_time1 = time.time()
    best_schedule_output, best_schedule_overtime_output = surgery_scheduling.regret_based_sampling(z=z, samples=samples, slack_method=slack_method)
    end_time1 = time.time()
    regret_based_sampling_time = end_time1 - start_time1
    print('regret finished, took:', regret_based_sampling_time, "minutes")
    total_slack_regret_based = calculate_total_slack(best_schedule_output, surgery_scheduling, slack_method)
    total_free_time_regret_based = calculate_total_free_time(best_schedule_output, surgery_scheduling, slack_method)
    total_free_days_regret = calculate_free_days(best_schedule_output)
    return best_schedule_output, best_schedule_overtime_output, regret_based_sampling_time, total_slack_regret_based, total_free_time_regret_based, total_free_days_regret


if __name__ == '__main__':
    surgery_scheduling = SurgeryScheduling(number_surgeries=400, distr="normal", number_of_ors=3, alpha=50)

    # debug_part = run_regret_based(surgery_scheduling, 5, 5, "alternate")

    # individual_waiting_5 = pickle.load(open('individual_params_5_surgeries.pickle', 'rb'))
    # loaded_waiting_5 = pickle.load(open('waiting_list_5_surgeries.pickle', 'rb'))
    # print('stored', individual_waiting_5)
    # print(loaded_waiting_5)

    print(f'from called function waiting list: {surgery_scheduling.waiting_list_mixture_surgeries}')
    print("individual waiting list:", surgery_scheduling.waiting_list_individual_surgeries)

    # waiting_list_400_surgeries_multi = surgery_scheduling.waiting_list_mixture_surgeries
    # waiting_individual_400_surgeries_multi = surgery_scheduling.waiting_list_individual_surgeries

    # with open('waiting_list_400_surgeries_multi.pickle', 'wb') as handle:
    #     pickle.dump(waiting_list_400_surgeries_multi, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #
    # with open('waiting_individual_400_surgeries_multi.pickle', 'wb') as handle:
    #     pickle.dump(waiting_individual_400_surgeries_multi, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # waiting_list_400_surgeries_multi_stored = pickle.load(open('waiting_list_400_surgeries_multi.pickle', 'rb'))
    # waiting_individual_400_surgeries_multi_stored = pickle.load(open('waiting_individual_400_surgeries_multi.pickle', 'rb'))

    # Verifying the loaded dictionary
    # print("stored waiting list:", waiting_list_500_surgeries_stored)
    # print("stored 400:", waiting_list_400_surgeries_multi_stored)
    # print("individual stored 400:", waiting_individual_400_surgeries_multi_stored)


    or_capacity = surgery_scheduling.max_capacity * surgery_scheduling.number_of_ors * \
                  surgery_scheduling.number_working_days

    print(f"Total surgery time: {surgery_scheduling.total_surgery_time()/60} hours")
    print(f"Total OR capacity: {or_capacity/60} hours")

    with Pool(cpu_count()) as pool:
    #     # Run first_fit with original slack method
    #     ff_results_original = pool.apply_async(run_first_fit, (surgery_scheduling, 'original'))
    #     # # Run first_fit with alternate slack method
    #     ff_results_alternate =pool.apply_async(run_first_fit, (surgery_scheduling, 'alternate'))
    #     #
    #     # # Run lpt with original slack method
    #     lpt_results_original = pool.apply_async(run_lpt, (surgery_scheduling, 'original'))
    #     # # Run lpt with alternate slack method
    #     lpt_results_alternate = pool.apply_async(run_lpt, (surgery_scheduling, 'alternate'))
    #
        # Run regret_based_sampling with original slack method
        # regret_results_original = pool.apply_async(run_regret_based, (surgery_scheduling, 3, 5, 'original'))
        # Run regret_based_sampling with alternate slack method
        # regret_results_alternate = pool.apply_async(run_regret_based, (surgery_scheduling, 3, 5, 'alternate'))

    #     # Get results
    #     ff_schedule_original, ff_overtime_original, ff_time_original, total_slack_ff_original, total_free_time_ff_original, total_free_days_ff_original = ff_results_original.get()
    #     ff_schedule_alternate, ff_overtime_alternate, ff_time_alternate, total_slack_ff_alternate, total_free_time_ff_alternate, total_free_days_ff_alternate = ff_results_alternate.get()
    #
    #     lpt_schedule_original, overtime_lpt_original, lpt_time_original, total_slack_lpt_original, total_free_time_lpt_original, free_time_lpt_original, total_free_days_lpt_original = lpt_results_original.get()
    #     lpt_schedule_alternate, overtime_lpt_alternate, lpt_time_alternate, total_slack_lpt_alternate, total_free_time_lpt_alternate, free_time_lpt_alternate, total_free_days_lpt_alternate = lpt_results_alternate.get()
    #
        # best_schedule_output_original, best_schedule_overtime_output_original, regret_based_sampling_time_original, total_slack_regret_based_original, \
        # total_free_time_regret_based_original, total_free_days_regret_original = regret_results_original.get()
        # best_schedule_output_alternate, best_schedule_overtime_output_alternate, regret_based_sampling_time_alternate, \
        # total_slack_regret_based_alternate, total_free_time_regret_based_alternate, total_free_days_regret_alternate = regret_results_alternate.get()

        regret_results_alternate7 = pool.apply_async(run_regret_based, (surgery_scheduling, 50, 10, 'alternate'))
        regret_results_alternate10 = pool.apply_async(run_regret_based, (surgery_scheduling, 50, 15, 'alternate'))
        regret_results_alternate15 = pool.apply_async(run_regret_based, (surgery_scheduling, 50, 35, 'alternate'))
        # regret_results_alternate25 = pool.apply_async(run_regret_based, (surgery_scheduling, 50, 100, 'alternate'))
        # regret_results_alternate50 = pool.apply_async(run_regret_based, (surgery_scheduling, 50, 150, 'alternate'))

        best_schedule_output_alternate7, best_schedule_overtime_output_alternate7, regret_based_sampling_time_alternate7, \
        total_slack_regret_based_alternate7, total_free_time_regret_based_alternate7, total_free_days_regret_alternate7 = regret_results_alternate7.get()

        best_schedule_output_alternate10, best_schedule_overtime_output_alternate10, regret_based_sampling_time_alternate10, \
        total_slack_regret_based_alternate10, total_free_time_regret_based_alternate10, total_free_days_regret_alternate10 = regret_results_alternate10.get()

        best_schedule_output_alternate15, best_schedule_overtime_output_alternate15, regret_based_sampling_time_alternate15, \
        total_slack_regret_based_alternate15, total_free_time_regret_based_alternate15, total_free_days_regret_alternate15 = regret_results_alternate15.get()

        # best_schedule_output_alternate25, best_schedule_overtime_output_alternate25, regret_based_sampling_time_alternate25, \
        # total_slack_regret_based_alternate25, total_free_time_regret_based_alternate25, total_free_days_regret_alternate25 = regret_results_alternate25.get()
        #
        # best_schedule_output_alternate50, best_schedule_overtime_output_alternate50, regret_based_sampling_time_alternate50, \
        # total_slack_regret_based_alternate50, total_free_time_regret_based_alternate50, total_free_days_regret_alternate50 = regret_results_alternate50.get()

    # # Print results for original slack method
    # print("RESULTS ORIGINAL SLACK")
    # print("Parameter setting regret based: z=5, samples=5, ORs=3, alpha=100, 400 surgeries, 3 months (5 working days): 60 days, overtime probability=0.3")
    # # print("Best schedule using ff:", ff_schedule_original)
    # # print("Best schedule using lpt:", lpt_schedule_original)
    # print("Best schedule using regret based:", best_schedule_output_original)
    # #
    # # print(f"Total overtime using ff: {ff_overtime_original / 60} hours")
    # # print(f"Total overtime for ff (in free time function): {total_free_time_ff_original[1] / 60} hours")
    # # print(f"Total overtime using lpt: {overtime_lpt_original / 60} hours")
    # # print(f"Total overtime for lpt (in free time function): {total_free_time_lpt_original[1] / 60} hours")
    # print(f"Total overtime using regret based: {best_schedule_overtime_output_original / 60} hours")
    # print(f"Total overtime for regret based (in free time function): {total_free_time_regret_based_original[1] / 60} hours")
    # #
    # # print(f"Overtime occurred in {total_free_time_ff_original[2]} OR-plans for ff")
    # # print(f"Overtime occurred in {total_free_time_lpt_original[2]} OR-plans for lpt")
    # print(f"Overtime occurred in {total_free_time_regret_based_original[2]} OR-plans for regret")
    # #
    # # print(f"Total free time for ff: {total_free_time_ff_original[0] / 60} hours")
    # # print(f"Total free time for lpt: {total_free_time_lpt_original[0] / 60} hours")
    # # print(f'Total free time lpt within lpt function: {free_time_lpt_original / 60} hours')
    # print(f"Total free time for regret based: {total_free_time_regret_based_original[0] / 60} hours")
    # #
    # # print(f"Total running time for ff: {ff_time_original / 60} minutes")
    # # print(f"Total running time for lpt: {lpt_time_original / 60} minutes")
    # print(f"Total running time for regret based: {regret_based_sampling_time_original / 60} minutes")
    # #
    # # print(f"Total planned slack for ff: {total_slack_ff_original / 60} hours")
    # # print(f"Total planned slack for lpt: {total_slack_lpt_original / 60} hours")
    # print(f"Total planned slack for regret based: {total_slack_regret_based_original / 60} hours")
    # #
    # # print(f"Total free OR-plans for ff: {total_free_days_ff_original} days")
    # # print(f"Total free OR-plans for lpt: {total_free_days_lpt_original} days")
    # print(f"Total free OR-plans for regret_based: {total_free_days_regret_original} days")
    # #
    # # # Print results for alternate slack method
    # print("RESULTS ALTERNATE SLACK (Z=7)")
    # # # print("Parameter setting regret based: z=9, samples=10, ORs=3, alpha=10, 500 surgeries, 3 months (5 working days)")
    # # print("Best schedule using ff:", ff_schedule_alternate)
    # # print("Best schedule using lpt:", lpt_schedule_alternate)
    # print("Best schedule using regret based:", best_schedule_output_alternate)
    # #
    # # print(f"Total overtime using ff: {ff_overtime_alternate / 60} hours")
    # # print(f"Total overtime for ff (in free time function): {total_free_time_ff_alternate[1] / 60} hours")
    # # print(f"Total overtime using lpt: {overtime_lpt_alternate / 60} hours")
    # # print(f"Total overtime for lpt (in free time function): {total_free_time_lpt_alternate[1] / 60} hours")
    # print(f"Total overtime using regret based: {best_schedule_overtime_output_alternate / 60} hours")
    # print(f"Total overtime for regret based (in free time function): {total_free_time_regret_based_alternate[1] / 60} hours")
    #
    # # print(f"Overtime occurred in {total_free_time_ff_alternate[2]} OR-plans for ff")
    # # print(f"Overtime occurred in {total_free_time_lpt_alternate[2]} OR-plans for lpt")
    # print(f"Overtime occurred in {total_free_time_regret_based_alternate[2]} OR-plans for lpt")
    # #
    # # print(f"Total free time for ff: {total_free_time_ff_alternate[0] / 60} hours")
    # # print(f"Total free time for lpt: {total_free_time_lpt_alternate[0] / 60} hours")
    # # print(f'Total free time lpt within lpt function: {free_time_lpt_alternate / 60} hours')
    # print(f"Total free time for regret based: {total_free_time_regret_based_alternate[0] / 60} hours")
    # #
    # # print(f"Total running time for ff: {ff_time_alternate / 60} minutes")
    # # print(f"Total running time for lpt: {lpt_time_alternate / 60} minutes")
    # print(f"Total running time for regret based: {regret_based_sampling_time_alternate / 60} minutes")
    # #
    # # print(f"Total planned slack for ff: {total_slack_ff_alternate / 60} hours")
    # # print(f"Total planned slack for lpt: {total_slack_lpt_alternate / 60} hours")
    # print(f"Total planned slack for regret based: {total_slack_regret_based_alternate / 60} hours")
    # #
    # # print(f"Total free OR-plans for ff: {total_free_days_ff_alternate} days")
    # # print(f"Total free OR-plans for lpt: {total_free_days_lpt_alternate} days")
    # print(f"Total free OR-plans for regret_based: {total_free_days_regret_alternate} days")
    #
    print("samples=10:")

    print("Best schedule using regret based:", best_schedule_output_alternate7)
    #
    # print(f"Total overtime using ff: {ff_overtime_alternate / 60} hours")
    # print(f"Total overtime for ff (in free time function): {total_free_time_ff_alternate[1] / 60} hours")
    # print(f"Total overtime using lpt: {overtime_lpt_alternate / 60} hours")
    # print(f"Total overtime for lpt (in free time function): {total_free_time_lpt_alternate[1] / 60} hours")
    print(f"Total overtime using regret based: {best_schedule_overtime_output_alternate7 / 60} hours")
    print(f"Total overtime for regret based (in free time function): {total_free_time_regret_based_alternate7[1] / 60} hours")

    # print(f"Overtime occurred in {total_free_time_ff_alternate[2]} OR-plans for ff")
    # print(f"Overtime occurred in {total_free_time_lpt_alternate[2]} OR-plans for lpt")
    print(f"Overtime occurred in {total_free_time_regret_based_alternate7[2]} OR-plans for lpt")
    #
    # print(f"Total free time for ff: {total_free_time_ff_alternate[0] / 60} hours")
    # print(f"Total free time for lpt: {total_free_time_lpt_alternate[0] / 60} hours")
    # print(f'Total free time lpt within lpt function: {free_time_lpt_alternate / 60} hours')
    print(f"Total free time for regret based: {total_free_time_regret_based_alternate7[0] / 60} hours")
    #
    # print(f"Total running time for ff: {ff_time_alternate / 60} minutes")
    # print(f"Total running time for lpt: {lpt_time_alternate / 60} minutes")
    print(f"Total running time for regret based: {regret_based_sampling_time_alternate7 / 60} minutes")
    #
    # print(f"Total planned slack for ff: {total_slack_ff_alternate / 60} hours")
    # print(f"Total planned slack for lpt: {total_slack_lpt_alternate / 60} hours")
    print(f"Total planned slack for regret based: {total_slack_regret_based_alternate7 / 60} hours")
    #
    # print(f"Total free OR-plans for ff: {total_free_days_ff_alternate} days")
    # print(f"Total free OR-plans for lpt: {total_free_days_lpt_alternate} days")
    print(f"Total free OR-plans for regret_based: {total_free_days_regret_alternate7} days")

    print("samples=15:")

    print("Best schedule using regret based:", best_schedule_output_alternate10)
    #
    # print(f"Total overtime using ff: {ff_overtime_alternate / 60} hours")
    # print(f"Total overtime for ff (in free time function): {total_free_time_ff_alternate[1] / 60} hours")
    # print(f"Total overtime using lpt: {overtime_lpt_alternate / 60} hours")
    # print(f"Total overtime for lpt (in free time function): {total_free_time_lpt_alternate[1] / 60} hours")
    print(f"Total overtime using regret based: {best_schedule_overtime_output_alternate10 / 60} hours")
    print(
        f"Total overtime for regret based (in free time function): {total_free_time_regret_based_alternate10[1] / 60} hours")

    # print(f"Overtime occurred in {total_free_time_ff_alternate[2]} OR-plans for ff")
    # print(f"Overtime occurred in {total_free_time_lpt_alternate[2]} OR-plans for lpt")
    print(f"Overtime occurred in {total_free_time_regret_based_alternate10[2]} OR-plans for lpt")
    #
    # print(f"Total free time for ff: {total_free_time_ff_alternate[0] / 60} hours")
    # print(f"Total free time for lpt: {total_free_time_lpt_alternate[0] / 60} hours")
    # print(f'Total free time lpt within lpt function: {free_time_lpt_alternate / 60} hours')
    print(f"Total free time for regret based: {total_free_time_regret_based_alternate10[0] / 60} hours")
    #
    # print(f"Total running time for ff: {ff_time_alternate / 60} minutes")
    # print(f"Total running time for lpt: {lpt_time_alternate / 60} minutes")
    print(f"Total running time for regret based: {regret_based_sampling_time_alternate10 / 60} minutes")
    #
    # print(f"Total planned slack for ff: {total_slack_ff_alternate / 60} hours")
    # print(f"Total planned slack for lpt: {total_slack_lpt_alternate / 60} hours")
    print(f"Total planned slack for regret based: {total_slack_regret_based_alternate10/ 60} hours")
    #
    # print(f"Total free OR-plans for ff: {total_free_days_ff_alternate} days")
    # print(f"Total free OR-plans for lpt: {total_free_days_lpt_alternate} days")
    print(f"Total free OR-plans for regret_based: {total_free_days_regret_alternate10} days")

    print("samples = 35:")

    print("Best schedule using regret based:", best_schedule_output_alternate15)
    #
    # print(f"Total overtime using ff: {ff_overtime_alternate / 60} hours")
    # print(f"Total overtime for ff (in free time function): {total_free_time_ff_alternate[1] / 60} hours")
    # print(f"Total overtime using lpt: {overtime_lpt_alternate / 60} hours")
    # print(f"Total overtime for lpt (in free time function): {total_free_time_lpt_alternate[1] / 60} hours")
    print(f"Total overtime using regret based: {best_schedule_overtime_output_alternate15/ 60} hours")
    print(
        f"Total overtime for regret based (in free time function): {total_free_time_regret_based_alternate15[1] / 60} hours")

    # print(f"Overtime occurred in {total_free_time_ff_alternate[2]} OR-plans for ff")
    # print(f"Overtime occurred in {total_free_time_lpt_alternate[2]} OR-plans for lpt")
    print(f"Overtime occurred in {total_free_time_regret_based_alternate15[2]} OR-plans for lpt")
    #
    # print(f"Total free time for ff: {total_free_time_ff_alternate[0] / 60} hours")
    # print(f"Total free time for lpt: {total_free_time_lpt_alternate[0] / 60} hours")
    # print(f'Total free time lpt within lpt function: {free_time_lpt_alternate / 60} hours')
    print(f"Total free time for regret based: {total_free_time_regret_based_alternate15[0] / 60} hours")
    #
    # print(f"Total running time for ff: {ff_time_alternate / 60} minutes")
    # print(f"Total running time for lpt: {lpt_time_alternate / 60} minutes")
    print(f"Total running time for regret based: {regret_based_sampling_time_alternate15/ 60} minutes")
    #
    # print(f"Total planned slack for ff: {total_slack_ff_alternate / 60} hours")
    # print(f"Total planned slack for lpt: {total_slack_lpt_alternate / 60} hours")
    print(f"Total planned slack for regret based: {total_slack_regret_based_alternate15/ 60} hours")
    #
    # print(f"Total free OR-plans for ff: {total_free_days_ff_alternate} days")
    # print(f"Total free OR-plans for lpt: {total_free_days_lpt_alternate} days")
    print(f"Total free OR-plans for regret_based: {total_free_days_regret_alternate15} days")

    # print("samples = 100:")
    #
    # print("Best schedule using regret based:", best_schedule_output_alternate25)
    # #
    # # print(f"Total overtime using ff: {ff_overtime_alternate / 60} hours")
    # # print(f"Total overtime for ff (in free time function): {total_free_time_ff_alternate[1] / 60} hours")
    # # print(f"Total overtime using lpt: {overtime_lpt_alternate / 60} hours")
    # # print(f"Total overtime for lpt (in free time function): {total_free_time_lpt_alternate[1] / 60} hours")
    # print(f"Total overtime using regret based: {best_schedule_overtime_output_alternate25 / 60} hours")
    # print(
    #     f"Total overtime for regret based (in free time function): {total_free_time_regret_based_alternate25[1] / 60} hours")
    #
    # # print(f"Overtime occurred in {total_free_time_ff_alternate[2]} OR-plans for ff")
    # # print(f"Overtime occurred in {total_free_time_lpt_alternate[2]} OR-plans for lpt")
    # print(f"Overtime occurred in {total_free_time_regret_based_alternate25[2]} OR-plans for lpt")
    # #
    # # print(f"Total free time for ff: {total_free_time_ff_alternate[0] / 60} hours")
    # # print(f"Total free time for lpt: {total_free_time_lpt_alternate[0] / 60} hours")
    # # print(f'Total free time lpt within lpt function: {free_time_lpt_alternate / 60} hours')
    # print(f"Total free time for regret based: {total_free_time_regret_based_alternate25[0] / 60} hours")
    # #
    # # print(f"Total running time for ff: {ff_time_alternate / 60} minutes")
    # # print(f"Total running time for lpt: {lpt_time_alternate / 60} minutes")
    # print(f"Total running time for regret based: {regret_based_sampling_time_alternate25 / 60} minutes")
    # #
    # # print(f"Total planned slack for ff: {total_slack_ff_alternate / 60} hours")
    # # print(f"Total planned slack for lpt: {total_slack_lpt_alternate / 60} hours")
    # print(f"Total planned slack for regret based: {total_slack_regret_based_alternate25 / 60} hours")
    # #
    # # print(f"Total free OR-plans for ff: {total_free_days_ff_alternate} days")
    # # print(f"Total free OR-plans for lpt: {total_free_days_lpt_alternate} days")
    # print(f"Total free OR-plans for regret_based: {total_free_days_regret_alternate25} days")
    #
    #
    # print("samples = 150:")
    #
    # print("Best schedule using regret based:", best_schedule_output_alternate50)
    # #
    # # print(f"Total overtime using ff: {ff_overtime_alternate / 60} hours")
    # # print(f"Total overtime for ff (in free time function): {total_free_time_ff_alternate[1] / 60} hours")
    # # print(f"Total overtime using lpt: {overtime_lpt_alternate / 60} hours")
    # # print(f"Total overtime for lpt (in free time function): {total_free_time_lpt_alternate[1] / 60} hours")
    # print(f"Total overtime using regret based: {best_schedule_overtime_output_alternate50 / 60} hours")
    # print(
    #     f"Total overtime for regret based (in free time function): {total_free_time_regret_based_alternate50[1] / 60} hours")
    #
    # # print(f"Overtime occurred in {total_free_time_ff_alternate[2]} OR-plans for ff")
    # # print(f"Overtime occurred in {total_free_time_lpt_alternate[2]} OR-plans for lpt")
    # print(f"Overtime occurred in {total_free_time_regret_based_alternate50[2]} OR-plans for lpt")
    # #
    # # print(f"Total free time for ff: {total_free_time_ff_alternate[0] / 60} hours")
    # # print(f"Total free time for lpt: {total_free_time_lpt_alternate[0] / 60} hours")
    # # print(f'Total free time lpt within lpt function: {free_time_lpt_alternate / 60} hours')
    # print(f"Total free time for regret based: {total_free_time_regret_based_alternate50[0] / 60} hours")
    # #
    # # print(f"Total running time for ff: {ff_time_alternate / 60} minutes")
    # # print(f"Total running time for lpt: {lpt_time_alternate / 60} minutes")
    # print(f"Total running time for regret based: {regret_based_sampling_time_alternate50/ 60} minutes")
    # #
    # # print(f"Total planned slack for ff: {total_slack_ff_alternate / 60} hours")
    # # print(f"Total planned slack for lpt: {total_slack_lpt_alternate / 60} hours")
    # print(f"Total planned slack for regret based: {total_slack_regret_based_alternate50 / 60} hours")
    # #
    # # print(f"Total free OR-plans for ff: {total_free_days_ff_alternate} days")
    # # print(f"Total free OR-plans for lpt: {total_free_days_lpt_alternate} days")
    # print(f"Total free OR-plans for regret_based: {total_free_days_regret_alternate50} days")