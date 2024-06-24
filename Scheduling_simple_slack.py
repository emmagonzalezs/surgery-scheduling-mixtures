from WaitingList import waiting_list
from scipy.stats import norm
from scipy.optimize import fsolve
import numpy as np
import pickle
import random
import itertools
import time

class SurgeryScheduling:
    def __init__(self, number_surgeries, distr, number_of_ors):
        self.number_surgeries = number_surgeries
        self.distr = distr
        self.number_of_ors = number_of_ors
        # Initialize waiting list to schedule in ORs
        self.waiting = waiting_list(number_surgeries, distr)
        # print(waiting[0])
        self.waiting_list_mixture_surgeries = self.waiting[0]  # {'1': ["general code", (mu,sigma)]}
        self.waiting_list_mixture_surgeries_list = [[key, *value] for key, value in
                                                    self.waiting_list_mixture_surgeries.items()]
        self.waiting_list_individual_surgeries = self.waiting[1]  # {'1': [(mu11,sigma11), (mu12, sigma12)]}

        # Weights per surgery in waiting list, for delta calculations
        self.weights_dictionary = pickle.load(open('weights_for_mixture.pkl', 'rb'))  # {"general code": [w11,w12,w13]}
        # self.weights_surgeries_waiting_list = [self.weights_dictionary[key] for key in
        #                                        self.waiting_list_mixture_surgeries if key in self.weights_dictionary]
        self.weights_surgeries_waiting_list = [self.weights_dictionary[value[0]] for key, value in
                                               self.waiting_list_mixture_surgeries.items() if
                                               value[0] in self.weights_dictionary]  # surgery in position 0
        # weights_surgeries_waiting_list = [[w11,w12,w13], [w21,w22], ...]

        # self.ors = {i: [] for i in range(1, number_of_ors + 1)}  # {"1":[], "2":[]}
        self.number_working_days = 1
        self.ors = {(k, t): [] for t in range(1, self.number_working_days + 1) for k in range(1, number_of_ors + 1)}
        # self.ors = {(OR, day):} for 1 year, append here ['surgery'
        self.max_capacity = 450  # per OR in minutes (8h)
        self.list_of_ors = [i for i in range(1, number_of_ors + 1)]  # [1,2,..., number_of_ors]
        self.list_of_ors = [(k, t) for t in range(1, self.number_working_days + 1) for k in range(1, number_of_ors + 1)]
        self.overtime_prob = 0.3
        self.alpha = 10
        self.slack_stored = {}

    def sort_waiting_list(self):
        ordered_waiting_list_dict = dict(sorted(self.waiting_list_mixture_surgeries.items(), key=lambda item:
                                                item[1][1][0], reverse=True))
        ordered_waiting_list_list = [[k, *v] for k, v in ordered_waiting_list_dict.items()]

        sorted_surgeries = list(ordered_waiting_list_dict.keys())

        return ordered_waiting_list_dict, ordered_waiting_list_list, sorted_surgeries

    def total_surgery_time(self):
        total_time = sum(surgery[2][0] for surgery in self.waiting_list_mixture_surgeries_list)
        return total_time

    def first_fit(self):  # surgeries not ordered
        current_schedule_ff = {or_plan: [] for or_plan in self.ors}
        remaining_surgeries = len(self.waiting_list_mixture_surgeries_list)
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
                                  + self.slack_time(current_schedule_ff[k] + [surgery_info]) - self.max_capacity)
                current_schedule_ff[min_or_used].append(surgery_info)

            remaining_surgeries -= 1
            print(f"FF: Iteration {len(self.waiting_list_mixture_surgeries_list) - remaining_surgeries}: "
                  f"{remaining_surgeries} surgeries remaining")

        total_overtime = 0
        for or_id, or_surgeries in current_schedule_ff.items():
            total_time = sum(t[2][0] for t in or_surgeries) + self.slack_time(or_surgeries)
            overtime = max(total_time - self.max_capacity, 0)
            total_overtime += overtime

        for or_plan, surgeries in current_schedule_ff.items():
            total_time = sum(surgery[2][0] for surgery in surgeries)
            total_slack = self.slack_time(surgeries)
            print(f"OR {or_plan} scheduled time: {total_time} hours, slack: {total_slack} hours")

        return current_schedule_ff, total_overtime

    def lpt(self):
        _, ordered_waiting_list, _ = self.sort_waiting_list()
        current_schedule_lpt = {or_plan: [] for or_plan in self.ors}
        remaining_surgeries = len(ordered_waiting_list)

        for surgery_info in ordered_waiting_list:
            possible_ors = [or_id for or_id, surgeries in current_schedule_lpt.items() if not surgeries]
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
                                  + self.slack_time(current_schedule_lpt[k] + [surgery_info]) - self.max_capacity)
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

        for or_plan, surgeries in current_schedule_lpt.items():
            total_time = sum(surgery[2][0] for surgery in surgeries)
            total_slack = self.slack_time(surgeries)
            print(f"OR {or_plan} scheduled time: {total_time} hours, slack: {total_slack} hours")

        return current_schedule_lpt, total_overtime, total_free_time

    def slack_time(self, surgeries_for_slack):
        if not surgeries_for_slack:  # If empty, no slack
            return 0

        if isinstance(surgeries_for_slack[0], str):
            surgeries_for_slack = [surgeries_for_slack]
        else:
            pass

        tuple_map = tuple(tuple(sublist) for sublist in surgeries_for_slack)
        if tuple_map in self.slack_stored.keys():
            return self.slack_stored[tuple_map]

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
        self.slack_stored[tuple_map] = slack

        return slack

    def calculate_priority(self, current_surgery, possible_ors, schedule):  # current_surgery: [k,surgery,(mu,sigma)]
        diff_per_or = []
        omega_per_or = {}

        for ors in possible_ors:
            surgeries_with_current_surgery = []
            current_surgeries_in_or = schedule[ors]  # list of surgeries in specified OR

            if current_surgeries_in_or:
                surgeries_with_current_surgery.extend(current_surgeries_in_or)
                surgeries_with_current_surgery.append(current_surgery)
                diff_or = self.slack_time(surgeries_with_current_surgery) - self.slack_time(current_surgeries_in_or)
                diff_per_or.append(diff_or)
                omega_or = self.slack_time(current_surgery) - diff_or
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

    def regret_based_sampling(self, z, samples):
        sorted_waiting_dict, sorted_waiting_list, sorted_ops = self.sort_waiting_list()
        best_schedule = None
        best_schedule_overtime = float("inf")
        final_slack = 0

        for _ in range(samples):
            current_schedule = {or_plan: [] for or_plan in self.ors}
            remaining_surgeries = sorted_waiting_list[:]  # create a copy of the sorted waiting list
            while remaining_surgeries:
                print("REGRET: this surgeries left", len(remaining_surgeries))
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
                        priority_surgery_i = self.calculate_priority(surgery, possible_ors, current_schedule)
                        # priority_surgery_i = ["id",surgery,(mu,sigma)], priority, best_or
                        priorities_z_surgeries.append(priority_surgery_i)
                        surgeries_to_draw.append(priority_surgery_i[0])  # whole surgery info, ["id",surgery,(mu,sigma)]

                    else:
                        min_or_used = min(current_schedule.keys(), key=lambda k: sum(t[2][0] for t in
                                          current_schedule[k]) + surgery[2][0] + self.slack_time(current_schedule[k]
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
                overtime = max(sum(t[2][0] for t in or_surgeries) + self.slack_time(or_surgeries) - self.max_capacity,
                               0)

                total_overtime += overtime
                total_slack += self.slack_time(or_surgeries)
                # print(f'Overtime in OR {or_id} using regret-based = {overtime}')
                # print(f'Slack for OR {or_id} using regret-based = {self.slack_time(or_surgeries)}')
            # print("total overtime", total_overtime)

            # Update best schedule if overtime is less
            if total_overtime < best_schedule_overtime:
                best_schedule = current_schedule
                best_schedule_overtime = total_overtime
                final_slack = total_slack

            for or_plan, surgeries in best_schedule.items():
                total_time = sum(surgery[2][0] for surgery in surgeries)
                total_slack = self.slack_time(surgeries)
                print(f"OR {or_plan} scheduled time: {total_time} hours, slack: {total_slack} hours")

        return best_schedule, best_schedule_overtime


surgery_scheduling = SurgeryScheduling(number_surgeries=10, distr="normal", number_of_ors=4)

start_time3 = time.time()
ff_schedule, ff_overtime = surgery_scheduling.first_fit()
end_time3 = time.time()
ff_time = end_time3 - start_time3
print("ff finished, took:", ff_time)

start_time2 = time.time()
lpt_schedule, overtime_lpt, free_time_lpt = surgery_scheduling.lpt()
end_time2 = time.time()
lpt_time = end_time2 - start_time2
print("lpt finished, took:", lpt_time)

start_time1 = time.time()
best_schedule_output, best_schedule_overtime_output = surgery_scheduling.regret_based_sampling(z=4, samples=1)
end_time1 = time.time()
regret_based_sampling_time = end_time1 - start_time1
print('regret finished, took:', regret_based_sampling_time)


def calculate_total_slack(schedule):
    total_slack = 0
    for or_id, or_surgeries in schedule.items():
        slack_to_add = surgery_scheduling.slack_time(or_surgeries)
        total_slack = total_slack + slack_to_add
    return total_slack


# Calculate total planned slack for both schedules
total_slack_ff = calculate_total_slack(ff_schedule)
total_slack_lpt = calculate_total_slack(lpt_schedule)
total_slack_regret_based = calculate_total_slack(best_schedule_output)


def calculate_total_free_time(schedule):
    total_free_time = 0
    total_overtime = 0
    for or_id, or_surgeries in schedule.items():
        total_scheduled_time = sum(surgery[2][0] for surgery in or_surgeries)
        total_slack_time = surgery_scheduling.slack_time(or_surgeries)
        total_or_time = total_scheduled_time + total_slack_time
        total_free_time += max(0, surgery_scheduling.max_capacity - total_or_time)
        if total_or_time > surgery_scheduling.max_capacity:
            total_overtime += (total_or_time - surgery_scheduling.max_capacity)
    return total_free_time, total_overtime


total_free_time_ff = calculate_total_free_time(ff_schedule)
total_free_time_lpt = calculate_total_free_time(lpt_schedule)
total_free_time_regret_based = calculate_total_free_time(best_schedule_output)
or_capacity = surgery_scheduling.max_capacity * surgery_scheduling.number_of_ors * \
              surgery_scheduling.number_working_days

print(f"Total surgery time: {surgery_scheduling.total_surgery_time()/60} hours")
print(f"Total OR capacity: {or_capacity/60} hours")

print("Best schedule using ff:", ff_schedule)
print(f"Total overtime using ff: {ff_overtime/60} hours")
print(f"Total running time for ff: {ff_time/60} minutes")
print(f"Total planned slack for ff: {total_slack_ff/60} hours")
print(f"Total free time for ff: {total_free_time_ff[0]/60} hours")
print(f"Total overtime for ff (in free time function): {total_free_time_ff[1]/60} hours")

print("Best schedule using lpt:", lpt_schedule)
print(f"Total overtime using lpt: {overtime_lpt/60} hours")
print(f"Total running time for lpt: {lpt_time/60} minutes")
print(f"Total planned slack for lpt: {total_slack_lpt/60} hours")
print(f"Total free time for lpt: {total_free_time_lpt[0]/60} hours")
print(f'Total free time lpt within lpt function: {free_time_lpt/60} hours')
print(f"Total overtime for lpt (in free time function): {total_free_time_lpt[1]/60} hours")

print("Best schedule using regret based:", best_schedule_output)
print("Parameter setting regret based: z=9, samples=10, ORs=4, alpha=10, 2500 surgeries, 1 year (5 working days)")
print(f"Total overtime using regret based: {best_schedule_overtime_output/60} hours")
print(f"Total running time for regret based: {regret_based_sampling_time/60} minutes")
print(f"Total planned slack for regret based: {total_slack_regret_based/60} hours")
# print(f'Total slack inside function: {slack_regret/60} hours')
print(f"Total free time for regret based: {total_free_time_regret_based[0]/60} hours")
print(f"Total overtime for regret based (in free time function): {total_free_time_regret_based[1]/60} hours")
