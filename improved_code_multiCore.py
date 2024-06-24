from WaitingList import waiting_list
from scipy.stats import norm
from scipy.optimize import fsolve
import numpy as np
import pickle
import random
import itertools
import time
from collections import defaultdict
import multiprocessing

class SurgeryScheduling:
    def __init__(self, number_surgeries, distr, number_of_ors):
        self.number_surgeries = number_surgeries
        self.distr = distr
        self.number_of_ors = number_of_ors
        self.waiting = waiting_list(number_surgeries, distr)
        self.waiting_list_mixture_surgeries = self.waiting[0]
        self.waiting_list_mixture_surgeries_list = [[key, *value] for key, value in self.waiting_list_mixture_surgeries.items()]
        self.waiting_list_individual_surgeries = self.waiting[1]
        self.weights_dictionary = pickle.load(open('weights_for_mixture.pkl', 'rb'))
        self.weights_surgeries_waiting_list = [self.weights_dictionary[value[0]] for key, value in self.waiting_list_mixture_surgeries.items() if value[0] in self.weights_dictionary]
        self.number_working_days = 90 // 5  # 3 months
        self.ors = defaultdict(list, {(k, t): [] for t in range(1, self.number_working_days + 1) for k in range(1, number_of_ors + 1)})
        self.max_capacity = 450  # per OR in minutes (8h)
        self.overtime_prob = 0.3
        self.alpha = 10
        self.slack_stored = {}

    def sort_waiting_list(self):
        ordered_waiting_list_dict = dict(sorted(self.waiting_list_mixture_surgeries.items(), key=lambda item: item[1][1][0], reverse=True))
        ordered_waiting_list_list = [[k, *v] for k, v in ordered_waiting_list_dict.items()]
        sorted_surgeries = list(ordered_waiting_list_dict.keys())
        return ordered_waiting_list_dict, ordered_waiting_list_list, sorted_surgeries

    def total_surgery_time(self):
        return sum(surgery[2][0] for surgery in self.waiting_list_mixture_surgeries_list)

    def first_fit(self):
        current_schedule_ff = defaultdict(list, {or_plan: [] for or_plan in self.ors})
        remaining_surgeries = len(self.waiting_list_mixture_surgeries_list)

        for surgery_info in self.waiting_list_mixture_surgeries_list:
            assigned = False
            for or_plan, surgeries in current_schedule_ff.items():
                total_time = sum(t[2][0] for t in surgeries) + surgery_info[2][0] + self.slack_time(surgeries + [surgery_info])
                if total_time <= self.max_capacity:
                    current_schedule_ff[or_plan].append(surgery_info)
                    assigned = True
                    break

            if not assigned:
                min_or_used = min(current_schedule_ff, key=lambda k: sum(x[2][0] for x in current_schedule_ff[k]) + surgery_info[2][0] + self.slack_time(current_schedule_ff[k] + [surgery_info]) - self.max_capacity)
                current_schedule_ff[min_or_used].append(surgery_info)

            remaining_surgeries -= 1
            print(f"FF: Iteration {len(self.waiting_list_mixture_surgeries_list) - remaining_surgeries}: {remaining_surgeries} surgeries remaining")

        total_overtime = sum(max(sum(t[2][0] for t in or_surgeries) + self.slack_time(or_surgeries) - self.max_capacity, 0) for or_surgeries in current_schedule_ff.values())
        return current_schedule_ff, total_overtime

    def lpt(self):
        _, ordered_waiting_list, _ = self.sort_waiting_list()
        current_schedule_lpt = defaultdict(list, {or_plan: [] for or_plan in self.ors})
        remaining_surgeries = len(ordered_waiting_list)

        for surgery_info in ordered_waiting_list:
            possible_ors = [or_id for or_id, surgeries in current_schedule_lpt.items() if not surgeries]

            for or_plan, surgeries_in_or in current_schedule_lpt.items():
                if surgeries_in_or:
                    surgeries_or_with_surgery = surgeries_in_or + [surgery_info]
                    slack = self.slack_time(surgeries_or_with_surgery)
                    if sum(t[2][0] for t in surgeries_in_or) + surgery_info[2][0] + slack <= self.max_capacity:
                        possible_ors.append(or_plan)

            if possible_ors:
                or_to_use = possible_ors[0]
                current_schedule_lpt[or_to_use].append(surgery_info)
            else:
                min_or_used = min(current_schedule_lpt, key=lambda k: sum(x[2][0] for x in current_schedule_lpt[k]) + surgery_info[2][0] + self.slack_time(current_schedule_lpt[k] + [surgery_info]) - self.max_capacity)
                current_schedule_lpt[min_or_used].append(surgery_info)

            remaining_surgeries -= 1
            print(f"LPT: Iteration {len(self.waiting_list_mixture_surgeries_list) - remaining_surgeries}: {remaining_surgeries} surgeries remaining")

        total_overtime = sum(max(sum(t[2][0] for t in or_surgeries) + self.slack_time(or_surgeries) - self.max_capacity, 0) for or_surgeries in current_schedule_lpt.values())
        return current_schedule_lpt, total_overtime

    def delta_expression(self, surgeries_for_slack, delta):
        if isinstance(surgeries_for_slack[0], str):
            surgeries_for_slack = [surgeries_for_slack]

        weights_for_slack = [self.weights_dictionary[surgery[1]] for surgery in surgeries_for_slack]
        patient_id = [patient_info[0] for patient_info in surgeries_for_slack]

        total_mu = sum(self.waiting_list_mixture_surgeries[patient_info[0]][1][0] for patient_info in surgeries_for_slack)
        mu_list, sigma_list = zip(*[zip(*self.waiting_list_individual_surgeries[ids][0]) for ids in patient_id])

        weights = np.array(weights_for_slack, dtype=object)
        mus = np.array(mu_list, dtype=object)
        sigmas = np.array(sigma_list, dtype=object)
        num_elements = [len(sublist) for sublist in weights_for_slack]

        index_combinations = list(itertools.product(*[range(n) for n in num_elements]))
        selected_weights = [[weight[i] for i, weight in zip(indices, weights)] for indices in index_combinations]
        selected_mus = [[mu[i] for i, mu in zip(indices, mus)] for indices in index_combinations]
        selected_sigmas = [[sigma[i] for i, sigma in zip(indices, sigmas)] for indices in index_combinations]

        weight_products = np.prod(np.array(selected_weights), axis=1)
        mu_sums = np.sum(np.array(selected_mus), axis=1)
        sigma_sums = np.sum(np.array(selected_sigmas), axis=1)
        cdf_arguments = (total_mu + delta - mu_sums) / sigma_sums
        cdf_values = norm.cdf(cdf_arguments)
        result = np.sum(weight_products * cdf_values)
        return result

    def to_solve(self, delta, surgeries_for_slack):
        return self.delta_expression(surgeries_for_slack, delta) - (1 - self.overtime_prob)

    def slack_time(self, surgeries_for_slack):
        if not surgeries_for_slack:
            return 0

        tuple_map = tuple(tuple(sublist) for sublist in surgeries_for_slack)
        if tuple_map in self.slack_stored:
            return self.slack_stored[tuple_map]

        solution_slack = fsolve(self.to_solve, x0=np.array([0]), args=surgeries_for_slack)
        self.slack_stored[tuple_map] = abs(solution_slack[0])
        return abs(solution_slack[0])

    def calculate_priority(self, current_surgery, possible_ors, schedule):
        diff_per_or = []
        omega_per_or = {}

        for ors in possible_ors:
            current_surgeries_in_or = schedule[ors]
            surgeries_with_current_surgery = current_surgeries_in_or + [current_surgery] if current_surgeries_in_or else [current_surgery]
            diff_or = self.slack_time(surgeries_with_current_surgery) - self.slack_time(current_surgeries_in_or)
            diff_per_or.append(diff_or)
            omega_or = self.slack_time(current_surgery) - diff_or if current_surgeries_in_or else 0
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

        for _ in range(samples):
            current_schedule = defaultdict(list, {or_plan: [] for or_plan in self.ors})
            remaining_surgeries = sorted_waiting_list[:]
            while remaining_surgeries:
                print("REGRET: this surgeries left", len(remaining_surgeries))
                z_surgeries = remaining_surgeries[:min(z, len(remaining_surgeries))]

                priorities_z_surgeries = []
                surgeries_to_draw = []
                for surgery in z_surgeries:
                    possible_ors = [or_id for or_id, surgeries in current_schedule.items() if not surgeries]
                    for or_plan, surgeries_in_or in current_schedule.items():
                        if surgeries_in_or:
                            surgeries_or_with_surgery = surgeries_in_or + [surgery]
                            slack = self.slack_time(surgeries_or_with_surgery)
                            if sum(t[2][0] for t in surgeries_in_or) + surgery[2][0] + slack <= self.max_capacity:
                                possible_ors.append(or_plan)

                    if possible_ors:
                        priority_surgery_i = self.calculate_priority(surgery, possible_ors, current_schedule)
                        priorities_z_surgeries.append(priority_surgery_i)
                        surgeries_to_draw.append(priority_surgery_i[0])
                    else:
                        min_or_used = min(current_schedule.keys(), key=lambda k: sum(t[2][0] for t in current_schedule[k]) + surgery[2][0] + self.slack_time(current_schedule[k] + [surgery]) - self.max_capacity)
                        current_schedule[min_or_used].append(surgery)
                        remaining_surgeries.remove(surgery)

                if priorities_z_surgeries:
                    probabilities = self.drawing_probabilities(priorities_z_surgeries)
                    drawn_surgery = random.choices(surgeries_to_draw, probabilities, k=1)[0]
                    for surgery, _, best_or in priorities_z_surgeries:
                        if surgery == drawn_surgery:
                            current_schedule[best_or].append(surgery)
                            remaining_surgeries.remove(surgery)
                            break

            total_overtime = sum(max(sum(t[2][0] for t in or_surgeries) + self.slack_time(or_surgeries) - self.max_capacity, 0) for or_surgeries in current_schedule.values())

            if total_overtime < best_schedule_overtime:
                best_schedule = current_schedule
                best_schedule_overtime = total_overtime

        return best_schedule, best_schedule_overtime

# Function to execute scheduling methods in parallel
def execute_method(method, args):
    return method(*args)

if __name__ == '__main__':
    surgery_scheduling = SurgeryScheduling(number_surgeries=2500, distr="normal", number_of_ors=3)

    # Define arguments for each method
    methods = [
        (surgery_scheduling.first_fit, ()),
        (surgery_scheduling.lpt, ()),
        (surgery_scheduling.regret_based_sampling, (9, 10))
    ]

    start_time = time.time()
    with multiprocessing.Pool(processes=3) as pool:  # Use 3 processes for 3 methods
        results = pool.starmap(execute_method, methods)
    end_time = time.time()

    ff_schedule, ff_overtime = results[0]
    lpt_schedule, overtime_lpt = results[1]
    best_schedule_output, best_schedule_overtime_output = results[2]

    print("Total time taken:", end_time - start_time)


    def calculate_total_slack(schedule):
        return sum(surgery_scheduling.slack_time(or_surgeries) for or_surgeries in schedule.values())


    total_slack_ff = calculate_total_slack(ff_schedule)
    total_slack_lpt = calculate_total_slack(lpt_schedule)
    total_slack_regret_based = calculate_total_slack(best_schedule_output)

    def calculate_total_free_time(schedule):
        total_free_time = 0
        total_overtime = 0
        for or_surgeries in schedule.values():
            total_scheduled_time = sum(surgery[2][0] for surgery in or_surgeries)
            total_slack_time = surgery_scheduling.slack_time(or_surgeries)
            total_or_time = total_scheduled_time + total_slack_time
            total_free_time += max(0, surgery_scheduling.max_capacity - total_or_time)
            if total_or_time > surgery_scheduling.max_capacity:
                total_overtime += (surgery_scheduling.max_capacity - total_or_time)
        return total_free_time, total_overtime

    total_free_time_ff = calculate_total_free_time(ff_schedule)
    total_free_time_lpt = calculate_total_free_time(lpt_schedule)
    total_free_time_regret_based = calculate_total_free_time(best_schedule_output)
    or_capacity = surgery_scheduling.max_capacity * surgery_scheduling.number_of_ors * \
                  surgery_scheduling.number_working_days

    print(f"Total surgery time: {surgery_scheduling.total_surgery_time()/60} hours")
    print(f"Total OR capacity: {or_capacity/60} hours")

    print(f"Total overtime using ff: {ff_overtime/60} hours")
    #print(f"Total running time for ff: {ff_time/60} minutes")
    print(f"Total planned slack for ff: {total_slack_ff} hours")
    print(f"Total free time for ff: {total_free_time_ff[0]/60} hours")
    print(f"Total overtime for ff (in free time function): {total_free_time_ff[1]/60} hours")

    print(f"Total overtime using lpt: {overtime_lpt/60} hours")
    #print(f"Total running time for lpt: {lpt_time/60} minutes")
    print(f"Total planned slack for lpt: {total_slack_lpt} hours")
    print(f"Total free time for lpt: {total_free_time_lpt[0]/60} hours")
    print(f"Total overtime for lpt (in free time function): {total_free_time_lpt[1]/60} hours")

    print("Parameter setting regret based: z=9, samples=10, ORs=4, alpha=10, 2500 surgeries, 1 year (5 working days)")
    print(f"Total overtime using regret based: {best_schedule_overtime_output/60} hours")
    #print(f"Total running time for regret based: {regret_based_sampling_time/60} minutes")
    print(f"Total planned slack for regret based: {total_slack_regret_based/60} hours")
    print(f"Total free time for regret based: {total_free_time_regret_based[0]/60} hours")
    print(f"Total overtime for regret based (in free time function): {total_free_time_regret_based[1]/60} hours")
