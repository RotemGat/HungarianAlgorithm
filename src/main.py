import pickle
import numpy as np
from munkres import Munkres

from HungarianAlgorithm import hungarian_algorithm, ans_calculation
from scipy.optimize import linear_sum_assignment

def greedy_assignment(cost_matrix):
    # Greedy assignment algorithm
    n = cost_matrix.shape[0]
    assignment = []
    total_cost = 0
    available_rows = set(range(n))
    available_cols = set(range(n))

    while available_rows and available_cols:
        min_cost = np.inf
        min_row = -1
        min_col = -1

        # Find the minimum cost assignment
        for row in available_rows:
            for col in available_cols:
                if cost_matrix[row, col] < min_cost:
                    min_cost = cost_matrix[row, col]
                    min_row = row
                    min_col = col

        # Make the assignment
        assignment.append((min_row, min_col))
        total_cost += 100 - min_cost
        available_rows.remove(min_row)
        available_cols.remove(min_col)

    return assignment, total_cost

def main():
    with open('matching_df.pickle', 'rb') as f:
        df = pickle.load(f)
    with open('people.pickle', 'rb') as f:
        people_df = pickle.load(f)

    people_df = people_df.iloc[:, :1]
    # keep only the women on the rows and men in the columns, to prevent no-equal results
    males = people_df[people_df['sex'] == 1].index.to_list()
    females = people_df[people_df['sex'] == 0].index.to_list()

    # Adjust the n_people to run for less people
    n_people = 900
    n = min(len(males), len(females), n_people)

    # Select the first n males and first n females
    selected_males = males[:n]
    selected_females = females[:n]

    # Map male names to their original indices
    male_indices = [idx for idx, name in enumerate(df.columns) if name in selected_males]

    # Reorder the correlation matrix
    filtered_matrix = df.loc[male_indices, selected_females]
    cost_matrix = 100 * np.ones_like(filtered_matrix) - filtered_matrix.to_numpy()

    # Greedy assignment:
    greedy_matching, greedy_total_cost = greedy_assignment(cost_matrix)
    print(f'Greedy total cost: {greedy_total_cost}')

    # Hungarian Algorithm - Our implementation
    ans_pos = hungarian_algorithm(cost_matrix.copy())  # Get the element position.
    ans, ans_mat = ans_calculation(cost_matrix, ans_pos)  # Get the minimum or maximum value and corresponding matrix.
    print(f"Linear Assignment problem total cost: {ans:.0f}")

    # Hungarian Algorithm - PyPi
    m = Munkres()
    matches_pypi = m.compute(cost_matrix.copy())
    total = 0
    for row, column in matches_pypi:
        value = cost_matrix[row][column]
        total += 100 - value
        # print(f'({row}, {column}) -> {value}')
    print(f'PyPi total cost: {total}')

    # Scipy
    rows, columns = linear_sum_assignment(cost_matrix.copy())
    total_sp = 0
    for row, column in zip(rows, columns):
        value = cost_matrix[row][column]
        total_sp += 100 - value
        # print(f'({row}, {column}) -> {value}')
    print(f'Scipy total cost: {total_sp}')




if __name__ == '__main__':
    main()
