import numpy as np #(működik a Moodle-ben is)
import math

######################## 1. feladat, entrópiaszámítás #########################
def get_entropy(n_cat1: int, n_cat2: int) -> float:
    entropy = 0
    if n_cat1 == 0 or n_cat2 == 0: return entropy
  
    cat_sum = n_cat1 + n_cat2
    prob1 = n_cat1 / cat_sum
    prob2 = n_cat2 / cat_sum
    entropy = -((prob1 * math.log(prob1, 2)) + (prob2 * math.log(prob2, 2)))

    return entropy


def get_min_max(features: list, col: int) -> (int, int):
    min_val = max_val = features[0][col]
    for row in features:
        if row[col] > max_val: max_val = row[col]
        elif row[col] < min_val: min_val = row[col]

    return min_val, max_val

###################### 2. feladat, optimális szeparáció #######################
def get_best_separation(features: list, labels: list) -> (int, int):
    best_separation_feature, best_separation_value = 0, 0
    base_labels = [0, 0]
    for cntr in range(labels.size):
        base_labels[labels[cntr]] += 1

    base_entropy = get_entropy(base_labels[0], base_labels[1])

    best_separation_value = 0
    current_best_info_gain = 0

    for col in range(features.shape[1]):
        (min_a, max_a) = get_min_max(features, col)

        for a_iterator in range(min_a, max_a, 1):
            e_labels = [0, 0]
            f_labels = [0, 0]
            e = 0
            f = 0
            for second_scan in range(features.shape[0]):
                if features[second_scan][col] <= a_iterator:
                    e += 1
                    e_labels[labels[second_scan]] += 1
                else:
                    f += 1
                    f_labels[labels[second_scan]] += 1
            current_info_gain = base_entropy - (e / (e+f) * get_entropy(e_labels[0], e_labels[1]) + f/(e+f) * get_entropy(f_labels[0], f_labels[1])) 
            if current_info_gain > current_best_info_gain:
                current_best_info_gain = current_info_gain
                best_separation_value = a_iterator
                best_separation_feature = col

    return best_separation_feature, best_separation_value

################### 3. feladat, döntési fa implementációja ####################
def main():
    #TODO: implementálja a döntési fa tanulását!
    


    return 0

if __name__ == "__main__":
    main()
