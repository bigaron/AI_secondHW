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

###################### 2. feladat, optimális szeparáció #######################
def get_best_separation(features: list,
                        labels: list) -> (int, int):
    best_separation_feature, best_separation_value = 0, 0
    #TODO számítsa ki a legjobb szeparáció tulajdonságát és értékét!
    return best_separation_feature, best_separation_value

################### 3. feladat, döntési fa implementációja ####################
def main():
    #TODO: implementálja a döntési fa tanulását!
    return 0

if __name__ == "__main__":
    main()
