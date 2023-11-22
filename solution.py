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
def get_best_separation(features: list, labels: list) -> (int, int):
    best_separation_feature, best_separation_value = 0, 0
    e, f = 0, 0
    e0 = 0
    e1 = 0
    f0 = 0
    f1 = 0
    a = 0
    b0 = 0
    b1 = 0

    for i in range(features.shape[1]):
        if labels[i] == 0: b0 += 1
        else: b1 += 1

    base_entropy = get_entropy(b0, b1)

    for i in range(features.shape[1]):
        cntr = 0
        for row in features:
            a = row[i]
            for sep in features:
                if sep[i] <= a: 
                    e += 1
                    if labels[cntr] == 0: e0 += 1
                    else: e1 += 1
                else: 
                    f += 1
                    if labels[cntr] == 0: f0 += 1
                    else: f1 += 1
            info_win = base_entropy - (e / (e+f) * get_entropy(e0, e1) + f/(f+e) * get_entropy(f0, f1))    
            if info_win > best_separation_value:
                best_separation_value = info_win
                best_separation_feature = i
            cntr += 1
        
    return best_separation_feature, best_separation_value

################### 3. feladat, döntési fa implementációja ####################
def main():
    #TODO: implementálja a döntési fa tanulását!
    


    return 0

if __name__ == "__main__":
    main()
