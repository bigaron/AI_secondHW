import numpy as np #(működik a Moodle-ben is)
import math
import csv

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
    base_labels = [0, 0]
    for cntr in range(labels.size):
        base_labels[labels[cntr]] += 1

    base_entropy = get_entropy(base_labels[0], base_labels[1])

    best_separation_value = 0
    current_best_info_gain = 0

    for col in range(features.shape[1]):
        for row in features:
            e_labels = [0, 0]
            f_labels = [0, 0]
            e = 0
            f = 0
            a_iterator = row[col]
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

def read_file(path: str, training_data: bool) -> (np.array, np.array):
    file = open(path)
    csv_reader = csv.reader(file)
    features = np.array([[0,0,0,0,0,0,0,0]]).astype(int)
    labels = np.empty([]).astype(int)
    for row in csv_reader:
        for i in range(len(row)):
                row[i] = int(row[i])
        if not training_data: features = np.append(features, [row], axis=0)
        else:
            without_label = np.array([row[0:len(row)-1]])
            label = row[len(row) - 1]
            features = np.append(features, without_label, axis=0)
            labels = np.append(labels, label)

    file.close()
    features = np.delete(features, 0, axis=0)
    labels = np.delete(labels, 0)
    if not training_data: labels = None
    return features, labels

def get_sub_arrays(features: list, labels:list, a: int, col: int) -> ((list, list), (list, list)):
    e = np.array([[0,0,0,0,0,0,0,0]]).astype(int)
    f = np.array([[0,0,0,0,0,0,0,0]]).astype(int)

    e_labels = np.array(0).astype(int)
    f_labels = np.array(0).astype(int)
    for row in range(features.shape[0]):
        if features[row][col] <= a:
            e = np.append(e, [features[row]], axis=0)
            e_labels = np.append(e_labels, labels[row]) 
        else:
            f = np.append(f, [features[row]], axis=0)
            f_labels = np.append(f_labels, labels[row])

    e = np.delete(e, 0, axis=0)
    f = np.delete(f, 0, axis=0)

    e_labels = np.delete(e_labels, 0)
    f_labels = np.delete(f_labels, 0)

    return (e, e_labels), (f, f_labels)

def get_entropy_of_array(labels: list) -> float:
    label_count = [0, 0]
    for row in labels:
        label_count[row] += 1

    return get_entropy(label_count[0], label_count[1])

class TreeNode:
    def __init__(self):
        self.children = list()
        self.values = (int, int)
    label_value = -1


def populate_tree(root: TreeNode ,features: list, labels: list):
    e_child = TreeNode()
    f_child = TreeNode()
    ((e_features, e_labels),(f_features, f_labels)) = get_sub_arrays(features, labels, root.values[1], root.values[0])
    
    if np.isclose(get_entropy_of_array(labels), 0.0):
        root.label_value = 1 if labels.tolist().count(0) == 0 else 0
        return 

    #e component

    e_child.values = get_best_separation(e_features, e_labels)
    root.children.append(e_child)
    populate_tree(e_child, e_features, e_labels)

    #f component

    f_child.values = get_best_separation(f_features, f_labels)
    root.children.append(f_child)
    populate_tree(f_child, f_features, f_labels)

def iterate_tree(root: TreeNode, row: list) -> int:
    if root.label_value != -1: return root.label_value
    if root.values[1] >= row[root.values[0]]: return iterate_tree(root.children[0], row)
    else: return iterate_tree(root.children[1], row)


def speculate_label(features: list, root: TreeNode):
    with open('results.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for row in features:
            writer.writerow([iterate_tree(root, row)])

################### 3. feladat, döntési fa implementációja ####################
def main():
    (features, labels) = read_file('train.csv', True)
    
    root = TreeNode()
    root.values = get_best_separation(features, labels)
    populate_tree(root, features, labels) 

    (features, labels) = read_file('test.csv', False)

    speculate_label(features, root)

    return 0

if __name__ == "__main__":
    main()
