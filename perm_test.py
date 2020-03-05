# Reusing code from here: https://github.com/irenenikk/sentiment-analysis-comparison

def get_accuracy(preds, Y):
       return (preds == Y).sum()/len(preds) 

def swap_randomly(A, B, shift_indx=None):
    assert len(A) == len(B), 'Both lists have to be the same size.'
    if shift_indx is None:
        # sample random amount of indexes to swap using a coin toss
        shift_amount = np.random.binomial(len(A), 0.5)
        indexes = list(range(len(A)))
        # flip the values in lists in random places
        index_perm = np.random.permutation(indexes)
        shift_indx = index_perm[:shift_amount]
    tmp = A[shift_indx]
    A[shift_indx] = B[shift_indx]
    B[shift_indx] = tmp
    return A, B

def permutation_test(true_labels, results_A, results_B, R=5000):
    """ Monte carlo permutation test on two different prediction lists. """
    acc_differences = np.zeros(R)
    true_acc_A = get_accuracy(results_A, true_labels)
    true_acc_B = get_accuracy(results_B, true_labels)
    true_diff = np.abs(true_acc_A - true_acc_B)
    for i in range(R):
        shuff_A, shuff_B = swap_randomly(results_A, results_B)
        acc_A = (shuff_A == true_labels).sum()/len(shuff_A)
        acc_B = (shuff_B == true_labels).sum()/len(shuff_B)
        acc_differences[i] = np.abs(acc_A - acc_B)
    return ((acc_differences >= true_diff).sum()+1)/(R+1)
    