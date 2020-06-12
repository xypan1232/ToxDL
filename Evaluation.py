__author__ = 'jasper.zuallaert, Xiaoyong.Pan'
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.metrics import f1_score, precision_recall_curve, matthews_corrcoef
# Evaluates the predictions as written to the a file. The evaluation is done on a per_protein level, as specified in
# the DeepGO publication (or in our documentation)
# - file: the location of the file to be read. The file should have alternating lines of
#              a) predictions   e.g. 0.243,0.234,0.431,0.013,0.833
#              b) labels        e.g. 0,1,1,1,0,0
# - classN: the class index if we want to evaluate for just 1 single term
def run_eval_per_protein(file, classN = None):
    import numpy as np
    mx = 0.0
    allLines = open(file).readlines()
    height = len(allLines)//2
    width = len(allLines[0].split(','))

    preds = np.zeros((height,width),dtype=np.float32)
    labels = np.zeros((height,width),dtype=np.int32)

    for h in range(height):
        l1 = allLines[h*2].split(',')
        l2 = allLines[h*2+1].split(',')
        for w in range(width):
            preds[h][w] = float(l1[w])
            labels[h][w] = int(l2[w])

    if classN != None:
        preds2 = np.zeros((height,1),dtype=np.float32)
        labels2 = np.zeros((height,1),dtype=np.int32)
        for h in range(height):
            preds2[h][0] = preds[h][classN]
            labels2[h][0] = labels[h][classN]
        preds = preds2
        labels = labels2

    precisions = []
    recalls = []
    fprs = []

    tp_final,fp_final,tn_final,fn_final = 0,0,0,0

    for t in range(0,1000):
        thr = t/1000
        preds_after_thr = (preds[:, :] > thr).astype(np.int32)

        tp = preds_after_thr * labels
        tp_per_sample = np.sum(tp,axis=1)

        fp_per_sample = np.sum(preds_after_thr * (np.ones_like(labels) - labels),axis=1)

        pos_predicted_for_sample = np.sum(preds_after_thr,axis=1)
        pos_labels_for_sample = np.sum(labels,axis=1)
        neg_labels_for_sample = len(labels[0]) - pos_labels_for_sample

        num_of_predicted_samples = np.sum((np.sum(preds_after_thr,axis=1)[:] > 0).astype(np.int32))
        pr_i_s = np.nan_to_num(tp_per_sample / pos_predicted_for_sample)
        se_i_s = np.nan_to_num(tp_per_sample / pos_labels_for_sample)
        fpr_i_s = np.nan_to_num(fp_per_sample / neg_labels_for_sample)
        pr = np.nan_to_num(np.sum(pr_i_s) / num_of_predicted_samples)

        if classN == None:
            se_divisor = height
            fpr_divisor = height
        else:
            se_divisor = np.sum(labels)
            fpr_divisor = np.sum(np.ones_like(labels)-labels)

        se = np.sum(se_i_s) / se_divisor
        fpr = np.nan_to_num(np.sum(fpr_i_s) / fpr_divisor)

        precisions.append(pr)
        recalls.append(se)
        fprs.append(fpr)

        f = 2 * pr * se / (pr + se)
        if f > mx:
            mx = f
        if classN != None and se > 0.50:
            tp_final = np.sum(tp)
            fp_final = np.sum(pos_predicted_for_sample) - tp_final
            fn_final = np.sum(pos_labels_for_sample) - tp_final
            tn_final = len(tp_per_sample) - tp_final - fn_final - fp_final

    auROC = auc(fprs,recalls)
    auPRC = auc(recalls,precisions)

    print('Total average per protein: {} {} F1={:1.7f} auROC={:1.7f} auPRC={:1.7f}'.format(file,classN,mx,auROC,auPRC))


# Evaluates the predictions as written to the a file. The evaluation is done on a per_term level (see our documentation)
# - file: the location of the file to be read. The file should have alternating lines of
#              a) predictions   e.g. 0.243,0.234,0.431,0.013,0.833
#              b) labels        e.g. 0,1,1,1,0,0
def run_eval_per_term(file):
    import numpy as np

    allLines = open(file).readlines()
    height = len(allLines)//3
    width = len(allLines[0].split(','))

    preds = np.zeros((width,height),dtype=np.float32)
    labels = np.zeros((width,height),dtype=np.int32)

    for h in range(height):
        l1 = allLines[h*3].split(',')
        l2 = allLines[h*3+1].split(',')
        for w in range(width):
            preds[w][h] = float(l1[w])
            labels[w][h] = int(l2[w])

    all_auROC = []
    all_auPRC = []
    all_Fmax = []
    from math import isnan
    for termN in range(width):
        auROC, auPRC, Fmax, mcc = _calc_metrics_for_term(preds[termN],labels[termN])
        #auROC, auPRC, Fmax, tp,fn,tn,fp = _calc_metrics_for_term(sorted(zip(preds[termN],labels[termN])))
        #if not isnan(auROC) and not isnan(auROC):
        #    all_auROC.append(auROC)
        #    all_auPRC.append(auPRC)
        #    all_Fmax.append(Fmax)
        print('auROC:', auROC, 'auPRC', auPRC, 'F1:', Fmax, 'MCC', mcc)
    return auROC, auPRC, Fmax, mcc 
        #print(f'Term {termN: 3d}: auROC {auROC:1.4f}, auPRC {auPRC:1.4f}, Fmax {Fmax:1.4f} --- Example: TP {tp: 4d}, FP {fp: 4d}, TN {tn: 4d}, FN {fn: 4d}')
    #print(f'Total average per term: auROC {sum(all_auROC)/len(all_auROC)}, {sum(all_auPRC)/len(all_auPRC)}, {sum(all_Fmax)/len(all_Fmax)}')

def _calc_metrics_for_term(preds, test_label):
    auroc = roc_auc_score(test_label, preds)
    precision, recall, thresholds = precision_recall_curve(test_label, preds)
    auprc = auc(recall, precision)
    preds[preds>=0.5] = 1
    preds[preds<0.5] = 0
    f1score = f1_score(test_label, preds, average='binary')
    mcc = matthews_corrcoef(test_label, preds)
    return auroc, auprc, f1score, mcc

# Calculate the metrics for a single term
# - pred_and_lab_for_term: predictions and labels for this particular term, of format [(pred1, lab1), (pred2, lab2), ...]
def _calc_metrics_for_term1(pred_and_lab_for_term):
    pred_and_lab_for_term = pred_and_lab_for_term[::-1]
    total_pos = sum([x for _,x in pred_and_lab_for_term])
    tp,fp = 0,0
    fn = total_pos
    tn = len(pred_and_lab_for_term) - total_pos
    allSens, allPrec, allFPR = [],[],[]
    Fmax = 0

    tp_final, fn_final, tn_final, fp_final = -1, -1, -1, -1

    allSens.append(0.0)
    allPrec.append(0.0)
    allFPR.append(0.0)

    index = 0
    while index < len(pred_and_lab_for_term):
        last_with_this_probability = index < len(pred_and_lab_for_term) - 1 and pred_and_lab_for_term[index][0] != pred_and_lab_for_term[index+1][0]
        if pred_and_lab_for_term[index][1] == 1:
            tp += 1
            fn -= 1
        else: # 0
            fp += 1
            tn -= 1

        sens = tp / (tp + fn)
        prec = tp / (tp + fp)
        fpr =  fp / (fp + tn)

        if sens > 0.5 and tp_final == -1:
            tp_final = tp
            tn_final = tn
            fp_final = fp
            fn_final = fn

        if last_with_this_probability:
                allSens.append(sens)
                allPrec.append(prec)
                f1 = 2 * sens * prec / (sens + prec)
                if f1 > Fmax:
                    Fmax = f1
                allFPR.append(fpr)

        index += 1
    allSens.append(1.0)
    allPrec.append(total_pos / len(pred_and_lab_for_term))
    allFPR.append(1.0)

    auROC = auc(allFPR, allSens)
    auPRC = auc(allSens, allPrec)
    return auROC, auPRC, Fmax, tp_final,fn_final,tn_final,fp_final

