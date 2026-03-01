import numpy as np

def accuracy_assessment(img_gt, predict_img):
    esp = 1e-6

    gt   = np.asarray(img_gt).ravel().astype(int)
    pred = np.asarray(predict_img).ravel().astype(int)

    TP = sum((pred == 1) & (gt == 1))
    TN = sum((pred == 0) & (gt == 0))
    FP = sum((pred == 1) & (gt == 0))
    FN = sum((pred == 0) & (gt == 1))

    conf_mat = [[TN, FP],
                [FN, TP]]

    TP = float(TP)
    TN = float(TN)
    FP = float(FP)
    FN = float(FN)

    P   = TP / (TP + FP + esp)
    R   = TP / (TP + FN + esp)
    F1  = 2 * P * R / (P + R + esp)
    acc = (TP + TN) / (TP + TN + FP + FN + esp)
    oa  = acc

    total = TP + TN + FP + FN + esp
    po = oa
    pe = ((TP + FP) * (TP + FN) + (FN + TN) * (FP + TN)) / (total * total)
    kappa = (po - pe) / (1 - pe + esp)

    return conf_mat, oa, kappa, P, R, F1, acc

