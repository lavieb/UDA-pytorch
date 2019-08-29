from ignite.metrics.confusion_matrix import cmPrecision
from ignite.metrics.confusion_matrix import cmRecall


def cmFbeta(cm, beta, average=True):

    p = cmPrecision(cm, average=False)
    r = cmRecall(cm, average=False)
    fbeta = (1 + beta ** 2) * p * r / (beta ** 2 * p + r + 1e-20)
    if average:
        return fbeta.mean()
    return fbeta
