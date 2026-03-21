# training data credit: https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset

# input='training_output.txt', epoch=25, lr=.3, wordNgrams=5, verbose=True, seed=42
{
    "__label__legit": {
        "precision": 0.9915108104523146,
        "recall": 0.9836820634293986,
        "f1score": 0.9875809221825869,
    },
    "__label__phishing": {
        "precision": 0.9856098410119531,
        "recall": 0.9925207432511394,
        "f1score": 0.9890532199836963,
    },
}


# input='training_output.txt', epoch=5, lr=.3, wordNgrams=5, verbose=True, seed=42
{
    "__label__legit": {
        "precision": 0.9907309322033898,
        "recall": 0.9846032372680615,
        "f1score": 0.9876575803577322,
    },
    "__label__phishing": {
        "precision": 0.9864016736401674,
        "recall": 0.9918195629309338,
        "f1score": 0.9891031991142707,
    },
}


# input='training_output.txt', epoch=5, lr=.3, wordNgrams=2, verbose=True, seed=42
{
    "__label__legit": {
        "precision": 0.9913861648555526,
        "recall": 0.9844716410053954,
        "f1score": 0.9879168042258171,
    },
    "__label__phishing": {
        "precision": 0.986295005807201,
        "recall": 0.9924038798644385,
        "f1score": 0.9893400128152852,
    },
}

# input='training_output.txt', epoch=5, lr=.3, wordNgrams=1, verbose=True, seed=42
{
    "__label__legit": {
        "precision": 0.9888888888888889,
        "recall": 0.9838136596920647,
        "f1score": 0.9863447456956264,
    },
    "__label__phishing": {
        "precision": 0.9856910190786412,
        "recall": 0.9901834755171205,
        "f1score": 0.987932140150411,
    },
}

# input='training_output.txt', epoch=5, lr=2, wordNgrams=1, verbose=True, seed=42
{
    "__label__legit": {
        "precision": 0.9879422287001458,
        "recall": 0.9811817344387419,
        "f1score": 0.9845503763369866,
    },
    "__label__phishing": {
        "precision": 0.9833894761296318,
        "recall": 0.9893654318102139,
        "f1score": 0.9863684026564138,
    },
}

# input='training_output.txt', epoch=5, lr=3, wordNgrams=1, verbose=True, seed=42
{
    "__label__legit": {
        "precision": 0.9873083024854574,
        "recall": 0.9827608895907356,
        "f1score": 0.9850293477544022,
    },
    "__label__phishing": {
        "precision": 0.9847532588454376,
        "recall": 0.9887811148767092,
        "f1score": 0.9867630765642311,
    },
}

# input='training_output.txt', epoch=5, lr=.001, wordNgrams=1, verbose=True, seed=42
{
    "__label__legit": {
        "precision": 0.9253731343283582,
        "recall": 0.016317936570601394,
        "f1score": 0.032070347859821546,
    },
    "__label__phishing": {
        "precision": 0.5334540007489702,
        "recall": 0.9988313661329905,
        "f1score": 0.695471744171854,
    },
}


#model = floret.train_supervised(input='training_output.txt', autotuneValidationFile='testing_output.txt', autotuneDuration=7200, autotuneModelSize="900M", seed=42)
{'autotuneDuration': 300, 'autotuneMetric': 'f1', 'autotuneModelSize': '', 'autotunePredictions': 1, 'autotuneValidationFile': '', 'bucket': 168809, 'cutoff': 0, 'dim': 165, 'dsub': 2, 'epoch': 100, 'hashCount': 1, 'input': '', 'label': '__label__', 'loss': <loss_name.softmax: 3>, 'lr': 0.05, 'lrUpdateRate': 100, 'maxn': 6, 'minCount': 1, 'minCountLabel': 0, 'minn': 3, 'mode': <mode_name.fasttext: 1>, 'model': <model_name.supervised: 3>, 'neg': 5, 'output': '', 'pretrainedVectors': '', 'qnorm': False, 'qout': False, 'retrain': False, 'saveOutput': False, 'seed': 0, 'setManual': "<bound method PyCapsule.setManual of <floret_pybind.args object at 0x7f1f01dbe7b0>>", 't': 0.0001, 'thread': 12, 'verbose': 2, 'wordNgrams': 5, 'ws': 5}
{
    "__label__legit": {
        "precision": 0.9848088004190676,
        "recall": 0.9896038952493749,
        "f1score": 0.9872005251066623,
    },
    "__label__phishing": {
        "precision": 0.9907276995305164,
        "recall": 0.9864438471426902,
        "f1score": 0.9885811325174211,
    },
}

# takes the autotuned model and applies it to testing data from NDIT
{'__label__legit': {'precision': 0.3807471264367816, 'recall': 0.28804347826086957, 'f1score': 0.327970297029703}, '__label__phishing': {'precision': 0.71700151220566, 'recall': 0.793829227457546, 'f1score': 0.7534619750283769}}
