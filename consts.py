import torch

AUTOPHASE_MEAN = torch.tensor([4.6688e+00, 1.9132e+01, 7.1305e+01, 5.0779e+01, 1.5975e+01, 5.8694e+01,
        3.2220e+01, 6.1659e+00, 2.1574e+01, 4.3878e+01, 3.2922e+00, 2.1238e+01,
        2.5634e+00, 9.1586e+01, 2.3801e+01, 1.0257e+02, 2.0502e+01, 1.7531e+01,
        1.4645e+02, 1.5134e+02, 6.2780e+01, 5.5970e+01, 9.8307e+01, 5.8694e+01,
        2.9113e+01, 2.0971e-01, 2.9028e+01, 6.3104e+01, 3.4423e-01, 1.1466e+01,
        1.0392e+02, 3.1820e+01, 1.0257e+02, 5.0625e+01, 5.4677e+01, 4.4062e+01,
        2.7024e-01, 1.4766e+02, 5.1781e+00, 5.8407e-01, 5.6808e+01, 6.7520e+00,
        1.2990e+01, 7.9500e-01, 1.4267e+00, 1.3244e+02, 3.6180e+00, 9.9664e-01,
        1.3290e-01, 5.9825e+00, 1.1539e+02, 7.7327e+02, 4.4850e+02, 2.1601e+01,
        1.0208e+02, 2.6947e+02])
AUTOPHASE_STD = torch.tensor([8.2933e+00, 2.2122e+01, 5.1213e+01, 3.8246e+01, 1.4216e+01, 4.6063e+01,
        2.3720e+01, 7.5252e+00, 1.6349e+01, 3.1603e+01, 3.7120e+00, 2.4700e+01,
        7.7506e+00, 6.0218e+01, 2.6546e+01, 7.5441e+01, 6.8392e+00, 2.6552e+01,
        1.0610e+02, 1.4240e+02, 8.1920e+01, 3.2670e+01, 1.4347e+02, 4.6063e+01,
        2.1702e+01, 5.2106e-01, 2.7271e+01, 1.3448e+02, 1.0337e+00, 1.4904e+01,
        7.0555e+01, 9.5391e+01, 7.5441e+01, 1.6909e+01, 7.4241e+01, 2.9976e+01,
        6.6510e-01, 2.3576e+02, 2.6579e+00, 5.2362e-01, 9.4754e+01, 1.8853e+00,
        8.9905e+00, 1.8767e+00, 1.6170e+00, 2.2854e+02, 7.6418e+00, 1.9120e+00,
        4.0921e-01, 1.2663e+01, 7.7863e+01, 8.5334e+02, 6.3183e+02, 6.1164e+00,
        1.5785e+02, 4.3255e+02])