"""
sample_predict
"""
import numpy as np

def sample_predict(model, inp):
    """Run predictions

    :param model: object to call predict method upon
    :param inp: input
    """
    return np.squeeze(model.predict(inp[np.newaxis, :, :]), axis=0)
    