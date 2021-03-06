from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

import os
import sys
# LIBPATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(LIBPATH)

from nlu_tussie.components import Component


class Featurizer(Component):

    @staticmethod
    def _combine_with_existing_text_features(message,
                                             additional_features):
        if message.get("text_features") is not None:
            return np.hstack((message.get("text_features"),
                              additional_features))
        else:
            return additional_features
