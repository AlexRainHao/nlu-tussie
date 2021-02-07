"""pass"""

from typing import Dict, Text, List, Any, Set, Union, Optional

import numpy as np

def create_intent_dict(intents: Union[List, Set]) -> Dict:
    """obtain intent to index dictionary"""

    uni_intents = set(intents)

    return {x: idx for idx, x in enumerate(uni_intents)}


def create_intent_token_dict(intents: Union[List, Set], symbols: Optional[Text]):
    """pass"""

    tokens = set(token for intent in intents for token in intent.split(symbols))

    return {token: idx for idx, token in enumerate(tokens)}


def create_encoded_intent_bag(intent_dict: Dict,
                              symbols: Union[Text, None]) -> np.ndarray:
    """pass"""
    if symbols:
        uni_intents = list(intent_dict.keys())
        token_dict = create_intent_token_dict(uni_intents, symbols)

        encoded_intent_bag = np.zeros((len(uni_intents), len(token_dict)))

        for int, idx in intent_dict.items():
            for tok in int.split(symbols):
                encoded_intent_bag[idx, token_dict[tok]] = 1

        return encoded_intent_bag

    else:
        return np.eye(len(intent_dict))