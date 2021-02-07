"""
a wrapper for pre-processing raw text

    default handler includes
        * normal -> no op conduct
        * digit_zero_flat -> convert all the digits to 0
        * digit_zero_shrink -> convert all the digits to special symbols as `__NUMBER__` with length of 1

    others self-defined handlers allowed by
        construct a dictionary with value of `callable` methods receiving raw text and return processed text
        and with key of a corresponding names
"""

from typing import Text, List, Dict, Union, Callable

import re

__all__ = ["Normalizers"]

class Normalizers:
    """
    main Normalizers op conduct

    E.X.
        assuming no self defined normalization need,
        and we want convert all single digit to 0, where shown as config of `digit_zero_flat`
        >>> text = "今天是2020年一月一号，我今天满18岁啦"
        >>> op = Normalizers()
        >>> config = {"digit_zero_flat": True}
        >>> method_list = op.parse_dict_config(config)
        >>> print(method_list)
            [ < function Normalizers.passHandler at 0x7fc1dc672158 >]
        >>> for _func in method_list:
                text = _func(text)
        >>> print(text)
                今天是0000年0月0号，我今天满00岁啦

        a series of self defined normalizations also supported
        we need created a dictionary with pairs of operation name and corresponding callable function,
        as
        >>> config = {
                    "your_name1": lambda x: x.replace("<url=", ""),
                    "your_name2: lambda x: x.replace("a", "b")
                    }

        and we could use
        >>> Normalizers(map_method = config).parse(["your_name1"])
        or
        >>> Normalizers().parse_dict_config(config = {"your_name1": True}, map_method = config)
        to return a list of callable normalizations method to conduct

        each normalization method could
        registered or removed dynamically to `Normalizers` class defaults method mapping dictionary
        with function `register_method` and `remove_method`
    """

    def __init__(self, map_method: Dict = {}):

        self.defaults = {"normal": self.passHandler,
                         "digit_zero_flat": self.zeroDigitsFlat,
                         "digit_zero_shrink": self.zeroDigitsShrink,
                         }

        for _name, _func in map_method.items():
            self.register_method(_name, _func)

    def register_method(self, name: Text, method: Callable):
        """pass"""

        self.defaults.update({name: method})

    def remove_method(self, name: Union[Text, List]):
        """pass"""
        if isinstance(name, Text):
            name = [name]

        for _name in name:
            if name in self.defaults:
                self.defaults.pop(_name)

    @staticmethod
    def passHandler(text, **kwargs):
        """no op"""
        return text

    @staticmethod
    def zeroDigitsFlat(text, **kwargs):
        """convert all the digits to 0"""
        if text.strip():
            pat = re.compile("[\d零幺一二三四五六七八九十百千万]")
            text = pat.sub("0", text)

        return text

    @staticmethod
    def zeroDigitsShrink(text, **kwargs):
        """convert all the digits to 0 and shrink span to length of 1"""

        if text.strip():
            pat = re.compile(r"[\d零幺一二三四五六七八九十百千万]+")
            text = pat.sub("__NUMBER__", text)

        return text

    @property
    def _default_handler(self):
        """return default handler"""
        return self.passHandler

    def parse(self, names: List[Text], map_method = None) -> List[Callable]:
        """
        determines each normalize handler by given names,
        and return a list of method used to call

        Parameters
        ----------
        names: List[Text], list of handler names needed to parse


        """
        map_method = map_method or self.defaults

        if len(names) == 0:
            return [self._default_handler]

        if isinstance(names, Text):
            # return map_method.get(names, cls._default_handler)
            names = [names]

        tar_methods = []
        for name in names:
            tar_methods.append(map_method.get(name, self._default_handler))

        return tar_methods

    def parse_dict_config(self, config: Dict, map_method: Dict = None):
        """determines handlers through dictionary config

        E.X.
            config = {"digit_norm": True,
                      "xxx": False}

        Parameters
        ----------
        map_method : self defined methods if needed
        """

        names = list(filter(config.get, config))

        return self.parse(names, map_method)