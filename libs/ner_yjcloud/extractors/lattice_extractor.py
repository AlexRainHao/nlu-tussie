"""
Usage: Flat Lattice Ner Extractor

ACL 2020 paper realization
    `FLAT: Chinese NER Using Flat-Lattice Transformer`
    https://arxiv.org/pdf/2004.11795.pdf

Results not great as expected

Not supporting train procedure temporarily
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from typing import Any, Dict, List, Optional, Text, Tuple
import logging
import os
import os.path as opt
import sys
import glob

import torch

LIBPATH = opt.abspath(opt.join(opt.abspath(__file__), *[opt.pardir] * 2, "models", "flat_lattice"))
sys.path.append(LIBPATH)

from ner_yjcloud.extractors import EntityExtractor, ConjuncEntityDynamic
from ner_yjcloud.model import Metadata
from ner_yjcloud.training_data import Message

# load model
from ner_yjcloud.models.flat_lattice.load_data import *
from ner_yjcloud.models.flat_lattice.V0 import predict


logger = logging.getLogger(__name__)

LATTICE_PATH = "lattice"

class LatticeNerEntityExtractor(EntityExtractor):
    name = "lattice_entity_extractor"

    provides = ["entities"]

    requires = []

    defaults = {"interest_entities": {"PER": "PER",
                                       "LOC": "RESIDENT",
                                       "TIME": "Date",
                                       "TITLE": "TITLE",
                                       "EDU": "EDU"},
                "pattern": {"ORG": [["[0-9日月号，。！,\.、?？]+(.*)", "clear"],
                                    ["(.*)[0-9]，。！,\.、?？[日月号]?", "clear"],
                                    ["(.*)例$", "clear"]],
                            "RESIDENT": [["[），。（,\.、？?!！]$", "clear"],
                                         ["^.{0,2}.$"], "clear"]},
                "confidence_threshold": 0.7,
                "cache_dir": "/home/user/yuanyh/Flat_Lattice_NER.bak/cache",
                "model_dir": "/home/user/yuanyh/Flat_Lattice_NER.bak/out"}


    def __init__(self, component_config = None, vocabs = None, embeddings = None,
                 w_trie = None, model = None, device = None):
        super(LatticeNerEntityExtractor, self).__init__(component_config)
        self.component_config = component_config
        
        self.user_cache_path = self.component_config.get("cache_dir")
        self.user_model_path = self.component_config.get("model_dir")
        
        self.vocabs = vocabs
        self.embeddings = embeddings
        self.w_trie = w_trie
        self.model = model
        self.device = device
        
        
    def train(self, training_data, config, **kwargs):
        logger.warning("Not supporting training procedure temporarily")
        self.component_config = config.for_component(self.name, self.defaults)
        
    
    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None
        # extracted = self.add_extractor_name(predict.predict(message))
        
        raw_entity = message.get("entities", [])
        text = message.text
        
        if text.strip() == '':
            return raw_entity
        
        # logger.info("*" * 40)
        # logger.info(self.device)
        
        # import time
        # b_t = time.time()
        pred_res = predict.predict(text = text,
                                   vocabs = self.vocabs,
                                   embeddings = self.embeddings,
                                   w_trie = self.w_trie,
                                   device = self.device,
                                   model = self.model)
        
        # logger.info(f"cost {time.time() - b_t}")
        
        for res in pred_res:
            res["entity"] = self.component_config["interest_entities"].get(res["entity"], "")
            if not res["entity"]:
                continue

            if len(res["value"]) < 1:
                continue                

            raw_entity.append(res)
            
        extracted = self.add_extractor_name(raw_entity)
        message.set("entities", extracted, add_to_output = True)
    
    
    @classmethod
    def create(cls, cfg):
        component_conf = cfg.for_component(cls.name, cls.defaults)
        model_dir = component_conf.get("model_dir", "")
        cache_dir = component_conf.get("cache_dir", "")
        
        if not model_dir or not cache_dir:
            logger.error(f"Failed to load embeddings cache and lattice model from \n{model_dir}\n{cache_dir}")
            raise Exception(f"Failed to load embeddings cache and lattice model from \n{model_dir}\n{cache_dir}")

        device = predict.load_device()
        torch.cuda.set_device(device)

        if "cpu" in device.type:
            logger.info("Using CPU to load lattice model")
        
        else:
            logger.info("Using CUDA to load lattice model")
        
        vocabs, embeddings, w_trie = predict.load_vocab(model_dir, cache_dir)
        model = predict.load_model(model_dir, device)
    
        return LatticeNerEntityExtractor(component_conf, vocabs, embeddings, w_trie, model, device)
    
    
    @classmethod
    def load(cls,
             model_dir = None, # type: Optional[Text]
             model_metadata = None, # type: Optional[Metadata]
             cached_component = None, # type: Optional[LatticeNerEntityExtractor]
             **kwargs # type: Any
             ):

        component_meta = model_metadata.for_component(cls.name)

        model_file = component_meta.get("model_dir", "%s/out" % LATTICE_PATH)
        cache_file = component_meta.get("cache_dir", "%s/cache" % LATTICE_PATH)

        lattice_model_dir = opt.join(model_dir, model_file)
        lattice_cache_dir = opt.join(model_dir, cache_file)

        if not opt.exists(lattice_model_dir) or not opt.exists(lattice_cache_dir):
            logger.error(f"Failed to load embeddings cache and lattice model from\n{lattice_model_dir}\n{lattice_cache_dir}")
            raise Exception(f"Failed to load embeddings cache and lattice model from\n{lattice_model_dir}\n{lattice_cache_dir}")

        device = predict.load_device()

        if "cpu" in device.type:
            logger.info("Using CPU to load lattice model")
        else:
            logger.info("Using CUDA to load lattice model")
            torch.cuda.set_device(device)


        vocabs, embeddings, w_trie = predict.load_vocab(lattice_model_dir, lattice_cache_dir)
        model = predict.load_model(lattice_model_dir, device)

        return cls(component_meta, vocabs, embeddings, w_trie, model, device)

    
    @staticmethod
    def copy_files_dir_to_dir(input_dir, output_dir):
        import shutil
        
        suffix = opt.basename(input_dir)
        output_dir = opt.join(output_dir, suffix)
        
        # make sure target path exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        target_file_list = glob.glob("{}/*".format(input_dir))
        for target_file in target_file_list:
            shutil.copy2(target_file, output_dir)
    
    
    def persist(self, model_dir):
        """The lattice model would save to sub-folder of persisted model named lattice"""
        lattice_path = opt.join(model_dir, LATTICE_PATH)
        
        if not opt.exists(lattice_path):
            os.mkdir(lattice_path)
        
        self.copy_files_dir_to_dir(self.user_model_path, lattice_path)
        self.copy_files_dir_to_dir(self.user_cache_path, lattice_path)
        
        
        # return {"model_dir": opt.join(lattice_path, "out"),
        #         "cache_dir": opt.join(lattice_path, "cache")}
        return {"model_dir": opt.join(LATTICE_PATH, "out"),
                "cache_dir": opt.join(LATTICE_PATH, "cache")}
