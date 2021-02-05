import os.path as opt
import os
import sys

# 确保能 import 到 `ner_yjcloud`
sys.path.append(opt.abspath(opt.join(__file__, *[opt.pardir] * 2, "libs")))

os.chdir(os.path.dirname(__file__))

from ner_yjcloud.config import RasaNLUModelConfig
from ner_yjcloud.training_data.loading import load_data
from ner_yjcloud import config as nlu_config
from ner_yjcloud.model import Trainer
from ner_yjcloud.model import Metadata, Interpreter


def train_nlu(data = './data/case-search-demo.json',
              config = './config/config_crfpp.yml',
              model_dir = './out'):

    if "config_normal" in config:
        raise ValueError("no supported training for this config file")

    training_data = load_data(data)
    trainer = Trainer(nlu_config.load(config))
    trainer.train(training_data)
    model_directory = trainer.persist(model_dir)

if __name__ == '__main__':
    train_nlu()
#     # print("yyh")
