import os
import os.path as opt
import sys

from nlu_tussie.model import Interpreter
import json

os.chdir(os.path.dirname(__file__))


def run_cmdline(model_path):
    interpreter = Interpreter.load(model_path)
    while True:
        text = input().strip()
        output = interpreter.parse(text)
        print(json.dumps(output, indent = 2))



if __name__ == '__main__':
    model_path = './out/default'
    latest_model = os.popen('ls %s| tail -n 1' % model_path)
    latest_model = latest_model.readlines()[0][:-1]
    run_cmdline(model_path = '%s/%s' % (model_path, latest_model))
