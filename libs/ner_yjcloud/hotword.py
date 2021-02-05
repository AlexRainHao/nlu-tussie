"""
This file support add self-define token word and corresponding entity name

could receive 2 main object:
    1. label: str, for register a label, E.X. "married-relation"
    2. dict file name: path str, for add hot token to dictionary, each line has a format as follows
        夫妻 married-relation

Has not conduct QA Test yet.
"""
import json
import codecs
import os.path as opt

MODEL_PATH = opt.abspath(opt.join(__file__, opt.pardir, opt.pardir, opt.pardir,
                                  "model/NerModel/default/model_20200710-015956"))

class hotWord():
    def __init__(self, model_path = MODEL_PATH):
        
        raise NotImplementedError('This method has not tested yet')
        
        self._check_path(model_path)

        self.model_path = model_path
        self.load_model_config()
        
    def _check_path(self, path):
        if not opt.exists(path):
            raise FileNotFoundError("No model path found")
        
    def load_model_config(self):
        self.metaconfig = opt.join(self.model_path, "metadata.json")
        
        try:
            self.raw_content = json.loads(codecs.open(self.metaconfig, encoding = 'utf-8').read())
            pipelines = self.raw_content["pipeline"]
            
            for idx, pipe in enumerate(pipelines):
                if pipe.get('name') == "spacy_entity_extractor":
                    self.idx = idx
                    self.pipe = pipe
                    break
        except:
            raise Exception(f"No model config found: {self.metaconfig}")
            
    
    def register_label(self, label):
        if isinstance(label, str):
            label = [label]
        elif isinstance(label, list):
            pass
        
        for _label in label:
            self.pipe["mapping_label_dict"][_label] = _label
            self.pipe["interest_entities"][_label] = [_label]
    
    
    def register_dict(self, fname,
                      target_file = opt.join(MODEL_PATH, "tokenizer_spacy/similary_case.txt")):
        if isinstance(fname, str):
            fname = [fname]
        elif isinstance(fname, list):
            pass
        
        idx = 0
        
        with codecs.open(target_file, 'a', encoding = 'utf-8') as f:
            for _f in fname:
                src_lines = codecs.open(_f, encoding = 'utf-8').readlines()
                idx += len(src_lines)
                f.writelines(src_lines)

        print(f'Have Update self-defined token: {idx}')
        
    
    def dump_config(self):
        self.raw_content["pipeline"][self.idx] = self.pipe
        self.metaconfig = opt.join(self.model_path, "metadata.json")
        with codecs.open(self.metaconfig, 'w', encoding = 'utf-8') as f:
            json.dump(self.raw_content, f, ensure_ascii = False)
        
        
# if __name__ == '__main__':
#     a = hotWord()
#     a.register_label("LOAN")
#     a.register_label("Relations")
#     a.register_dict("/your/path/dict")
#     a.dump_config()