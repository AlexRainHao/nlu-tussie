'''
Useg: load Albert for service

this project based on `bert-as-service`

The extreme change is Albert model structure replaced for based Bert model
So, it is reasonable to start other bert model once using responding `modeling.py`.

'''
from argparse import ArgumentParser

parser = ArgumentParser(description='Start a Albert for serving')

parser.add_argument('-model_dir', type=str, required=True,
                    help='directory of a pretrained Albert model')

parser.add_argument('-ckpt_name', type=str, required=False,
                    default="albert_model.ckpt", help = "ckpt name for Albert model")

parser.add_argument('-config_name', type = str, required=False,
                    default="albert_config_tiny.json", help = "config name for Albert model")

parser.add_argument('-max_seq_len', type = int, required=False,
                    default = 25, help = "max sequence length for encoding")

parser.add_argument('-do_lower_case', type = bool, required=False,
                    default = True, help = "whether do lower case")

parser.add_argument("-num_worker", type = int, required=False,
                    default = 4, help = "number of server instances")

parser.add_argument("-max_batch_size", type = int, required=False,
                    default = 256, help = "batch size")

parser.add_argument("-cpu", type = bool, required=False,
                    default = False, help = "server deployment on GPU or CPU")

parser.add_argument("-http_max_connect", type = int, required=False,
                    default = 10, help = "maximum http server calls at same time")

parser_args = parser.parse_args()


class objdict(object):
    def __init__(self, _dic):
        self.__dict__ = _dic


def main(args):
    from server import BertServer
    with BertServer(args) as server:
        server.join()


if __name__ == '__main__':

    args = {'model_dir': parser_args.model_dir,
            'tuned_model_dir': None,
            'ckpt_name': parser_args.ckpt_name,
            'config_name': parser_args.config_name,
            'graph_tmp_dir': None,
            'max_seq_len': parser_args.max_seq_len,
            'do_lower_case': parser_args.do_lower_case,
            'pooling_layer': [-2],
            'pooling_strategy': 'REDUCE_MEAN',
            'mask_cls_sep': False,
            'no_special_token': False,
            'show_tokens_to_client': False,
            'no_position_embeddings': False,
            'port': 5555,
            'port_out': 5556,
            'http_port': 5557,
            'http_max_connect': parser_args.http_max_connect,
            'cors': '*',
            'num_worker': parser_args.num_worker,
            'max_batch_size': parser_args.max_batch_size,
            'priority_batch_size': 16,
            'cpu': parser_args.cpu,
            'xla': False,
            'fp16': False,
            'gpu_memory_fraction': 0.5,
            'device_map': [],
            'prefetch_size': 10,
            'fixed_embed_length': False,
            'verbose': False,
            'version': 'yyh_albert'}

    args = objdict(args)

    main(args)
