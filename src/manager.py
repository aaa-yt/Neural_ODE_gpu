import argparse
from logging import getLogger

from config import Config
from lib.logger import setup_logger

logger = getLogger(__name__)

CMD_LIST = ['predict', 'train']

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", help="What to do 'predict' or 'train'.", choices=CMD_LIST)
    parser.add_argument("-n", "--new", help="Create a new dataset.", action="store_true")
    return parser

def setup(config: Config, args):
    config.resource.create_directories()
    setup_logger(config.resource.main_log_path)
    config.load_parameter()

def start():
    parser = create_parser()
    args = parser.parse_args()
    config = Config()
    setup(config, args)

    if args.new:
        import data_processor
        data_processor.start(config)

    logger.info("command: {}".format(args.cmd))

    if args.cmd == 'train':
        import trainer
        return trainer.start(config)
    elif args.cmd == 'predict':
        import model_api
        return model_api.start(config)