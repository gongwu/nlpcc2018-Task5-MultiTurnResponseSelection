import tensorflow as tf
from configs.config_dialogue_selection_mbmn import process_config
from data_loader.dialogueSelection_data_generator import DialogueDataGenerator
from models.MBMN_model import MBMNModel
from trainers.MBMN_trainer import MBMNTrainer
from utils.dirs import create_dirs
from utils.logger import Logger
from utils.utils import get_args
import os
configfile = 'config_dialogue_selection_mbmn.json'


def main():
    # capture the config path from the run arguments
    # then process the json configration file
    try:
        args = get_args(configfile)
        config = process_config(args.config)
    except:
        print("missing or invalid arguments")
        exit(0)
    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir, config.dic_dir, config.result_dir])
    # create tensorflow session
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    gpu_config = tf.ConfigProto()
    gpu_config.allow_soft_placement = True
    gpu_config.gpu_options.allow_growth = True
    sess = tf.Session(config=gpu_config)
    # create your data generator
    data = DialogueDataGenerator(config)
    # create instance of the model you want
    # create tensorboard logger
    # create trainer and path all previous components to it
    model = MBMNModel(config, data)
    logger = Logger(sess, config)
    trainer = MBMNTrainer(sess, model, data, config, logger)
    # here you train your model
    trainer.do_test()


if __name__ == '__main__':
    main()
