"""
help function for experiment.

"""

import time
import subprocess
from argparse import ArgumentParser

VIDEO_LENGTH = ['1']
VIDEO_FRAME = ['17']

MAIN_FILE_PATH = '/workspace/Two_Stream_PyTorch/project/main.py'

def get_parameters():
    '''
    The parameters for the model training, can be called out via the --h menu
    '''
    parser = ArgumentParser()

    # model hyper-parameters
    parser.add_argument('--model', type=str, default='multi', choices=['multi', 'single', 'multi_single'])
    parser.add_argument('--fusion', type=str, default='sum_loss', choices=['different_loss', 'sum_loss', 'concat'])

    # Training setting
    parser.add_argument('--gpu_num', type=int, default=0, choices=[0, 1], help='the gpu number whicht to train')

    # Transfor_learning
    parser.add_argument('--transfor_learning', action='store_true', help='if use the transformer learning')

    # pre process flag
    parser.add_argument('--pre_process_flag', action='store_true', help='if use the pre process video, which detection.')

    # Path
    parser.add_argument('--log_path', type=str, default='./logs', help='the lightning logs saved path')

    return parser.parse_known_args()


if __name__ == '__main__':

    config, unkonwn = get_parameters()

    transfor_learning = config.transfor_learning
    pre_process_flag = config.pre_process_flag
    model = config.model
    fusion = config.fusion

    if model == 'single' or model == 'multi_single':
        VIDEO_LENGTH = ['1']
        VIDEO_FRAME = ['31']

    symbol = '_'

    for length in VIDEO_LENGTH:
        for frames in VIDEO_FRAME:

            data = str(time.localtime().tm_mon) + str(time.localtime().tm_mday)
            
            if transfor_learning:
                
                if not pre_process_flag:

                    version = symbol.join([data, length, frames, 'not_pre_process'])
                    log_path = '/workspace/Two_Stream_PyTorch/logs/' + symbol.join([version, model, fusion]) + '.log'

                    with open(log_path, 'w') as f:

                        # start one train.
                        subprocess.run(['python', MAIN_FILE_PATH,
                                        '--version', version,
                                        '--model', model,
                                        '--fusion', fusion,
                                        '--clip_duration', length,
                                        '--uniform_temporal_subsample_num', frames,
                                        '--gpu_num', str(config.gpu_num),
                                        # '--pre_process_flag',
                                        '--transfor_learning',
                                        ], stdout=f, stderr=f)

                else:

                    version = symbol.join([data, length, frames])
                    log_path = '/workspace/Two_Stream_PyTorch/logs/' + symbol.join([version, model, fusion]) + '.log'

                    with open(log_path, 'w') as f:

                        # start one train.
                        subprocess.run(['python', MAIN_FILE_PATH,
                                        '--version', version,
                                        '--model', model,
                                        '--fusion', fusion,
                                        '--clip_duration', length,
                                        '--uniform_temporal_subsample_num', frames,
                                        '--gpu_num', str(config.gpu_num),
                                        '--pre_process_flag',
                                        '--transfor_learning',
                                        ], stdout=f, stderr=f)

            else:

                version = symbol.join([data, length, frames, 'not_transfor_learning'])
                log_path = '/workspace/Two_Stream_PyTorch/logs/' + symbol.join([version, model, fusion]) + '.log'

                with open(log_path, 'w') as f:

                    # start one train.
                    subprocess.run(['python', MAIN_FILE_PATH,
                                    '--version', version,
                                    '--model', model,
                                    '--fusion', fusion,
                                    '--clip_duration', length,
                                    '--uniform_temporal_subsample_num', frames,
                                    '--gpu_num', str(config.gpu_num),
                                    '--pre_process_flag',
                                    '--max_epochs', '100'
                                    # '--transfor_learning',
                                    ], stdout=f, stderr=f)

            print('finish %s' % log_path)
