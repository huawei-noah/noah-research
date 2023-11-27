
# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

#
import os
import click
import torch
import pathlib
import subprocess
from typing import List
import torch
import time

os.chdir(os.path.split(os.path.realpath(__file__))[0])



@click.command(context_settings={"ignore_unknown_options": True})
@click.option('--init_method', default='0:5013', help='tcp_port', required=False)
@click.option('--run_test', default=True, type=bool)
@click.option('--data_url', help='data root dir', type=click.Path(exists=True),
              default='/cache/cityscapes/', required=True)
@click.option('--train_url', help='output root dir', 
              default='./tmp_exp', required=True)
@click.argument('command_args', nargs=-1)
def main(init_method: str,
         data_url: str,
         train_url: str,
         run_test,
         command_args: List[str]):

    if not os.path.exists(train_url):
        os.makedirs(train_url)

    ngpus = torch.cuda.device_count()
    print(f'>-----------> init_method: {init_method}')
    master_addr, master_port = init_method.replace('tcp://', '').split(':')

    if master_addr == '' or master_addr is None:
        master_addr = '192.168.0.8'

    command_args = ' '.join(command_args)

    train_script = './train.py'
    train_script = pathlib.Path(__file__).parent / train_script

    commands = []

    pretrained_models = [
        'SSS_Swin-B_GTAV_Cityscapes_50.5.pth',
        'SSS_Swin-B_Synthia_Cityscapes_44.6_49.8.pth',
        'GtA_Swin-B_GTAV_Cityscapes_52.8.pth',
        'GtA_Swin-B_Synthia_Cityscapes_48.8_55.0.pth',
        'SSS_MiT-B5_GTAV_Cityscapes_52.5.pth',
        'SSS_MiT-B5_Synthia_Cityscapes_48.6_55.0.pth',
        'GtA_MiT-B5_GTAV_Cityscapes_53.0.pth',
        'GtA_MiT-B5_Synthia_Cityscapes_50.0_56.2.pth'
    ]

    for pretrained_model in pretrained_models:
        
        if 'Swin' in pretrained_model:
            config_names = [
                            'X_to_cityscapes_swin_daformer_prompt_multiscale.py',
            ]
        elif 'MiT' in pretrained_model:
            config_names = [
                            'X_to_cityscapes_mit_b5_daformer_prompt_multiscale.py',

            ]

        for config_name in config_names:
            pretrained_from = os.path.join('model/pretrained_models/', pretrained_model)
            config = os.path.join('configs/Uni-UVPT', config_name)
        
            work_dir = os.path.join(train_url, pretrained_model + '_' + config_name)
            model_name = "MiT" if "mit" in config else "Swin"
            lr = 4e-6 if "mit" in config else 6e-6

            if not os.path.exists(work_dir):
                os.makedirs(work_dir) 

            cmd = f'python {train_script.absolute().as_posix()} {config} ' \
                f'--data-root {data_url} ' \
                f'--work-dir {work_dir} --lr {lr} ' \
                f'--pretrained_from {pretrained_from} ' \
                f'--log-freq 200 ' \
                f'--eval-freq 2000 ' \
                f'--pseudo-update online ' \
                f'--slope-diff-threshold 0.95 ' \
                f'--trustable-quantile 0.80 ' 

            commands.append(cmd)


    available_gpus = [str(x) for x in range(torch.cuda.device_count())]
    n_gpus = len(available_gpus)
    procs_by_gpu = [None]*n_gpus
    while True:
        while len(commands) > 0:
            for idx, gpu_idx in enumerate(available_gpus):
                proc = procs_by_gpu[idx]
                if (proc is None) or (proc.poll() is not None):
                    # Nothing is running on this GPU; launch a command.
                    cmd = commands.pop(0)
                    print(cmd)
                    new_proc = subprocess.Popen(
                        f'CUDA_VISIBLE_DEVICES={gpu_idx} {cmd}', shell=True)
                    procs_by_gpu[idx] = new_proc
                    break
            
            time.sleep(1)

        flag = 0
        for proc in procs_by_gpu:
            if proc.poll() is not None:
                flag+=1
        if flag == n_gpus:
            exit()
        else:
            time.sleep(40)
            
if __name__ == '__main__':
    main()
