

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
              default='/cache/', required=True)
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

    train_script = './pretrain.py'
    train_script = pathlib.Path(__file__).parent / train_script

    commands = []

    config_names = [
        'daformer_swin_base_patch4_window7_512x512_400k_6e-6_gta.py',
        'daformer_swin_base_patch4_window7_512x512_400k_2e-6_synthia.py',
        'daformer_mit-b5_512x512_400k_2e-5_gta.py',
        'daformer_mit-b5_512x512_400k_2e-6_synthia.py',
        'GtA_daformer_swin_base_patch4_window7_512x512_40k_1e-5_gta.py',
        'GtA_daformer_swin_base_patch4_window7_512x512_40k_2e-6_synthia.py',
        'GtA_daformer_mit-b5_512x512_40k_8e-6_gta.py',
        'GtA_daformer_mit-b5_512x512_40k_4e-6_synthia.py'
    ]

    for config_name in config_names:
        config = os.path.join('configs/pretrain', config_name)
    
        work_dir = os.path.join(train_url, config_name)

        if not os.path.exists(work_dir):
            os.makedirs(work_dir) 
        
        source_name = 'gta' if 'gta' in config_name else 'synthia'
        data_root_source = os.path.join(data_url, source_name)
        data_root_target = os.path.join(data_url, 'cityscapes')

        cmd = f'python {train_script.absolute().as_posix()} {config} ' \
                f'--data-root-source {data_root_source}  ' \
                f'--data-root-target {data_root_target}  ' \
                f'--work-dir {work_dir}  ' 

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
