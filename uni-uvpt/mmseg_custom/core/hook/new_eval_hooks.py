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


import os.path as osp
import os

import torch.distributed as dist
from mmcv.runner import DistEvalHook as _DistEvalHook
from mmcv.runner import EvalHook as _EvalHook
from torch.nn.modules.batchnorm import _BatchNorm
import pandas as pd
import warnings


class NewEvalHook(_EvalHook):
    """Single GPU EvalHook, with efficient test support.

    Args:
        by_epoch (bool): Determine perform evaluation by epoch or by iteration.
            If set to True, it will perform by epoch. Otherwise, by iteration.
            Default: False.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Default: False.
    Returns:
        list: The prediction results.
    """

    greater_keys = ['mIoU', 'mAcc', 'aAcc']

    def __init__(self, *args, by_epoch=False, efficient_test=False, **kwargs):
        super().__init__(*args, by_epoch=by_epoch, **kwargs)
        self.efficient_test = efficient_test

    def evaluate(self, runner, results):
        """Evaluate the results.

        Args:
            runner (:obj:`mmcv.Runner`): The underlined training runner.
            results (list): Output results.
        """
        eval_res = self.dataloader.dataset.evaluate(
            results, logger=runner.logger, efficient_test=True, **self.eval_kwargs)

        for name, val in eval_res.items():
            runner.log_buffer.output[name] = val
        runner.log_buffer.ready = True

        if self.save_best is not None:
            # If the performance of model is pool, the `eval_res` may be an
            # empty dict and it will raise exception when `self.save_best` is
            # not None. More details at
            # https://github.com/open-mmlab/mmdetection/issues/6265.
            if not eval_res:
                warnings.warn(
                    'Since `eval_res` is an empty dict, the behavior to save '
                    'the best checkpoint will be skipped in this evaluation.')
                return None

            if self.key_indicator == 'auto':
                # infer from eval_results
                self._init_rule(self.rule, list(eval_res.keys())[0])
            return eval_res[self.key_indicator]

        return None

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        if not self._should_evaluate(runner):
            return

        from mmseg.apis import single_gpu_test
        print(self.dataloader.dataset.img_dir)
        results = single_gpu_test(
            runner.model,
            self.dataloader,
            show=False,
            efficient_test=self.efficient_test)
        if type(results[0]) == tuple:
            if type(results[0]) == tuple:
                for i in range(len(results)):
                    results[i] = results[i][1][0]
        runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
        mode = "train" if "train" in self.dataloader.dataset.img_dir else "val"
        
        key_score = self.evaluate(runner, results)
        if not osp.exists(os.path.join(self.out_dir, mode + "_IoURecord.csv")):
            Cate_IoU = {k: [runner.log_buffer.output[k]] for k in runner.log_buffer.output if "IoU" in k}
            Cate_IoU["mIoU_SYNTHIA16"] = [Cate_IoU["mIoU"][0] * 19 / 16]
            Cate_IoU["mIoU_SYNTHIA13"] = [(Cate_IoU["mIoU"][0] * 19 -
                                           (Cate_IoU["IoU.wall"][0] +
                                            Cate_IoU["IoU.fence"][0] + Cate_IoU["IoU.pole"][0])) / 13]
            pd.DataFrame(Cate_IoU).to_csv(os.path.join(self.out_dir, mode + "_IoURecord.csv"), index=False)
        else:
            Cate_IoU = {k: runner.log_buffer.output[k] for k in runner.log_buffer.output if "IoU" in k}
            Cate_IoU["mIoU_SYNTHIA16"] = Cate_IoU["mIoU"] * 19 / 16
            Cate_IoU["mIoU_SYNTHIA13"] = (Cate_IoU["mIoU"] * 19 - (Cate_IoU["IoU.wall"] +
                                                                   Cate_IoU["IoU.fence"] + Cate_IoU["IoU.pole"])) / 13
            data = pd.read_csv(os.path.join(self.out_dir, mode + "_IoURecord.csv"))
            data.loc[len(data)] = Cate_IoU
            data.to_csv(os.path.join(self.out_dir, mode + "_IoURecord.csv"), index=False)
        if self.save_best and mode == "val":
            self._save_ckpt(runner, key_score)


