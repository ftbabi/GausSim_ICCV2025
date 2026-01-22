# Copyright (c) Open-MMLab. All rights reserved.
import json
import datetime
from collections import OrderedDict
from typing import Dict, Optional, Union

import torch

import mmcv
from mmcv.runner import HOOKS, TextLoggerHook


@HOOKS.register_module()
class CusTextLoggerHook(TextLoggerHook):
    def __init__(self,
                by_epoch: bool = True,
                interval: int = 10,
                ignore_last: bool = True,
                reset_flag: bool = False,
                interval_exp_name: int = 1000,
                out_dir: Optional[str] = None,
                out_suffix: Union[str, tuple] = ('.log.json', '.log', '.py'),
                keep_local: bool = True,
                file_client_args: Optional[Dict] = None,
                indent: int = 4,
                depth: int = 3):
        super().__init__(
            by_epoch=by_epoch, interval=interval, ignore_last=ignore_last,
            reset_flag=reset_flag, interval_exp_name=interval_exp_name,
            out_dir=out_dir, out_suffix=out_suffix, keep_local=keep_local, file_client_args=file_client_args)
        
        self.indent = indent
        self.depth = depth

    def customized_output(self, log_dict, lv_splitter='.'):
        # Parse dict
        rst_dict = OrderedDict()
        for key, val in log_dict.items():
            k_parts = key.split(lv_splitter)
            cur_dict = self.register_level_key(lv_splitter, k_parts, f"{val}", depth=self.depth)
            self.merge_nested_dict(rst_dict, cur_dict)

        # log_str = self.format_log_dict(rst_dict, depth=self.depth)
        log_str = "\n" + "*"*100 + json.dumps(rst_dict, indent=4) + "*"*100 + "\n"
        return log_str
    
    def register_level_key(self, lv_splitter, key_list, val, depth):
        if depth <= 1:
            cur_key = lv_splitter.join(key_list)
            key_list = [cur_key]

        cur_dict = OrderedDict()
        if len(key_list) == 1:
            cur_dict[key_list[0]] = val
        else:
        # if len(key_list) > 1:
            out_dict = self.register_level_key(lv_splitter, key_list[1:], val, depth-1)
            cur_dict[key_list[0]] = out_dict
        return cur_dict
    
    def merge_nested_dict(self, dict_a, dict_b):
        for key, val in dict_b.items():
            if isinstance(val, dict):
                if key not in dict_a.keys():
                    dict_a[key] = OrderedDict()
                self.merge_nested_dict(dict_a[key], val)
            else:
                assert key not in dict_a.keys()
                dict_a[key] = val        

    def _log_info(self, log_dict: Dict, runner) -> None:
        # print exp name for users to distinguish experiments
        # at every ``interval_exp_name`` iterations and the end of each epoch
        if runner.meta is not None and 'exp_name' in runner.meta:
            if (self.every_n_iters(runner, self.interval_exp_name)) or (
                    self.by_epoch and self.end_of_epoch(runner)):
                exp_info = f'Exp name: {runner.meta["exp_name"]}'
                runner.logger.info(exp_info)

        if log_dict['mode'] == 'train':
            if isinstance(log_dict['lr'], dict):
                lr_str = []
                for k, val in log_dict['lr'].items():
                    lr_str.append(f'lr_{k}: {val:.3e}')
                lr_str = ' '.join(lr_str)  # type: ignore
            else:
                lr_str = f'lr: {log_dict["lr"]:.3e}'  # type: ignore

            if self.by_epoch:
                log_str = f'Epoch [{log_dict["epoch"]}]' \
                          f'[{log_dict["iter"]}/{len(runner.data_loader)}]\t'
            else:
                log_str = f'Iter [{log_dict["iter"]}/{runner.max_iters}]\t'
            log_str += f'{lr_str}, '

            if 'time' in log_dict.keys():
                self.time_sec_tot += (log_dict['time'] * self.interval)
                time_sec_avg = self.time_sec_tot / (
                    runner.iter - self.start_iter + 1)
                eta_sec = time_sec_avg * (runner.max_iters - runner.iter - 1)
                eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
                log_str += f'eta: {eta_str}, '
                if 'data_time' in log_dict.keys(): # Temporarily do this due to the iterbased runner
                    log_str += f'time: {log_dict["time"]:.3f}, ' \
                            f'data_time: {log_dict["data_time"]:.3f}, '
                # statistic memory
                if torch.cuda.is_available():
                    log_str += f'memory: {log_dict["memory"]}, '
        else:
            if self.by_epoch:
                log_str = f'Epoch({log_dict["mode"]}) ' \
                    f'[{log_dict["epoch"]}][{log_dict["iter"]}]\t'
            else:
                log_str = f'Iter({log_dict["mode"]}) [{log_dict["iter"]}]\t'

        log_items = dict()
        for name, val in log_dict.items():
            if name in [
                    'mode', 'Epoch', 'iter', 'lr', 'time', 'data_time',
                    'memory', 'epoch'
            ]:
                continue
            if isinstance(val, float):
                val = f'{val:.4f}'
            # log_items.append(f'{name}: {val}')
            log_items[name] = val

        log_str += self.customized_output(log_items)

        runner.logger.info(log_str)
