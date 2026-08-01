import os

import pynvml

PATTERN = "WORKER_{}"


# 为每个POD中每个进程均分GPUs
class GPUDispatcher:
    def __init__(self):
        pynvml.nvmlInit()
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if visible_devices:
            visible_devices = GPUDispatcher._unpack_gpus(visible_devices)
            self._gpus_num = len(visible_devices)
        else:
            self._gpus_num = pynvml.nvmlDeviceGetCount()
        assert self._gpus_num > 0
        self._per_proc_gpus_num = int(os.getenv("PER_PROC_GPUS", "-1"))
        if self._per_proc_gpus_num <= 0 or self._per_proc_gpus_num > self._gpus_num:
            self._per_proc_gpus_num = self._gpus_num
        assert self._gpus_num % self._per_proc_gpus_num == 0
        self._used = [False] * self.workers_num()
        self._users = {}

    def acquire(self, user):
        if user in self._users:
            return self._gpus_list(self._users[user])
        for index in range(len(self._used)):
            if not self._used[index]:
                self._used[index] = True
                self._users[user] = index
                return self._gpus_list(index)
        return None

    def release(self, user):
        if user in self._users:
            self._used[self._users[user]] = False
            del self._users[user]

    def workers_num(self):
        return int(self._gpus_num / self._per_proc_gpus_num)

    def gpus_num(self):
        return self._gpus_num

    def _gpus_list(self, index):
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if visible_devices:
            visible_devices = GPUDispatcher._unpack_gpus(visible_devices)
            gpus = []
            if self.workers_num() == 1:
                for i in range(self._gpus_num):
                    gpus.append(int(visible_devices[i]))
            else:
                for i in range(self._per_proc_gpus_num):
                    gpus.append(int(visible_devices[index * self._per_proc_gpus_num + i]))
        else:
            gpus = []
            if self.workers_num() == 1:
                for i in range(self._gpus_num):
                    gpus.append(i)
            else:
                for i in range(self._per_proc_gpus_num):
                    gpus.append(index * self._per_proc_gpus_num + i)
        assert len(gpus) > 0
        return gpus

    @staticmethod
    def _pack_gpus(gpus):
        res = []
        for gpu in gpus:
            res.append(str(gpu))
        return ",".join(res)

    @staticmethod
    def _unpack_gpus(gpus):
        tokens = gpus.split(",")
        res = []
        for token in tokens:
            res.append(int(token))
        return res

    @staticmethod
    def set_worker_gpus(pid, gpus):
        key = PATTERN.format(pid)
        os.environ[key] = GPUDispatcher._pack_gpus(gpus)

    @staticmethod
    def get_worker_gpus():
        key = PATTERN.format(os.getpid())
        gpus = os.getenv(key)
        # if gpus is not None:
        #   gpus = GPUDispatcher._unpack_gpus(gpus)
        if gpus == None:
            raise
        return gpus

    @staticmethod
    def del_worker_gpus(pid):
        key = PATTERN.format(pid)
        if key in os.environ:
            del os.environ[key]

    @staticmethod
    def bind_worker_gpus():
        gpus = GPUDispatcher.get_worker_gpus()
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
        # 特别注意：假如这里返回GPUs编号为[3, 4, 6, 7]，由于
        # 设置了CUDA_VISIBLE_DEVICES，进程内可用的GPU编号是0,1,2,3
        return GPUDispatcher._unpack_gpus(gpus)
