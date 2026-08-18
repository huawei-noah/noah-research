import os
import sys
from URLs.dispatcher import GPUDispatcher

# gunicorn config & hook

gdp = GPUDispatcher()

port = os.environ.get('PORT')
infer_type = os.environ.get('INFER_TYPE')

bind = '127.0.0.1:' + port
workers = gdp.workers_num()

# determine deploy type, currently support 'vLLM' and 'transformers'
if infer_type == "vLLM":
    wsgi_app = 'URLs.vllm_url_m:app'
elif infer_type == "transformers":
    wsgi_app = 'URLs.transformers_url_m:app'

proc_name = 'infer' 
accesslog = '-'
timeout=300


def on_starting(server):
    server.gd = gdp
    server.log.info(
        "gunicorn app, gpus_num={}, workers_num={}, per_worker_gpus={}".format(
            gdp.gpus_num(), gdp.workers_num(), int(gdp.gpus_num() / gdp.workers_num())
        )
    )


def pre_fork(server, worker):
    worker.gpus = server.gd.acquire(worker.age)
    assert worker.gpus is not None


def post_fork(server, worker):
    server.gd.set_worker_gpus(worker.pid, worker.gpus)
    server.log.info(
        "server.age={}, worker.age={}, worker.pid={}, gpus={}".format(
            server.worker_age, worker.age, worker.pid, worker.gpus
        )
    )


def child_exit(server, worker):
    server.log.warning("worker exit. pid={}, gpus={}".format(worker.pid, worker.gpus))
    server.gd.del_worker_gpus(worker.pid)
    server.gd.release(worker.age)
