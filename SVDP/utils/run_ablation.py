import subprocess
import os 
import logging 

from configure_logger import configure_logger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
configure_logger(logger, "run_ablation.log")

MODEL_PATH = os.environ["MODEL_PATH"]
MODEL_NAME = os.environ["MODEL_NAME"]
BENCHMARKS=os.environ["BENCHMARKS"]
DEPLOY_PORT=os.environ["DEPLOY_PORT"]
RANK=int(os.environ["RANK"])

case_study_dict = {
    "naive_svd" : {
        "ablation_config_whitening" : "w/o",
        "ablation_config_bias" : "w/o",
        "ablation_config_penalty": "gate"
    },
    "data_whitening" : {
        "ablation_config_whitening" : "cholesky",
        "ablation_config_bias" : "w/o",
        "ablation_config_penalty": "gate"
    },
    "bias_calibration_simple" : {
        "ablation_config_whitening" : "cholesky",
        "ablation_config_bias" : "full",
        "ablation_config_penalty": "gate"
    },
    "bias_calibration_full" : {
        "ablation_config_whitening" : "cholesky",
        "ablation_config_bias" : "full",
        "ablation_config_penalty": "full"
    },

}

logger.info(f"Running Ablation for {MODEL_PATH}!")

logger.info("<STEP> Evaluate baseline")
os.environ["IMPL_NAME"]=MODEL_NAME+"_baseline"
os.environ["TAG_NAME"]=MODEL_NAME+"_baseline"
subprocess.run(["./scripts/evaluate_quality.sh",])

for rank,s in ((1024,0.8), (RANK,0.5)):
    for case_study, params in case_study_dict.items():
        logger.info(f"<STEP> Ablation - {case_study} at rank {rank}")
        logger.info(f"<STEP> Constuct predictors - {case_study} at rank {rank}")
        os.environ["PREDICTORS_PATH"]=f"weights/ablation/{MODEL_NAME}/{case_study}_r{rank}_s{s}"
        subprocess.run(["python3", "./utils/construct_predictors.py", 
                        "--model_path", MODEL_PATH,
                        "--predictors_output_path", os.environ["PREDICTORS_PATH"],
                        "--rank", f"{rank}",
                        "--s", f"{s}",
                        "--sparsity_plot_output_file", "outputs/tmp.pdf",
                        "--calibration_prompts_path", "calibration_prompts.json",
                        "--torch_dtype", "float16",
                        "--device_map", "cuda:0",
                        "--ablation_config_whitening", params["ablation_config_whitening"],
                        "--ablation_config_bias", params["ablation_config_bias"],
                        "--ablation_config_penalty", params["ablation_config_penalty"],
                    ])
        
        logger.info(f"<STEP> Evaluate - {case_study} at rank {rank}")
        os.environ["IMPL_NAME"]=MODEL_NAME+"_svd_predictors"
        os.environ["TAG_NAME"]=MODEL_NAME+f"_svd_predictors_{case_study}_r{rank}_s{s}"
        subprocess.run(["./scripts/evaluate_quality.sh",])