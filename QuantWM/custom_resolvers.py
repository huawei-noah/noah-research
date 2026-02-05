import hydra
from omegaconf import OmegaConf

@hydra.main(config_path=None)
def register_resolvers(cfg):
    pass

# Define the resolver function
def replace_slash(value: str) -> str:
    return value.replace('/', '_')

# Register the resolver with Hydra
OmegaConf.register_new_resolver("replace_slash", replace_slash)

if __name__ == "__main__":
    register_resolvers()

