import transformers
import hydra 
from gpt2_trainer import Trainer

@hydra.main(version_base=None, config_path='../conf', config_name='gpt2_config.yaml')
def main(cfg):
    model_type = cfg.model.model_type
    policy = transformers.AutoModelForCausalLM.from_pretrained(model_type, device_map='cpu')
    trainer = Trainer(policy, cfg.train, cfg.model.seed)
    trainer.train()