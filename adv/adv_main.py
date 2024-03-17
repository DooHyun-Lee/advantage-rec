from torch.utils.data import DataLoader
import hydra
import os
from dataset_adv import total_sequence, AdvDataset
from model import AdvFunc
from train import Trainer

@hydra.main(version_base=None, config_path='../conf', config_name='adv_config.yaml')
def main(cfg):
    base_model_type = cfg.data.base_model_type
    dataset_name = cfg.data.dataset_name + '.txt'
    dataset_path = os.path.join('../data', base_model_type, dataset_name)
    total_sequences, total_item = total_sequence(dataset_path)

    trainDataset = AdvDataset(cfg.data, total_sequences, total_item, 'train')
    validDataset = AdvDataset(cfg.data, total_sequences, total_item, 'valid')
    testDataset = AdvDataset(cfg.data, total_sequences, total_item, 'test')
    trainLoader = DataLoader(trainDataset, batch_size=cfg.data.batch_size,shuffle=True)
    validLoader = DataLoader(validDataset, batch_size=cfg.data.batch_size,shuffle=False)
    testLoader = DataLoader(testDataset, batch_size=cfg.data.batch_size,shuffle=False)

    adv_func = AdvFunc(total_item, cfg.model)
    model_name = f'{base_model_type}_{cfg.data.dataset_name}_{cfg.data.model_idx}.pt'
    trainer = Trainer(total_sequences, total_item, adv_func, 
                      trainLoader, validLoader, testLoader, model_name, cfg.train)

    trainer.train()
    trainer.test(eval_type='test')

if __name__ == "__main__":
    main()