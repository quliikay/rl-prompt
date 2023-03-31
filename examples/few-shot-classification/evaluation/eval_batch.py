import sys

import hydra
import pandas as pd

sys.path.append("..")
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from rlprompt.utils.utils import colorful_print
from fsc_helpers import (make_few_shot_classification_dataset,
                         get_dataset_verbalizers)
from fsc_evaluator import PromptedClassificationEvaluator

path = '../outputs/2023-03-30/19-12-49/outputs/1000/prompt_trigger_dic_train.csv'
path_out = './outputs/test.csv'
df = pd.read_csv(path)


@hydra.main(version_base=None, config_path="./", config_name="eval_config")
def main(config: "DictConfig"):
    colorful_print(OmegaConf.to_yaml(config), fg='red')
    (train_dataset, val_dataset, test_dataset,
     num_classes, verbalizers, template) = \
        make_few_shot_classification_dataset(config)
    print('Test Size', len(test_dataset))
    print('Examples:', test_dataset[:5])
    test_loader = DataLoader(test_dataset,
                             shuffle=False,
                             batch_size=32,
                             drop_last=False)

    is_mask_lm = True if 'bert' in config.task_lm else False
    verbalizers = get_dataset_verbalizers(config.dataset)
    num_classes = len(verbalizers)
    if config.dataset == 'agnews' and is_mask_lm:
        template = "<mask> {prompt} {sentence_1}"
    elif config.dataset == 'dbpedia' and is_mask_lm:
        template = "{prompt} <mask> : {sentence_1}"
    else:
        template = None

    for index, row in df.iterrows():
        prompt = row['prompt']
        trigger = row['trigger']
        tester = PromptedClassificationEvaluator(
            task_lm=config.task_lm,
            is_mask_lm=config.is_mask_lm,
            num_classes=num_classes,
            verbalizers=verbalizers,
            template=template,
            prompt=prompt,
            trigger=trigger,
            target=config.target
        )
        acc, asr = tester.forward(test_loader)
        print(f'prompt={prompt}, trigger={trigger}, acc={acc:.3f}, asr={asr:.3f}')
        df.loc[index, 'acc_test'] = acc.item()
        df.loc[index, 'asr_test'] = asr.item()
    df.to_csv(path_out, index=False)


if __name__ == "__main__":
    main()
