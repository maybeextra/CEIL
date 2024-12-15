from .transform_llcm import train_transformer_LLCM_rgb_weak, train_transformer_LLCM_rgb_strong, \
    train_transformer_LLCM_ir, transform_extract_LLCM, transform_extract_LLCM_f
from .transform_regdb import train_transformer_RegDB_rgb_weak, train_transformer_RegDB_rgb_strong, \
    train_transformer_RegDB_ir, transform_extract_RegDB, transform_extract_RegDB_f
from .transform_sysu import train_transformer_SYSU_rgb_weak, train_transformer_SYSU_rgb_strong, \
    train_transformer_SYSU_ir, transform_extract_SYSU, transform_extract_SYSU_f

transformers = {
    'SYSU': {
        'ir': train_transformer_SYSU_ir,
        'weak': train_transformer_SYSU_rgb_weak,
        'strong': train_transformer_SYSU_rgb_strong,
        'extract': [transform_extract_SYSU, transform_extract_SYSU_f]
    },
    'RegDB': {
        'ir': train_transformer_RegDB_ir,
        'weak': train_transformer_RegDB_rgb_weak,
        'strong': train_transformer_RegDB_rgb_strong,
        'extract': [transform_extract_RegDB, transform_extract_RegDB_f]
    },
    'LLCM': {
        'ir': train_transformer_LLCM_ir,
        'weak': train_transformer_LLCM_rgb_weak,
        'strong': train_transformer_LLCM_rgb_strong,
        'extract': [transform_extract_LLCM, transform_extract_LLCM_f]
    }
}

def create_transform(dataset, d_a_type):
    transform_rgb_2 = None

    if dataset in transformers:
        transform_ir = transformers[dataset]['ir']

        if d_a_type == 'fusion':
            transform_rgb_1 = transformers[dataset]['weak']
            transform_rgb_2 = transformers[dataset]['strong']
        else:
            transform_rgb_1 = transformers[dataset][d_a_type]
    else:
        raise KeyError("Unknown dataset:", dataset)

    return transform_rgb_1, transform_rgb_2, transform_ir

def creat_extract(dataset):
    transform = transformers[dataset]['extract']
    return transform