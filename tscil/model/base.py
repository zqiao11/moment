
from momentfm import MOMENTPipeline, L2pMOMENTPipeline
import torch.nn as nn
from tscil.model.classifier import SingleHead, CosineLinear, SplitCosineLinear
from tscil.utils.setup_elements import input_size_match, n_classes_per_task, get_num_classes


def setup_model(args):
    data = args.data
    n_offline_base_nodes = get_num_classes(args)

    if 'L2P' not in args.agent:
        model = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-large",
            model_kwargs={
                'task_name': 'classification',
                'n_channels': input_size_match[data][1],  # number of input channels
                'num_class': n_offline_base_nodes,
                'freeze_encoder': args.freeze_encoder,  # Freeze the patch embedding layer
                'freeze_embedder': args.freeze_embedder,  # Freeze the transformer encoder
                'freeze_head': False,  # The linear forecasting head must be trained
                ## NOTE: Disable gradient checkpointing to supress the warning when linear probing the model as MOMENT encoder is frozen
                'enable_gradient_checkpointing': False,
                'reduction': args.reduction,
            },
        )

    else:
        model = L2pMOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-large",
            model_kwargs={
                'task_name': 'classification',
                'n_channels': input_size_match[data][1],  # number of input channels
                'num_class': n_offline_base_nodes,
                'freeze_encoder': args.freeze_encoder,  # Freeze the patch embedding layer
                'freeze_embedder': args.freeze_embedder,  # Freeze the transformer encoder
                'freeze_head': False,  # The linear forecasting head must be trained
                'enable_gradient_checkpointing': False,
                'reduction': args.reduction,

                # Prompted related arguments:
                'prompt_pool': args.prompt_pool,
                'use_prompt_mask': args.use_prompt_mask,
                'prompt_length': args.prompt_length,
                'embedding_key': args.embedding_key,
                'prompt_key': args.prompt_key,
                'prompt_key_init': args.prompt_key_init,
                'pool_size': args.pool_size,
                'top_k': args.top_k,
                'batchwise_prompt': args.batchwise_prompt
            },
        )

    model.init()

    return model.to(args.device)
