
from momentfm import MOMENTPipeline
import torch.nn as nn
from tscil.model.classifier import SingleHead, CosineLinear, SplitCosineLinear
from tscil.utils.setup_elements import input_size_match, n_classes_per_task, get_num_classes


# class SingleHeadModel(nn.Module):
#     def __init__(self, encoder, head, input_channels, feature_dims, n_layers, seq_len, n_base_nodes, norm, dropout):
#         super(SingleHeadModel, self).__init__()
#
#
#         self.encoder = encoder
#
#         if head == 'Linear':
#             self.head = SingleHead(in_features=feature_dims, out_features=n_base_nodes)
#         elif head in ['CosineLinear', 'SplitCosineLinear']:
#             self.head = CosineLinear(in_features=feature_dims, out_features=n_base_nodes)
#         else:
#             raise ValueError("Wrong head type")
#         self.head_type = head
#
#     def feature_map(self, x):
#         """
#         Return the feature map produced by encoder, (N, D, L)
#         """
#         feature_map = self.encoder(x, pooling=False)
#         return feature_map
#
#     def feature(self, x):
#         """
#         Return the feature vector after GAP, (N, D)
#         """
#         feature = self.encoder(x, pooling=True)
#         return feature
#
#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.head(x)
#         return x
#
#     def update_head(self, n_new, task_now=None):
#         if self.head_type == 'SplitCosineLinear':
#             assert task_now is not None
#             assert task_now > 0
#             if task_now == 1:
#                 in_features, out_features = self.head.in_features, self.head.out_features
#                 new_head = SplitCosineLinear(in_features, out_features, n_new)
#                 new_head.fc1.weight.data = self.head.weight.data
#                 new_head.sigma.data = self.head.sigma.data
#                 self.head = new_head
#             else:
#                 in_features = self.head.in_features
#                 out_features1 = self.head.fc1.out_features
#                 out_features2 = self.head.fc2.out_features
#                 new_head = SplitCosineLinear(in_features, out_features1 + out_features2, n_new)
#                 new_head.fc1.weight.data[:out_features1] = self.head.fc1.weight.data
#                 new_head.fc1.weight.data[out_features1:] = self.head.fc2.weight.data
#                 new_head.sigma.data = self.head.sigma.data
#                 self.head = new_head
#         else:
#             self.head.increase_neurons(n_new)


def setup_model(args):
    data = args.data
    n_offline_base_nodes = get_num_classes(args)

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
    model.init()
    return model.to(args.device)
