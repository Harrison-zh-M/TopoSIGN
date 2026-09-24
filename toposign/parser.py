"""Unified argument parser for TopoSIGN experiments."""

import argparse
from toposign.configs.seed_config import TRAINING_SEEDS


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='TopoSIGN: Signed Graph Pre-Training and Prompt Learning')

    parser.add_argument('--dataset', type=str, default='setting1',
                        help='setting1 | setting2 | setting3 | setting4 | '
                             'settingN_runM | rainfall | sp1500 | all')

    parser.add_argument('--method', type=str, default='TopoMSGNN',
                        choices=[
                            'all',
                            'TopoMSGNN', 'TopoSSSNET', 'TopoSIGN', 'SSSNET', 'SigMaNet', 'MSGNN', 'DSGC',
                            'TopoMSGNN+GPPT', 'TopoMSGNN+Gprompt', 'TopoMSGNN+GPF', 'TopoMSGNN+All-in-one',
                            'GPPT', 'Gprompt', 'GPF',
                            'MSGNN+GPPT', 'MSGNN+Gprompt', 'MSGNN+GPF',
                            'All-in-one', 'SAMGPT', 'TopoDIG', 'DSGC+Topo', 'SSSNET+Topo', 'MSGNN+Topo', 'SigMaNet+Topo'
                        ],
                        help='Model method to run')

    parser.add_argument('--tda_type', type=str, default='signed',
                        choices=['signed', 'positive', 'none'],
                        help='Topological-feature filtration type')

    parser.add_argument('--no_structural', action='store_true',
                        help='Disable SGE branch (Topo-only ablation)')

    parser.add_argument('--pretrain_tasks', nargs='+', default=['SP'],
                        choices=['SP', 'DP', '3C', '4C', '5C'],
                        help='Pre-training tasks to use')
    parser.add_argument('--variant', type=str, default='all',
                        help='Optional table-specific variant filter')

    parser.add_argument('--prompt_method', type=str, default='GPPT',
                        choices=['GPPT', 'Gprompt', 'GPF', 'All-in-one', 'SAMGPT', 'TopoDIG'],
                        help='Prompt method for GFM fine-tuning')

    parser.add_argument('--backbone', type=str, default='TopoSIGN',
                        choices=['TopoSIGN', 'SSSNET', 'SigMaNet', 'MSGNN', 'DSGC', 'All-in-one', 'SAMGPT', 'TopoDIG'],
                        help='Encoder backbone for GFM methods')

    parser.add_argument('--hidden', type=int, default=32,
                        help='Hidden dimension for GNN layers')
    parser.add_argument('--topo_hidden', type=int, default=32,
                        help='Output dimension of the topological projection')
    parser.add_argument('--sigmanet_hidden', type=int, default=1,
                        help='Hidden dimension for SigMaNet baseline/backbone')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of MSConv layers')
    parser.add_argument('--K', type=int, default=1,
                        help='Chebyshev polynomial order')
    parser.add_argument('--q', type=float, default=0.25,
                        help='Magnetic-Laplacian charge parameter')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate')
    parser.add_argument('--normalization', type=str, default='sym',
                        choices=['sym', 'rw', None],
                        help='Laplacian normalization')
    parser.add_argument('-sd_input_features', '--sd_input_features',
                        dest='sd_input_features', action='store_true', default=True,
                        help='Use 4D signed-directed in/out degree input features')
    parser.add_argument('-no_sd_input_features', '--no_sd_input_features',
                        dest='sd_input_features', action='store_false')
    parser.add_argument('-weighted_input_features', '--weighted_input_features',
                        dest='weighted_input_features', action='store_true', default=True,
                        help='Weight input degree features by edge weights')
    parser.add_argument('-no_weighted_input_features', '--no_weighted_input_features',
                        dest='weighted_input_features', action='store_false')

    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='Fine-tuning learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='L2 weight decay')
    parser.add_argument('--early_stopping', type=int, default=40,
                        help='Early stopping patience (epochs)')
    parser.add_argument('--pretrain_batch_size', type=int, default=64,
                        help='Sampled link batch size for pre-training objectives')
    parser.add_argument('--pretrain_lr', type=float, default=1e-2,
                        help='Fixed learning rate for GFM pre-training (link-sign BCE)')
    parser.add_argument('--gppt_lr', type=float, default=2e-3,
                        help='Prompt fine-tuning learning rate for GPPT-style heads')
    parser.add_argument('--pixel_size', type=float, default=1.0,
                        help='Persistence-image pixel size')

    parser.add_argument('--pretrain_epochs', type=int, default=None,
                        help='Override pre-training epoch count')
    parser.add_argument('--lr_candidates', nargs='+', type=float, default=None,
                        help='Override LR search grid')

    parser.add_argument('--seeds', nargs='+', type=int, default=TRAINING_SEEDS,
                        help='Training seeds for multi-run averaging')

    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU even if GPU is available')
    parser.add_argument('--device_id', type=int, default=0,
                        help='CUDA device ID')

    parser.add_argument('-D', '--debug', action='store_true',
                        help='Debug mode: epochs=2, seeds=[0,10]')

    parser.add_argument('--results_dir', type=str, default=None,
                        help='Override results directory')
    parser.add_argument('--force', action='store_true',
                        help='Re-run completed jobs instead of resuming from saved results')
    parser.add_argument('--hparam_search', dest='hparam_search', action='store_true',
                        default=None,
                        help='Search over learning rates before final evaluation')
    parser.add_argument('--no_hparam_search', '--no-hparam-search',
                        dest='hparam_search', action='store_false',
                        help='Disable learning-rate search')

    return parser


def parse_args(argv=None) -> argparse.Namespace:
    parser = get_parser()
    args = parser.parse_args(argv)

    if args.debug:
        args.epochs = 2
        args.seeds  = [0, 10]

    args.use_structural = not args.no_structural
    args.sd_input_feat = args.sd_input_features
    args.weighted_input_feat = args.weighted_input_features

    return args
