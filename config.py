import argparse

parser = argparse.ArgumentParser(description='Hyper-parameters management')

parser.add_argument("--num_workers", type=int, default=8,
                    help="number of workers")
parser.add_argument('--cpu', action='store_true',
                    help='use cpu only')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')

parser.add_argument('--ckpt', type=str, default='ckpt',
                    help='the dir path to save model weight')
parser.add_argument('--w', type=str, default='',
                    help='the path of model wight to test or reload')
parser.add_argument('--suf', type=str, choices=['.dcm', '.JL', '.png'], default='.png',
                    help='suffix')
parser.add_argument('--eval', action="store_true",
                    help='eval only need weight')
parser.add_argument('--test_root', type=str,
                    help='root_dir')
parser.add_argument('--name', type=str, default='setr',
                    help='study name')
parser.add_argument('--input_size', type=int, default=512,
                    help='input size')


parser.add_argument("--num_heads", type=int, default=4,
                    help='the number of heads')
parser.add_argument("--num_layers", type=int, default=4,
                    help="the number of encoder block")
parser.add_argument("--patch_h", type=int, default=16,
                    help="patch height")
parser.add_argument("--patch_w", type=int, default=16,
                    help="patch width")

parser.add_argument("--emb_dim", type=int, default=768,
                    help='embedding dimension')
parser.add_argument("--mlp_dim", type=int, default=2048,
                    help="mlp dimension")
parser.add_argument("--attn_dropout_rate", type=float, default=0.1,
                    help="attn dropout rate")
parser.add_argument("--dropout_rate", type=float, default=0.1,
                    help="dropout rate")
parser.add_argument("--channels", type=int, default=3,
                    help="channel to recive continuous CT slices")

parser.add_argument('--dataset_path', default='./dataset',
                    help='trainset root path')
parser.add_argument('--batch_size', type=int, default=4,
                    help='batch size')
parser.add_argument('--max_epoch', type=int,
                    default=128)
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate (default: 0.001)')

args = parser.parse_args()
