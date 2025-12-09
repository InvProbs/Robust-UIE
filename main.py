"""
PhyNN-fixpoint: a robust UIE framework
created by: Peimeng Guan
"""
import matplotlib, csv,  sys

from networks import UIEunfoldext_net as nets
from utils.dataloader import *
from utils.misc import *
import configargparse, os
from tqdm import tqdm
from pandas import *
from utils.training_setup import *
from operators import training_mode as tr

parser = configargparse.ArgParser()
parser.add_argument("--file_name", type=str, default="2-UIEB/PhyNN_fixpoint/", help="saving folder name")
parser.add_argument("--save_path", type=str, default="../saved_models/", help="network saving directory")
parser.add_argument("--load_path", type=str, default='', help='model path')
parser.add_argument('--dataset', type=str, default="UIEB", help='dataset name')
parser.add_argument('--data_portion', type=float, default=1, help='amount of training data in percentage')
parser.add_argument("--data_path", type=str, default="../data/UIEB/", help='path to dataset')

parser.add_argument('--maxiters', type=int, default=3, help='max fixpoint iterations')
parser.add_argument('--n_epochs', type=int, default=200)
parser.add_argument('--kernel_size', type=int, default=3, help='conv layer kernel size')
parser.add_argument('--padding', type=int, default=1, help='conv layer padding')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
parser.add_argument('--batch_size_val', type=int, default=4, help='Batch size')
parser.add_argument("--device", type=str, default="cuda", help="cpu or cuda")

parser.add_argument('--pretrain', type=bool, default=True, help='if load utils and resume training')
parser.add_argument('--train', type=bool, default=False, help='training or eval mode')
parser.add_argument('--test_mode', type=str, default="noisy", help='MSE, AT, noisy')
parser.add_argument('--noise_level', type=float, default=0.05, help='Gaussian attack variance')

parser.add_argument('--epsilon', type=float, default=5, help="adv. attack level, ||e||<=epsilon")
parser.add_argument('--n_PGD_steps', type=int, default=5, help='max num PGD steps for adv. attack')
parser.add_argument('--PGD_stepsize', type=float, default=2, help='max num PGD steps for adv. attack')
args = parser.parse_args()
args.shared_eta = True
print(args)
cuda = True if torch.cuda.is_available() else False
print('cuda available: ' + str(cuda))
set_seed(1)
if args.location != 'pace': matplotlib.use("Qt5Agg")

""" LOAD DATA """
args.batch_size_val = args.batch_size_val if args.test_mode != 'AT' else 1
tr_loader, tr_length, val_loader, val_length, ts_loader, ts_length = load_UIEB(args)
set_save_path(args)

""" DEFINE MODELS """
net = nets.PhyNN_fixpoint(args).to(args.device)
optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
criteria = nn.MSELoss()
criteria_l1 = nn.L1Loss()

if args.pretrain:
    load_path = '../saved/model/path/'

    net.load_state_dict(torch.load(load_path)['state_dict'])
    print('load_path: ' + load_path)

""" BEGIN TRAINING """
if args.train:
    # set training loss saving files
    trajectory_path = args.save_path + '/trajectory.csv'
    fh = open(trajectory_path, 'a')
    csv_writer = csv.writer(fh)
    csv_writer.writerow(['train loss', 'val loss'])
    fh.close()

    for epoch in range(args.n_epochs):
        # set average meters to print
        loss_meters = AverageMeter()
        val_meters = AverageMeter()

        with (tqdm(total=(tr_length - tr_length % args.batch_size)) as _tqdm):
            _tqdm.set_description('epoch: {}/{}'.format(epoch + 1, args.n_epochs))
            for i, (y, X) in enumerate(tr_loader):
                bs = X.shape[0]
                X, y = X.to(args.device), y.to(args.device)
                Xk = net(y)

                loss = criteria(Xk, X)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_meters.update(loss.item(), bs)

                torch.cuda.empty_cache()
                dict = {f'tr_mse': f'{loss_meters.avg:.6f}'}
                dict.update({'val_mse': f'{val_meters.avg:.6f}'})
                _tqdm.set_postfix(dict)
                _tqdm.update(bs)

            """ SAVE STATES """
            if (epoch + 1) % 10 == 0:
                state = {
                    'epoch': epoch,
                    'state_dict': net.state_dict()}
                torch.save(state, os.path.join(args.save_path, f'epoch_{epoch}.state'))
            if (epoch + 1) % 50 == 0:
                plot_reconstruction(X, y, Xk, args, epoch, algo="UIE fixpoint")

            """ VALIDATION """
            with torch.no_grad():
                for y, X, _ in val_loader:
                    bs = X.shape[0]
                    X, y = X.to(args.device), y.to(args.device)
                    Xk = net(y)
                    val_loss = criteria(Xk, X)
                    val_meters.update(val_loss.item(), bs)

                    torch.cuda.empty_cache()
                    dict = {f'tr_mse': f'{loss_meters.avg:.6f}'}
                    dict.update({'val_mse': f'{val_meters.avg:.6f}'})
                    _tqdm.set_postfix(dict)
                    _tqdm.update(bs)

            fh = open(trajectory_path, 'a', newline='')  # a for append
            csv_writer = csv.writer(fh)
            csv_writer.writerow([loss_meters.avg, val_meters.avg])
            fh.close()

        """ READ TRAJECTORY """
        traj = read_csv(trajectory_path)
        tr_list = traj["train loss"].tolist()
        val_list = traj["val loss"].tolist()

        plt.figure()
        plt.semilogy(np.arange(len(tr_list)), tr_list)
        plt.semilogy(np.arange(len(val_list)), val_list)
        plt.legend(['train', 'val'])
        plt.xlabel('epochs')
        plt.ylabel('MSE')
        plt.tight_layout()
        plt.savefig(args.save_path + "/trajectory.png")
        plt.close()

else:
    """ Evaluation ... """
    criteria_title = ['MSE', 'PSNR', 'SSIM']
    len_meter = len(criteria_title)
    loss_meters = [AverageMeter() for _ in range(len_meter)]

    trajectory_path = args.save_path + '/trajectory_test.csv'
    fh = open(trajectory_path, 'a')
    csv_writer = csv.writer(fh)
    csv_writer.writerow(criteria_title)
    fh.close()
    net.eval()

    with tqdm(total=(ts_length - ts_length % args.batch_size_val)) as _tqdm:
        _tqdm.set_description('epoch: {}/{}'.format(1, 1))
        for i, (y, X, _) in enumerate(ts_loader):
            bs = X.shape[0]
            X, y = X.to(args.device), y.to(args.device)
            if args.test_mode == 'AT':
                delta, loss_list = tr.PGD(net, X, torch.clone(y), args.epsilon, args.PGD_stepsize, args.n_PGD_steps,
                                          dim=3, eps_mode='l2', nouts=1)
                y += delta
            elif args.test_mode == 'noisy':
                y += torch.randn_like(y) * args.noise_level

            with torch.no_grad():
                Xk = net(y)
            Xk = torch.clamp(Xk, 0, 1)
            ts_loss = criteria(Xk, X)
            _, recon_psnr, _, ssim = compute_metrics3chan(Xk, X, y)
            criteria_list = [ts_loss, recon_psnr, ssim]

            for k in range(len_meter):
                loss_meters[k].update(criteria_list[k].item(), bs)

            torch.cuda.empty_cache()
            dict = {f'{criteria_title[k]}': f'{loss_meters[k].avg:.6f}' for k in range(len_meter)}
            _tqdm.set_postfix(dict)
            _tqdm.update(bs)

        fh = open(trajectory_path, 'a', newline='')
        csv_writer = csv.writer(fh)
        csv_writer.writerow([loss_meters[k].avg for k in range(len_meter)])
        fh.close()

    plot_reconstruction(X, y, Xk, args, -1, algo="PhyNN fixpoint")
    print(f'{loss_meters[1].avg:.2f} / {loss_meters[2].avg:.3f}')
