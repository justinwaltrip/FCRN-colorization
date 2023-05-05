from datetime import datetime
import shutil
import socket
import time
import torch
from tensorboardX import SummaryWriter
from torch.optim import lr_scheduler

from dataloaders import kitti_dataloader, nyu_dataloader
from dataloaders.path import Path
from metrics import AverageMeter, Result
import utils
import criteria
import os
import torch.nn as nn
import wandb


from network import FCRN

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # use single GPU

args = utils.parse_command()
print(args)

best_loss = float('Inf')


def create_loader(args):
    traindir = os.path.join(Path.db_root_dir(args.dataset), 'train')
    if os.path.exists(traindir):
        print('Train dataset "{}" is existed!'.format(traindir))
    else:
        print('Train dataset "{}" is not existed!'.format(traindir))
        exit(-1)

    valdir = os.path.join(Path.db_root_dir(args.dataset), 'val')
    if os.path.exists(traindir):
        print('Train dataset "{}" is existed!'.format(valdir))
    else:
        print('Train dataset "{}" is not existed!'.format(valdir))
        exit(-1)

    if args.dataset == 'kitti':
        train_set = kitti_dataloader.KITTIDataset(traindir, type='train')
        val_set = kitti_dataloader.KITTIDataset(valdir, type='val')

        # sample 3200 pictures for validation from val set
        weights = [1 for i in range(len(val_set))]
        print('weights:', len(weights))
        sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=3200)
    elif args.dataset == 'nyu':
        # if overfit, train and val set are the same
        if args.overfit:
            train_set = nyu_dataloader.NYUDataset(
                traindir, type="train", small_subset=True
            )
            val_set = train_set
        else:
            train_set = nyu_dataloader.NYUDataset(traindir, type="train")
            val_set = nyu_dataloader.NYUDataset(valdir, type="val")
    else:
        print('no dataset named as ', args.dataset)
        exit(-1)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)

    if args.dataset == 'kitti':
        val_loader = torch.utils.data.DataLoader(
            val_set, batch_size=args.batch_size, sampler=sampler, num_workers=args.workers, pin_memory=True)
    else:
        val_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
        )

    return train_loader, val_loader


def main():
    global args, best_loss, output_directory

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # set random seed
    torch.manual_seed(args.manual_seed)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        args.batch_size = args.batch_size * torch.cuda.device_count()
    else:
        print(f"Using device {device}")
    # start new wandb run
    wandb.init(project="fcrn-colorization", notes="pretraining")

    train_loader, val_loader = create_loader(args)

    if args.resume:
        assert os.path.isfile(args.resume), \
            "=> no checkpoint found at '{}'".format(args.resume)
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)

        start_epoch = checkpoint['epoch'] + 1
        best_result = checkpoint['best_result']
        optimizer = checkpoint['optimizer']

        # model_dict = checkpoint['model'].module.state_dict()  # to load the trained model using multi-GPUs
        # model = FCRN.ResNet(output_size=train_loader.dataset.output_size, pretrained=False)
        # model.load_state_dict(model_dict)

        # solve 'out of memory'
        model = checkpoint['model']

        print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))

        # clear memory
        del checkpoint
        # del model_dict
        torch.cuda.empty_cache()
    else:
        print("=> creating Model")
        model = FCRN.ResNet(output_size=train_loader.dataset.output_size)
        print("=> model created.")
        start_epoch = 0

        # different modules have different learning rate
        train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr},
                        {'params': model.get_10x_lr_params(), 'lr': args.lr * 10}]

        optimizer = torch.optim.Adam(
            train_params, lr=args.lr, weight_decay=args.weight_decay
        )

        # You can use DataParallel() whether you use Multi-GPUs or not
        model = nn.DataParallel(model).cuda()

    # when training, use reduceLROnPlateau to reduce learning rate
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=args.lr_patience)

    # loss function
    criterion = nn.MSELoss()

    # create directory path
    output_directory = utils.get_output_directory(args)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    best_txt = os.path.join(output_directory, 'best.txt')
    config_txt = os.path.join(output_directory, 'config.txt')

    # write training parameters to config file
    if not os.path.exists(config_txt):
        with open(config_txt, 'w') as txtfile:
            args_ = vars(args)
            args_str = ''
            for k, v in args_.items():
                args_str = args_str + str(k) + ':' + str(v) + ',\t\n'
            txtfile.write(args_str)

    # create log
    log_path = os.path.join(output_directory, 'logs',
                            datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    if os.path.isdir(log_path):
        shutil.rmtree(log_path)
    os.makedirs(log_path)
    logger = SummaryWriter(log_path)

    for epoch in range(start_epoch, args.epochs):

        # remember change of the learning rate
        for i, param_group in enumerate(optimizer.param_groups):
            old_lr = float(param_group['lr'])
            logger.add_scalar('Lr/lr_' + str(i), old_lr, epoch)

        # train for one epoch
        pretrain(train_loader, model, criterion, optimizer, epoch, logger, device)

        # evaluate on validation set
        loss, img_merge = prevalidate(val_loader, model, criterion, epoch, logger, device)
        # remember best rmse and save checkpoint
        is_best = loss < best_loss
        if is_best:
            best_loss = loss
            with open(best_txt, 'w') as txtfile:
                txtfile.write(
                    "epoch={}, loss={:.4f}".format(epoch, best_loss))
            if img_merge is not None:
                img_filename = output_directory + '/comparison_best.png'
                utils.save_image(img_merge, img_filename)

        # save checkpoint for each epoch
        utils.save_checkpoint({
            'args': args,
            'epoch': epoch,
            'model': model,
            # 'best_result': best_result,
            'optimizer': optimizer,
        }, is_best, epoch, output_directory)

        # when rml doesn't fall, reduce learning rate
        scheduler.step(loss)

    logger.close()


# train
def pretrain(train_loader, model, criterion, optimizer, epoch, logger, device):
    # average_meter = AverageMeter()
    model.train()  # switch to train mode
    # end = time.time()
    # batch_num = len(train_loader)
    total_loss = 0.0
    for _, (input, target) in enumerate(train_loader):
        input, target = input.to(device), target.to(device)
        # print('input size  = ', input.size())
        # print('target size = ', target.size())
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        # data_time = time.time() - end

        # compute pred
        # end = time.time()

        pred = model(input)  # @wx 注意输出

        # print('pred size = ', pred.size())
        # print('target size = ', target.size())

        loss = criterion(pred, target)
        optimizer.zero_grad()
        loss.backward()  # compute gradient and do SGD step
        optimizer.step()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        # gpu_time = time.time() - end

        # measure accuracy and record loss
        # result = Result()
        # result.evaluate(pred.data, target.data)
        # average_meter.update(result, gpu_time, data_time, input.size(0))
        # end = time.time()
        

        # if (i + 1) % args.print_freq == 0:
        #     print('=> output: {}'.format(output_directory))
        #     print('Train Epoch: {0} [{1}/{2}]\t'
        #           't_Data={data_time:.3f}({average.data_time:.3f}) '
        #           't_GPU={gpu_time:.3f}({average.gpu_time:.3f})\n\t'
        #           'Loss={Loss:.5f} '
        #           'RMSE={result.rmse:.2f}({average.rmse:.2f}) '
        #           'RML={result.absrel:.2f}({average.absrel:.2f}) '
        #           'Log10={result.lg10:.3f}({average.lg10:.3f}) '
        #           'Delta1={result.delta1:.3f}({average.delta1:.3f}) '
        #           'Delta2={result.delta2:.3f}({average.delta2:.3f}) '
        #           'Delta3={result.delta3:.3f}({average.delta3:.3f})'.format(
        #         epoch, i + 1, len(train_loader), data_time=data_time,
        #         gpu_time=gpu_time, Loss=loss.item(), result=result, average=average_meter.average()))
            # current_step = epoch * batch_num + i
            # logger.add_scalar('Train/RMSE', result.rmse, current_step)
            # logger.add_scalar('Train/rml', result.absrel, current_step)
            # logger.add_scalar('Train/Log10', result.lg10, current_step)
            # logger.add_scalar('Train/Delta1', result.delta1, current_step)
            # logger.add_scalar('Train/Delta2', result.delta2, current_step)
            # logger.add_scalar('Train/Delta3', result.delta3, current_step)

        total_loss += loss.item()

    # avg = average_meter.average()

    wandb.log(
        {
            "train/loss": total_loss / len(train_loader),
        },
        step=epoch,
    )

# validation
def prevalidate(val_loader, model, criterion, epoch, logger, device):
    # average_meter = AverageMeter()

    model.eval()  # switch to evaluate mode

    # end = time.time()

    total_loss = 0.0

    for i, (input, target) in enumerate(val_loader):
        input, target = input.to(device), target.to(device)
        if device == "cuda":
            torch.cuda.synchronize()
        # data_time = time.time() - end

        # compute output
        # end = time.time()
        with torch.no_grad():
            pred = model(input)
            loss = criterion(pred, target)

        if device == "cuda":
            torch.cuda.synchronize()
        # gpu_time = time.time() - end

        # measure accuracy and record loss
        # result = Result()
        # result.evaluate(pred.data, target.data)

        # average_meter.update(result, gpu_time, data_time, input.size(0))
        # end = time.time()

        if i == 0:
            # save 8 images for visualization
            for j in range(8):
                rgb = input[j]
                _pred = pred[j]
                _target = target[j]
                if j == 0:
                    img_merge = utils.merge_into_row(rgb, _target, _pred)
                else:
                    row = utils.merge_into_row(rgb, _target, _pred)
                    img_merge = utils.add_row(img_merge, row)
            filename = output_directory + "/comparison_" + str(epoch) + ".png"
            utils.save_image(img_merge, filename)

        # if (i + 1) % args.print_freq == 0:
        #     print('Test: [{0}/{1}]\t'
        #           't_GPU={gpu_time:.3f}({average.gpu_time:.3f})\n\t'
        #           'RMSE={result.rmse:.2f}({average.rmse:.2f}) '
        #           'RML={result.absrel:.2f}({average.absrel:.2f}) '
        #           'Log10={result.lg10:.3f}({average.lg10:.3f}) '
        #           'Delta1={result.delta1:.3f}({average.delta1:.3f}) '
        #           'Delta2={result.delta2:.3f}({average.delta2:.3f}) '
        #           'Delta3={result.delta3:.3f}({average.delta3:.3f})'.format(
        #         i + 1, len(val_loader), gpu_time=gpu_time, result=result, average=average_meter.average()))

        total_loss += loss.item()

    # avg = average_meter.average()


    # print('\n*\n'
    #       'RMSE={average.rmse:.3f}\n'
    #       'Rel={average.absrel:.3f}\n'
    #       'Log10={average.lg10:.3f}\n'
    #       'Delta1={average.delta1:.3f}\n'
    #       'Delta2={average.delta2:.3f}\n'
    #       'Delta3={average.delta3:.3f}\n'
    #       't_GPU={time:.3f}\n'.format(
    #     average=avg, time=avg.gpu_time))

    # logger.add_scalar('Test/rmse', avg.rmse, epoch)
    # logger.add_scalar('Test/Rel', avg.absrel, epoch)
    # logger.add_scalar('Test/log10', avg.lg10, epoch)
    # logger.add_scalar('Test/Delta1', avg.delta1, epoch)
    # logger.add_scalar('Test/Delta2', avg.delta2, epoch)
    # logger.add_scalar('Test/Delta3', avg.delta3, epoch)

    wandb.log(
        {
            "val/loss": total_loss / len(val_loader),
        },
        step=epoch,
    )

    return total_loss / len(val_loader), img_merge


if __name__ == '__main__':
    main()
