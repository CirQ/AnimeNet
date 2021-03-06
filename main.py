from __future__ import division

import argparse
import logging
import os
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch import FloatTensor
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
import torchvision.utils as vutil

from data import TagImageDataset
from model import Generator, Discriminator

parser = argparse.ArgumentParser(description='PyTorch SRResNet')

parser.add_argument('--cuda', action='store_true', help='use cuda')
parser.add_argument('--gpus', default='0', type=str, help='gpu ids (default: 0)')

parser.add_argument('--tag', type=str, required=True, help='the path to tags')
parser.add_argument('--image', type=str, required=True, help='the path to images')
parser.add_argument('--threads', type=int, default=8, help='number of threads for data loader to use')
parser.add_argument('--batch', type=int, default=64, help='training batch size')
parser.add_argument('--image_size', type=int, default=128, help='the height and width of images')
parser.add_argument('--noise_size', type=int, default=128, help='the length of noise vector')

parser.add_argument('--features', type=int, default=30, help='the number of features')

parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam')
parser.add_argument('--step', type=int, default=10, help='decay the learning rate to the initial LR every n epoches')

parser.add_argument('--resume', default='', type=str, help='path to checkpoint (default: none)')
parser.add_argument('--pre_trained', default='', type=str, help='path to pretrained model (default: none)')

parser.add_argument('--epoch', type=int, default=100, help='number of epoches to train for')
parser.add_argument('--start_epoch', default=1, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--check_step', default=10, type=int, help='save checkpoint after so many epoch')


opt = parser.parse_args()


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

logfile = logging.FileHandler('train.log')
console = logging.StreamHandler()

logger.addHandler(logfile)
logger.addHandler(console)


def main():
    if opt.cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus
        if not torch.cuda.is_available():
            raise Exception('No GPU found or Wrong gpu id, please run without --cuda')

    logger.info('[INFO] Loading datasets')
    train_set = TagImageDataset(tag_path=opt.tag, img_path=opt.image)
    train_loader = DataLoader(train_set, num_workers=opt.threads, batch_size=opt.batch, shuffle=True, drop_last=True)

    logger.info('[INFO] Building model')
    G = Generator(opt.features)
    D = Discriminator(opt.features)
    criterion = nn.BCEWithLogitsLoss()

    logger.info('[INFO] Setting Optimizer')
    G_optim = optim.Adam(G.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    D_optim = optim.Adam(D.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    logger.info('[INFO] Setting GPU')
    if opt.cuda:
        G = G.cuda()
        D = D.cuda()
        criterion = criterion.cuda()

    if opt.resume:
        if os.path.isfile(opt.resume):
            logger.info('[LOAD] Loading checkpoint {}'.format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint['epoch'] + 1
            G.load_state_dict(checkpoint['g'])
            D.load_state_dict(checkpoint['d'])
            G_optim.load_state_dict(checkpoint['g_optim'])
            D_optim.load_state_dict(checkpoint['d_optim'])
        else:
            logger.warning('[ERROR] No checkpoint found at {}'.format(opt.resume))

    if opt.pre_trained:
        if os.path.isfile(opt.pre_trained):
            logger.info('[LOAD] Loading model {}'.format(opt.pre_trained))
            weights = torch.load(opt.pre_trained)
            G.load_state_dict(weights['g'].state_dict())
            D.load_state_dict(weights['d'].state_dict())
            G_optim.load_state_dict(weights['g_optim'].state_dict())
            D_optim.load_state_dict(weights['d_optim'].state_dict())
        else:
            logger.warning('[ERROR] No model found at {}'.format(opt.pre_trained))

    logger.info('[INFO] Start Training')
    train(train_loader, G, D, G_optim, D_optim, criterion)


def adjust_learning_rate(optimizer, epoch):
    lr = opt.lr * (0.1 ** (epoch // opt.step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def fake_features(tensor):
    # discrete
    # tensor.data.bernoulli_(0.16)

    # continuous
    # null = Variable(torch.zeros(1))
    # tensor.data.bernoulli_(0.16)
    # for vector in tensor:
    #     for ele in vector:
    #         if ele.data[0] > null.data[0]:
    #             ele.data.uniform_(0.1, 0.9)

    # concrete
    assert tensor.size() == torch.Size((opt.batch, 30))
    tensor.data.fill_(0)
    hair_color = random.randint(0, 12)
    hair_style = random.randint(13, 16)
    eyes_color = random.randint(17, 27)
    is_smile = random.choice((False, True))
    is_blush = random.choice((False, True))
    tensor.data[hair_color] = 1
    tensor.data[hair_style] = 1
    tensor.data[eyes_color] = 1
    if is_smile: tensor.data[-2] = 1
    if is_blush: tensor.data[-1] = 1

lambda_adv = opt.features
lambda_gp = 0.5
def train(train_loader, gen, dis, g_optim, d_optim, criterion):
    start_time = time.time()

    X = Variable(FloatTensor(opt.batch, 3, opt.image_size, opt.image_size))
    z = Variable(FloatTensor(opt.batch, opt.noise_size))
    tag_real = Variable(FloatTensor(opt.batch, opt.features))
    tag_fake = Variable(FloatTensor(opt.batch, opt.features))
    # y_real = Variable(torch.ones(opt.batch))
    # y_fake = Variable(torch.zeros(opt.batch))
    labels = Variable(FloatTensor(opt.batch))

    if opt.cuda:
        X, z = X.cuda(), z.cuda()
        tag_real, tag_fake = tag_real.cuda(), tag_fake.cuda()
        # y_real, y_fake = y_real.cuda(), y_fake.cuda()
        labels = labels.cuda()

    for epoch in range(opt.start_epoch, opt.epoch+1):
        adjust_learning_rate(g_optim, epoch)
        adjust_learning_rate(d_optim, epoch)

        for iteration, (tag, img) in enumerate(train_loader, start=1):
            X.data.copy_(img)
            tag_real.data.copy_(tag)

            ##########################
            # Training discriminator #
            ##########################
            dis.zero_grad()

            # trained with real image
            pred_real, pred_real_t = dis(X)
            labels.data.fill_(1.0)

            d_real_label_loss = criterion(pred_real, labels)
            d_real_tag_loss = criterion(pred_real_t, tag_real)
            d_real_loss = lambda_adv * d_real_label_loss + d_real_tag_loss
            d_real_loss.backward()

            # trained with fake image
            z.data.normal_(0, 1)
            fake_features(tag_fake)
            vec = torch.cat((z, tag_fake), 1)
            fake_X = gen(vec)
            pred_fake, pred_fake_t = dis(fake_X)
            labels.data.fill_(0.0)

            d_fake_label_loss = criterion(pred_fake, labels)
            d_fake_tag_loss = criterion(pred_fake_t, tag_fake)
            d_fake_loss = lambda_adv * d_fake_label_loss + d_fake_tag_loss
            d_fake_loss.backward()

            # gradient penalty
            shape = [opt.batch] + [1 for _ in range(X.dim()-1)]
            alpha = torch.rand(*shape)
            beta = torch.rand(X.size())
            if opt.cuda:
                alpha, beta = alpha.cuda(), beta.cuda()
            x_hat = Variable(alpha*X.data + (1-alpha)*(X.data+0.5*X.data.std()*beta), requires_grad=True)
            pred_hat, _ = dis(x_hat)
            grad_out = torch.ones(pred_hat.size())
            if opt.cuda:
                grad_out = grad_out.cuda()
            gradients = grad(outputs=pred_hat, inputs=x_hat, grad_outputs=grad_out,
                             create_graph=True, retain_graph=True, only_inputs=True)[0]
            gradient_penalty = lambda_gp * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
            gradient_penalty.backward()

            d_loss = d_real_loss + d_fake_loss + gradient_penalty
            d_optim.step()

            ######################
            # Training generator #
            ######################
            gen.zero_grad()

            z.data.normal_(0, 1)
            fake_features(tag_fake)
            vec = torch.cat((z, tag_fake), 1)
            gen_X = gen(vec)
            pred_gen, pred_gen_t = dis(gen_X)
            labels.data.fill_(1.0)

            g_label_loss = criterion(pred_gen, labels)
            g_tag_loss = criterion(pred_gen_t, tag_fake)
            g_loss = lambda_adv * g_label_loss + g_tag_loss
            g_loss.backward()

            g_optim.step()

            elapsed = time.time() - start_time
            logger.info('[%03d/%d] [%03d/%d]  elapsd: %-10.4f  loss_d: %10.4f  loss_g: %10.4f' % (epoch, opt.epoch, iteration, len(train_loader), elapsed, d_loss.data[0], g_loss.data[0]))

        save_stage(gen, dis, g_optim, d_optim, epoch)

def save_stage(gen, dis, gen_optim, dis_optim, epoch):
    if not os.path.exists('checkpoint'):
        os.makedirs('checkpoint')
    if not os.path.exists('samples'):
        os.makedirs('samples')

    checkpoint_out_path = os.path.join('checkpoint', 'checkpoint_epoch_{:03d}.pth'.format(epoch))
    state = {
        'epoch': epoch,
        'g': gen.state_dict(),
        'd': dis.state_dict(),
        'g_optim': gen_optim.state_dict(),
        'd_optim': dis_optim.state_dict(),
    }
    if epoch % opt.check_step == 0:
        torch.save(state, checkpoint_out_path)
        logger.info('[DUMP] checkpoint in epoch {} saved'.format(epoch))

    samples_out_path = os.path.join('samples', 'samples_epoch_{:03d}.png'.format(epoch))
    z = Variable(FloatTensor(opt.batch, opt.noise_size))
    tags = Variable(FloatTensor(opt.batch, opt.features))
    z.data.normal_(0, 1)
    fake_features(tags)
    if opt.cuda:
        z, tags = z.cuda(), tags.cuda()
    sample = gen(torch.cat((z, tags), 1))
    vutil.save_image(sample.data.view(opt.batch, 3, opt.image_size, opt.image_size), samples_out_path)
    logger.info('[DEMO] samples in epoch {} saved'.format(epoch))

if __name__ == '__main__':
    main()
