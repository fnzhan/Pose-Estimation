import os
import os.path as osp
import math
import time
import glob
import abc
from torch.utils.data import DataLoader
import torch.optim
import torchvision.transforms as transforms

from config import cfg
from dataset import DatasetLoader
from timer import Timer
from logger import colorlogger
from torch.nn.parallel.data_parallel import DataParallel
from model import get_full_net

# dynamic dataset import
for i in range(len(cfg.trainset)):
    exec('from ' + cfg.trainset[i] + ' import ' + cfg.trainset[i])
exec('from ' + cfg.testset + ' import ' + cfg.testset)


class Base(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, log_name='logs.txt'):
        self.cur_epoch = 0

        # timer
        self.tot_timer = Timer()
        self.gpu_timer = Timer()
        self.read_timer = Timer()

        # logger
        self.logger = colorlogger(cfg.log_dir, log_name=log_name)

    @abc.abstractmethod
    def _make_batch_generator(self):
        return

    @abc.abstractmethod
    def _make_model(self):
        return

    def save_model(self, state, epoch):
        file_path = osp.join(cfg.model_dir, 'snapshot_{}.pth.tar'.format(str(epoch)))
        torch.save(state, file_path)
        self.logger.info("Write snapshot into {}".format(file_path))

    def load_model(self, model, optimizer):
        model_file_list = glob.glob(osp.join(cfg.model_dir, '*.pth.tar'))
        cur_epoch = max([int(file_name[file_name.find('snapshot_') + 9: file_name.find('.pth.tar')]) for file_name in
                         model_file_list])
        ckpt = torch.load(osp.join(cfg.model_dir, 'snapshot_' + str(cur_epoch) + '.pth.tar'))
        start_epoch = ckpt['epoch'] + 1
        model.load_state_dict(ckpt['network'])
        optimizer.load_state_dict(ckpt['optimizer'])

        return start_epoch, model, optimizer


class Trainer(Base):

    def __init__(self):
        super(Trainer, self).__init__(log_name='train_logs.txt')

    def get_optimizer(self, model):

        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        return optimizer

    def set_lr(self, epoch):
        for e in cfg.lr_dec_epoch:
            if epoch < e:
                break
        if epoch < cfg.lr_dec_epoch[-1]:
            idx = cfg.lr_dec_epoch.index(e)
            for g in self.optimizer.param_groups:
                g['lr'] = cfg.lr / (cfg.lr_dec_factor ** idx)
        else:
            for g in self.optimizer.param_groups:
                g['lr'] = cfg.lr / (cfg.lr_dec_factor ** len(cfg.lr_dec_epoch))

    def get_lr(self):
        for g in self.optimizer.param_groups:
            cur_lr = g['lr']

        return cur_lr

    def _make_batch_generator(self):
        # data load and construct batch generator
        self.logger.info("Creating dataset...")
        trainset_loader = []
        batch_generator = []
        iterator = []
        for i in range(len(cfg.trainset)):
            if i > 0:
                ref_joints_name = trainset_loader[0].joints_name
            else:
                ref_joints_name = None
            trainset_loader.append(
                DatasetLoader(eval(cfg.trainset[i])("train"), ref_joints_name, True, transforms.Compose([ \
                    transforms.ToTensor(),
                    transforms.Normalize(mean=cfg.pixel_mean, std=cfg.pixel_std)] \
                    )))
            if i == 0:
                batch_generator.append(
                    DataLoader(dataset=trainset_loader[-1],
                               batch_size=32,
                               shuffle=True, num_workers=8, pin_memory=True))
            elif i == 1:
                batch_generator.append(
                    DataLoader(dataset=trainset_loader[-1],
                               batch_size=32,
                               shuffle=True, num_workers=8, pin_memory=True))
            iterator.append(iter(batch_generator[-1]))

        self.joint_num = trainset_loader[0].joint_num
        self.itr_per_epoch = math.ceil(
            trainset_loader[0].__len__() / 32)
        self.batch_generator = batch_generator
        self.iterator = iterator

    def _make_model(self):
        # prepare network
        self.logger.info("Creating graph and optimizer...")

        model = get_full_net(cfg, self.joint_num)
        model = DataParallel(model).cuda()
        optimizer = self.get_optimizer(model)
        if cfg.continue_train:
            start_epoch, model, optimizer = self.load_model(model, optimizer)
        else:
            start_epoch = 0
        model.train()

        self.start_epoch = start_epoch
        self.model = model
        self.optimizer = optimizer


class Tester(Base):

    def __init__(self, test_epoch):
        self.test_epoch = int(test_epoch)
        super(Tester, self).__init__(log_name='test_logs.txt')

    def _make_batch_generator(self):
        # data load and construct batch generator
        self.logger.info("Creating dataset...")
        testset = eval(cfg.testset)("test")
        testset_loader = DatasetLoader(testset, None, False, transforms.Compose([ \
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.pixel_mean, std=cfg.pixel_std)] \
            ))
        batch_generator = DataLoader(dataset=testset_loader, batch_size=cfg.num_gpus * cfg.test_batch_size,
                                     shuffle=False, num_workers=cfg.num_thread, pin_memory=True)

        self.testset = testset
        self.joint_num = testset_loader.joint_num
        self.skeleton = testset_loader.skeleton
        self.flip_pairs = testset.flip_pairs
        self.batch_generator = batch_generator

    def _make_singleimg_init(self):
        self.joint_num = 18  # original:17, but manually added 'Thorax'
        self.joints_name = (
            'Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck', 'Nose', 'Head',
            'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'Thorax')
        self.flip_pairs = ((1, 4), (2, 5), (3, 6), (14, 11), (15, 12), (16, 13))
        self.skeleton = (
            (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16), (0, 1), (1, 2),
            (2, 3), (0, 4), (4, 5), (5, 6))
        self.joints_have_depth = True

    def _make_model(self):
        model_path = os.path.join(cfg.model_dir, 'snapshot_%d.pth.tar' % self.test_epoch)
        assert os.path.exists(model_path), 'Cannot find model at ' + model_path
        self.logger.info('Load checkpoint from {}'.format(model_path))

        # prepare network
        self.logger.info("Creating graph...")

        model = get_full_net(cfg, self.joint_num)
        model = DataParallel(model).cuda()
        ckpt = torch.load(model_path)
        model.load_state_dict(ckpt['network'])
        model.eval()

        self.model = model

    def _evaluate(self, preds, result_save_path):
        self.testset.evaluate(preds, result_save_path)
