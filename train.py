# -*- coding: UTF-8 -*-
"""
training program
"""
import paddle.fluid as fluid
import numpy as np
import paddle
import reader
import os
import utils
import config
from maseresnext import MaSeResNeXt


def build_optimizer(parameter_list=None):
    """
    build optimizer
    :return:
    """
    epoch = config.train_parameters["num_epochs"]
    batch_size = config.train_parameters["train_batch_size"]
    iters = config.train_parameters["train_image_count"] // batch_size
    learning_strategy = config.train_parameters['sgd_strategy']
    lr = learning_strategy['learning_rate']

    boundaries = [int(epoch * i * iters) for i in learning_strategy["lr_epochs"]]
    values = [i * lr for i in learning_strategy["lr_decay"]]
    optimizer = fluid.optimizer.SGDOptimizer(learning_rate=fluid.layers.piecewise_decay(boundaries, values),
                                             regularization=fluid.regularizer.L2Decay(0.00005),
                                             parameter_list=parameter_list)
    utils.logger.info("use optimizer")
    return optimizer


def load_params(model, optimizer):
    """
    loading model parameters
    :param model:
    :return:
    """
    if config.train_parameters["continue_train"] and os.path.exists(config.train_parameters['save_model_dir']+'.pdparams'):
        utils.logger.info("load params from {}".format(config.train_parameters['save_model_dir']))
        # params, _ = fluid.dygraph.load_persistables(config.train_parameters['save_model_dir'])
        para_dict, opti_dict = fluid.dygraph.load_dygraph(config.train_parameters['save_model_dir'])
        model.set_dict(para_dict)
        optimizer.set_dict(opti_dict)
        # model.load_dict(params)


def train():
    """
    train loop
    :return:
    """
    # The hardware is automatically selected based on whether the current paddle is CPU or GPU version
    # If it's a GPU, block 0 is used by default
    # If you want to specify the use, you need to either pass in the place variable actively 
    #   or control the visible video card by setting the CUDA_VISIBLE_DEVICES environment variable
    utils.logger.info("start train")
    with fluid.dygraph.guard():
        epoch_num = config.train_parameters["num_epochs"]
        net = MaSeResNeXt(config.train_parameters['class_dim'])
        optimizer = build_optimizer(parameter_list=net.parameters())
        load_params(net, optimizer)
        file_list = os.path.join(config.train_parameters['data_dir'], config.train_parameters['train_file_list'])
        custom_reader = reader.custom_image_reader(file_list, config.train_parameters['data_dir'], mode='train')
        train_reader = paddle.batch(custom_reader,
                                    batch_size=config.train_parameters['train_batch_size'],
                                    drop_last=True)
        current_acc = 0.0
        for current_epoch in range(epoch_num):
            epoch_acc = 0.0
            batch_count = 0
            for batch_id, data in enumerate(train_reader()):
                dy_x_data = np.array([x[0] for x in data]).astype('float32')
                y_data = np.array([[x[1]] for x in data]).astype('int')

                img = fluid.dygraph.to_variable(dy_x_data)
                label = fluid.dygraph.to_variable(y_data)
                label.stop_gradient = True

                out, acc = net(img, label)
                softmax_out = fluid.layers.softmax(out, use_cudnn=False)
                loss = fluid.layers.cross_entropy(softmax_out, label)
                avg_loss = fluid.layers.mean(loss)

                avg_loss.backward()
                optimizer.minimize(avg_loss)
                net.clear_gradients()
                batch_count += 1
                epoch_acc += acc.numpy()
                if batch_id % 5 == 0 and batch_id != 0:
                    utils.logger.info("loss at epoch {} step {}: {}, acc: {}"
                                      .format(current_epoch, batch_id, avg_loss.numpy(), acc.numpy()))

            epoch_acc /= batch_count
            utils.logger.info("epoch {} acc: {}".format(current_epoch, epoch_acc))
            if epoch_acc >= current_acc:
                utils.logger.info("current epoch {} acc: {} better than last acc: {}, save model"
                                  .format(current_epoch, epoch_acc, current_acc))
                current_acc = epoch_acc
                
                fluid.dygraph.save_dygraph(net.state_dict(), config.train_parameters['save_model_dir'])
                fluid.dygraph.save_dygraph(optimizer.state_dict(), config.train_parameters['save_model_dir'])
        utils.logger.info("train till end")


if __name__ == "__main__":
    train()
