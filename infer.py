import paddle.fluid as fluid
from maseresnext import MaSeResNeXt
import numpy as np
from config import train_parameters, init_train_parameters
from PIL import Image
import config

def resize_img(img, target_size):
    """
    Force scale image
    :param img:
    :param target_size:
    :return:
    """
    img = img.resize((target_size[1], target_size[2]), Image.BILINEAR)
    return img

def read_img(img_path):
    img = Image.open(img_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = resize_img(img, train_parameters['input_size'])
    # HWC--->CHW && normalized
    img = np.array(img).astype('float32')
    img -= train_parameters['mean_rgb']
    img = img.transpose((2, 0, 1))  # HWC to CHW
    img *= 0.007843  # Normalized pixel values
    img = np.array([img]).astype('float32')
    return img
    
def infer():
    with fluid.dygraph.guard():
        params, _ = fluid.load_dygraph(config.train_parameters['save_model_dir'])
        
        net = Ma_ConvCardSeResNeXt(config.train_parameters['class_dim'])

        net.set_dict(params)
        net.eval()
        print("checkpoint loaded")
        
        label_dic = train_parameters["label_dict"]
        label_dic = {v: k for k, v in label_dic.items()}
        
        img_path = train_parameters['infer_img']
        img = read_img(img_path)
        
        results = net(fluid.dygraph.to_variable(img))
        lab = np.argsort(results.numpy())
        print("image {} Infer result is: {}".format(img_path, label_dic[lab[0][-1]]))
        

if __name__ == "__main__":
    init_train_parameters()
    infer()