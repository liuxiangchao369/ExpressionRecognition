# encoding=utf-8
import argparse
import os
from PIL import Image

import numpy as np
import onnxruntime as ort
from torchvision import transforms
import torch
from net import FACE_SHAPE

from torch import nn


class EmotionInference:
    def __init__(self, model_path, channels=3, is_tensor=False, device=None):
        """
        表情分类模型的调度程序
        :param model_path: onnx文件地址
        :param device: 运行设备GPU or CPU
        """
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model_path = model_path
        trans_list = []
        self.channels = channels
        if channels == 3:
            trans_list.append(transforms.Grayscale(num_output_channels=1))
        trans_list.append(transforms.Resize(FACE_SHAPE))
        if not is_tensor:
            trans_list.append(transforms.ToTensor())

        trans_list.append(transforms.Normalize([0.5, ], [0.24, ]))
        self.data_transforms = transforms.Compose(trans_list)
        self.softmax = nn.Softmax(dim=1)
        self.class_map = {
            0: "angry",
            1: "disgust",
            2: "fear",
            3: "happy",
            4: "neutral",
            5: "sad",
            6: "surprise",
        }
        self.session = ort.InferenceSession(self.model_path,
                                            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    def pre_process(self, img: np.ndarray):
        """
        前处理
        将原始图像加工为模型输入
        :param img: ndarray c,w,h
        :return: input_tensor
        """
        # transfer to 1 channel if img has 3 channels
        img = self.data_transforms(img)
        b, w, h = img.shape
        image_data = img.reshape(b, 1, w, h)
        # image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
        return image_data

    def inference(self, input_tensor):
        """
        onnx 模型推理
        :param input_tensor: 模型输入
        :return: 模型输出
        """
        # print(input_tensor.shape)
        model_inputs = self.session.get_inputs()
        outputs = self.session.run(None, {model_inputs[0].name: input_tensor.cpu().numpy()})
        return outputs

    def post_process(self, output_tensor, to_string=True):
        """
        后处理
        将模型输出进一步加工
        :param to_string: 输出string类型的标签还是数字类的表签
        :param output_tensor: model outputs
        :return:result 0 for female or 1 for male
        """
        output = self.softmax(torch.tensor(output_tensor[0]))
        output = torch.argmax(output, dim=1)
        if to_string:
            res = self.class_map[int(output.cpu())]
        else:
            res = int(output.cpu())
        return res

    def pipeline(self, img, to_string=True):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        img_data = self.pre_process(img)
        outputs = self.inference(img_data)

        return self.post_process(outputs, to_string=to_string)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, default="test.png", help='image path.')
    args = parser.parse_args()
    img_path = args.img_path
    # img_path = r"D:\DMS-project\emotion_classification\data\FER-2013\test\happy\PrivateTest_95094.jpg"
    img_ = Image.open(img_path)
    emotionInference = EmotionInference(model_path="./models/emotion_net.onnx")
    emotion = emotionInference.pipeline(img_)
    print(emotion)
