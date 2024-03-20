# encoding=utf-8
# 加载测试数据集，输入模型，对比结果，计算f1-score
import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, precision_recall_fscore_support
from torchvision import transforms
from tqdm import tqdm

from emotion_det import EmotionInference
from net import FACE_SHAPE


data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, ], [0.24, ])

])


def get_f1_score(output, y):
    """
    计算f1
    :param output:
    :param y:
    :return:
    """
    return f1_score(y, output, average='macro')


def test_with_onnx(model_path, img_path):
    """
    测试onnx模型的f1
    :param model_path: onnx文件路径
    :param img_path: 数据集路径
    :return: f1
    """

    labels = []
    outputs = []
    emotionInference = EmotionInference(model_path=model_path)
    for dir_name in tqdm(os.listdir(img_path)):
        for filename in os.listdir(f"{img_path}/{dir_name}"):
            labels.append(class_map[dir_name])
            image = cv2.imread(f"{img_path}/{dir_name}/{filename}", cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, FACE_SHAPE, interpolation=cv2.INTER_LINEAR)
            output = emotionInference.pipeline(img=image, to_string=False)
            outputs.append(output)

    # return get_f1_score(output=outputs, y=labels)

    plot_matrix(y_pred=outputs, y_true=labels)
    return precision_recall_fscore_support(y_pred=outputs, y_true=labels)


def plot_matrix(y_pred, y_true):
    """
    画出混淆矩阵
    :param y_pred:
    :param y_true:
    :return:
    """
    # 计算混淆矩阵
    conf_mat = confusion_matrix(y_true, y_pred)

    # 打印混淆矩阵
    print("Confusion Matrix:")
    print(conf_mat)
    accuracy = np.trace(conf_mat) / np.sum(conf_mat)

    print("Accuracy:", accuracy)
    # 使用 seaborn 绘制混淆矩阵的热图
    labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]  # 类别标签，根据你的具体情况进行调整
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.savefig("Confusion Matrix.jpg")


if __name__ == "__main__":
    class_map = {
        "angry": 0,
        "disgust": 1,
        "fear": 2,
        "happy": 3,
        "neutral": 4,
        "sad": 5,
        "surprise": 6,
    }
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="./models/emotion_net.onnx")
    parser.add_argument('--img_path', type=str, default="./data/FER-2013/test", help='test-set path')
    args = parser.parse_args()
    pre, rec, f1, num_sample = test_with_onnx(model_path=args.model_path, img_path=args.img_path)
    emotions = list(class_map.keys())
    print("expression    pre              rec              f1-score          num_sample")
    for i in range(7):
        print(f"{emotions[i]:<10}  {pre[i]:<15.4f}   {rec[i]:<15.4f}    {f1[i]:<15.4f}    {num_sample[i]:<10}")
