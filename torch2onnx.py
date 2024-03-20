import torch

from net import EmotionNet
from net import FACE_SHAPE
from onnxsim import simplify
import onnx


def to_onnx(model_path, dummy_input, output_path, device, num_classes=7):
    """
    pytorch参数文件+网络结构转为onnx模型
    固定输入尺寸
    :param num_classes: 类别数量
    :param model_path: pth模型路径
    :param dummy_input:输入样例或尺寸
    :param output_path:onnx输出路径
    :param device: CPU or cuda
    :return:
    """
    model = EmotionNet(num_classes=num_classes)
    model = model.to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    input_names = ['Input']
    output_names = ['Output']
    torch.onnx.export(model, dummy_input, output_path,
                      input_names=input_names,
                      output_names=output_names,
                      # operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
                      export_params=True,
                      )


def simplifier(onnx_model_path, output_path):
    onnx_model = onnx.load(onnx_model_path)
    model_simp, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, output_path)
    print('finished exporting onnx')


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device="cpu"
    model_path = "./models/best_network.pth"
    dummy_input = torch.randn(1, 1, FACE_SHAPE[0], FACE_SHAPE[1], device=device)
    output_path = 'models/emotion_net.onnx'
    to_onnx(model_path, dummy_input, output_path, device,num_classes=7)

    # simplifier_output_path = "./models/emotion_simplified.onnx"
    # simplifier(onnx_model_path=output_path, output_path=simplifier_output_path)
