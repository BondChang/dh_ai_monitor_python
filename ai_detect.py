# coding:utf-8
import argparse
import cv2
import numpy as np
import onnxruntime as ort
import torch
import os
import json


class YOLOv8:
    """YOLOv8目标检测模型类，用于处理推理和可视化操作。"""

    def __init__(self, onnx_model, confidence_thres=0.3, iou_thres=0.5):
        """
        初始化YOLOv8类的实例。
        参数:
            onnx_model: ONNX模型的路径。
            confidence_thres: 过滤检测的置信度阈值。
            iou_thres: 非极大抑制的IoU（交并比）阈值。
        """
        self.onnx_model = onnx_model
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres

        # 从COCO数据集的配置文件加载类别名称
        self.classes = {
            0: 'person',
            1: 'bicycle',
            2: 'car',
            3: 'motorcycle',
            4: 'airplane',
            5: 'bus',
            6: 'train',
            7: 'truck',
            8: 'boat',
            9: 'traffic light',
            10: 'fire hydrant',
            11: 'stop sign',
            12: 'parking meter',
            13: 'bench',
            14: 'bird',
            15: 'cat',
            16: 'dog',
            17: 'horse',
            18: 'sheep',
            19: 'cow',
            20: 'elephant',
            21: 'bear',
            22: 'zebra',
            23: 'giraffe',
            24: 'backpack',
            25: 'umbrella',
            26: 'handbag',
            27: 'tie',
            28: 'suitcase',
            29: 'frisbee',
            30: 'skis',
            31: 'snowboard',
            32: 'sports ball',
            33: 'kite',
            34: 'baseball bat',
            35: 'baseball glove',
            36: 'skateboard',
            37: 'surfboard',
            38: 'tennis racket',
            39: 'bottle',
            40: 'wine glass',
            41: 'cup',
            42: 'fork',
            43: 'knife',
            44: 'spoon',
            45: 'bowl',
            46: 'banana',
            47: 'apple',
            48: 'sandwich',
            49: 'orange',
            50: 'broccoli',
            51: 'carrot',
            52: 'hot dog',
            53: 'pizza',
            54: 'donut',
            55: 'cake',
            56: 'chair',
            57: 'couch',
            58: 'potted plant',
            59: 'bed',
            60: 'dining table',
            61: 'toilet',
            62: 'tv',
            63: 'laptop',
            64: 'mouse',
            65: 'remote',
            66: 'keyboard',
            67: 'cell phone',
            68: 'microwave',
            69: 'oven',
            70: 'toaster',
            71: 'sink',
            72: 'refrigerator',
            73: 'book',
            74: 'clock',
            75: 'vase',
            76: 'scissors',
            77: 'teddy bear',
            78: 'hair drier',
            79: 'toothbrush'
        }
        # 为类别生成颜色调色板
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))

        # 初始化ONNX会话
        self.session = self.initialize_session(self.onnx_model)

    def initialize_session(self, onnx_model):
        """
        初始化ONNX模型会话。
        返回:
            onnxruntime.InferenceSession: ONNX推理会话。
        """
        if torch.cuda.is_available():
            print("Using CUDA")
            providers = ["CUDAExecutionProvider"]
        else:
            print("Using CPU")
            providers = ["CPUExecutionProvider"]
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(onnx_model,
                                       session_options=session_options,
                                       providers=providers)
        return session

    def preprocess(self, img, input_width, input_height):
        """
        在进行推理之前，对输入图像进行预处理。
        参数:
            img: 原始图像（BGR格式）。
            input_width: 模型输入的宽度。
            input_height: 模型输入的高度。
        返回:
            image_data: 预处理后的图像数据，准备好进行推理。
        """
        # 将图像颜色空间从BGR转换为RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 将图像调整为匹配输入形状
        img_resized = cv2.resize(img_rgb, (input_width, input_height))

        # 将图像数据除以 255.0 进行归一化
        image_data = img_resized.astype(np.float32) / 255.0

        # 转置图像，使通道维度成为第一个维度 (3, input_height, input_width)
        image_data = np.transpose(image_data, (2, 0, 1))

        # 扩展图像数据的维度以匹配期望的输入形状 (1, 3, input_height, input_width)
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)

        return image_data

    def draw_detections(self, img, box, score, class_id):
        """
        根据检测到的对象在输入图像上绘制边界框和标签。
        参数:
            img: 要绘制检测的输入图像。
            box: 检测到的边界框 [x1, y1, w, h]。
            score: 对应的检测得分。
            class_id: 检测到的对象的类别ID。
        返回:
            None
        """
        x1, y1, w, h = box
        color = self.color_palette[class_id].tolist()

        # 在图像上绘制边界框
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

        # 创建包含类名和得分的标签文本
        label = f"{self.classes[class_id]}: {score:.2f}"

        # 计算标签文本的尺寸
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # 计算标签文本的位置
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + label_height + 10

        # 绘制填充的矩形作为标签文本的背景
        cv2.rectangle(
            img,
            (int(label_x), int(label_y - label_height)),
            (int(label_x + label_width), int(label_y + label_height)),
            color,
            cv2.FILLED
        )

        # 在图像上绘制标签文本
        cv2.putText(img, label, (int(label_x), int(label_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    def postprocess(self, img, output, input_width, input_height, img_width, img_height):
        """
        对模型的输出进行后处理，以提取边界框、分数和类别ID。
        参数:
            img (numpy.ndarray): 输入图像。
            output (list): 模型的输出。
            input_width (int): 模型输入的宽度。
            input_height (int): 模型输入的高度。
            img_width (int): 原始图像的宽度。
            img_height (int): 原始图像的高度。
        返回:
            detections: 检测到的目标信息列表。
        """
        outputs = np.transpose(np.squeeze(output[0]))
        rows = outputs.shape[0]
        boxes = []
        scores = []
        class_ids = []
        detections = []

        x_factor = img_width / input_width
        y_factor = img_height / input_height

        for i in range(rows):
            classes_scores = outputs[i][4:]
            max_score = np.amax(classes_scores)

            if max_score >= self.confidence_thres:
                class_id = np.argmax(classes_scores)
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

                left = int((x - w / 2) * x_factor)
                top = int((y - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)

                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])

        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)

        for i in indices:
            i = i[0] if isinstance(i, (list, tuple, np.ndarray)) else i
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]
            self.draw_detections(img, box, score, class_id)
            detection_info = {
                "class_id": int(class_id),
                "class_name": self.classes[class_id],
                "score": float(score),
                "box": {
                    "x": int(box[0]),
                    "y": int(box[1]),
                    "width": int(box[2]),
                    "height": int(box[3])
                }
            }
            detections.append(detection_info)

        return detections

    def run_inference(self, img):
        """
        使用ONNX模型对单张图像进行推理。
        参数:
            img (numpy.ndarray): 原始图像（BGR格式）。
        返回:
            img: 带有检测结果的图像。
            detections: 检测到的目标信息列表。
        """
        model_inputs = self.session.get_inputs()
        input_shape = model_inputs[0].shape
        input_width = input_shape[3]
        input_height = input_shape[2]

        img_height, img_width = img.shape[:2]

        img_data = self.preprocess(img, input_width, input_height)
        outputs = self.session.run(None, {model_inputs[0].name: img_data})

        detections = self.postprocess(img, outputs, input_width, input_height, img_width, img_height)

        return img, detections


def process_images(model_path, img_dir, confidence_thres, iou_thres, output_json):
    """
    处理指定目录下的三张图片，进行目标检测、标注并生成检测结果汇总的 JSON 文件。
    参数:
        model_path (str): ONNX模型的路径。
        img_dir (str): 存放图片的目录路径。
        confidence_thres (float): 置信度阈值。
        iou_thres (float): IoU阈值。
        output_json (str): 输出的 JSON 文件路径。
    """
    # 初始化YOLOv8模型
    detector = YOLOv8(model_path, confidence_thres, iou_thres)

    # 指定要处理的图片名称
    image_names = ["channel1.jpg", "channel2.jpg", "channel3.jpg"]
    results = {}

    for image_name in image_names:
        img_path = os.path.join(img_dir, image_name)
        if not os.path.isfile(img_path):
            print(f"图片 {image_name} 不存在于目录 {img_dir} 中。")
            continue

        # 读取图像
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取图片 {img_path}。")
            continue

        # 进行推理
        annotated_img, detections = detector.run_inference(img)

        # 如果没有检测到目标或目标得分低于阈值，跳过生成标注图像
        if not detections:
            print(f"图片 {image_name} 未检测到目标或所有目标得分低于阈值，跳过标注图像生成。")
            continue

        # 保存标注后的图片
        output_img_path = os.path.join(img_dir, f"annotated_{image_name}")
        cv2.imwrite(output_img_path, annotated_img)
        print(f"标注后的图片已保存至 {output_img_path}")

        # 记录检测结果
        results[image_name] = detections

    # 设置JSON文件的输出路径
    output_json_path = os.path.join(img_dir, output_json)

    # 保存检测结果到 JSON 文件
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"检测结果已保存至 {output_json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用YOLOv11 ONNX模型进行目标检测，并生成检测结果汇总的 JSON 文件。")
    parser.add_argument("--model", type=str, default='ai_detect.onnx', required=False, help="ONNX模型的路径.")
    parser.add_argument("--img_dir", type=str, required=False, default='C:/Users/bondc/Desktop/1',
                        help="存放图片的目录路径.")
    parser.add_argument("--conf-thres", type=float, default=0.5, help="置信度阈值.")
    parser.add_argument("--iou-thres", type=float, default=0.5, help="IoU（交并比）阈值.")
    parser.add_argument("--output-json", type=str, default="detection_results.json", help="输出的 JSON 文件路径.")
    args = parser.parse_args()

    process_images(args.model, args.img_dir, args.conf_thres, args.iou_thres, args.output_json)
