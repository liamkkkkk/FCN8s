import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from model_FCN8s import FCN8s

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 加载训练好的模型
model = FCN8s(n_class=21)
model.load_state_dict(torch.load('fcn8s_trained.pth'))
model = model.to(device)

# 定义标签颜色映射
VOC_COLORMAP = np.array([
    [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
    [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
    [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
    [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
    [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
    [0, 64, 128]
], dtype=np.uint8)


# 定义图像显示函数
def show_image(image, title=None):
    if title:
        plt.title(title)
    plt.imshow(image)
    plt.axis('off')


# 定义结果可视化函数
def visualize_results(model, image_paths, data_dir='./data'):
    model.eval()  # 设置模型为评估模式
    transform = get_transform()

    for img_path in image_paths:
        img_file = os.path.join(data_dir, 'VOCdevkit/VOC2007/JPEGImages', img_path + '.jpg')
        img = Image.open(img_file).convert('RGB')
        input_img = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_img)
            pred = output.argmax(1).squeeze(0).cpu().numpy()

        pred_colormap = VOC_COLORMAP[pred]

        # 显示原始图像和预测结果
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        show_image(img, title='Original Image')
        plt.subplot(1, 2, 2)
        show_image(pred_colormap, title='Predicted Segmentation')
        plt.show()


# 指定需要展示的测试图像ID
test_image_ids = [
    '000068', '000175', '000243', '000333',
    '000346', '000364', '000392', '000452'
]

# 可视化结果
visualize_results(model, test_image_ids)
