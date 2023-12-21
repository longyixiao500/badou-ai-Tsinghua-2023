import numpy as np
from keras import backend as K
from utils.config import Config


config = Config()

# 生成锚框
def generate_anchors(sizes=None, ratios=None):
    # 如果没用传入值，将使用默认值
    if sizes is None:  # 尺度
        sizes = config.anchor_box_scales
    if ratios is None:  # 比例
        ratios = config.anchor_box_ratios

    # 计算锚点数量
    num_anchors = len(sizes) * len(ratios)
    # 用于存储锚点
    anchors = np.zeros((num_anchors, 4))
    # 将数组sizes的0维（行）重复2次，1维（列）重复3次（比例数组长度），然后进行转置赋值给anchors的3、4列
    anchors[:, 2:] = np.tile(sizes, (2, len(ratios))).T

    # 遍历 ratios 数组,对于每个比例，将其分别乘以anchors的第3列和第4列，得到新的锚点宽度和高度。
    for i in range(len(ratios)):
        # 3、6、9行
        anchors[3*i: 3*i+3, 2] = anchors[3*i: 3*i+3, 2] * ratios[i][0]
        anchors[3*i: 3*i+3, 3] = anchors[3*i: 3*i+3, 3] * ratios[i][1]
    # 更新anchors矩阵的偶数列和奇数列的值，使其减去对应尺度或比例的一半。
    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T
    return anchors

# 计算锚点的所有可能位置
def shift(shape, anchors, stride=config.rpn_stride):
    """
    计算在给定形状和步长下，锚点的所有可能位置。
    :param shape: 一个包含两个整数的元组，表示网格形状
    :param anchors: 一个二维数组，表示锚点的位置
    :param stride: 步长
    :return:锚点的所有可能位置。
    """
    # K.floatx() 获取当前默认的浮点数类型
    shift_x = (np.arange(0, shape[0], dtype=K.floatx()) + 0.5) * stride
    shift_y = (np.arange(0, shape[1], dtype=K.floatx()) + 0.5) * stride
    # 创建网格
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    # 重塑为一维数组
    shift_x = np.reshape(shift_x, [-1])
    shift_y = np.reshape(shift_y, [-1])
    # 进行堆叠，堆叠后每一行表示锚点的4个坐标
    shifts = np.stack([
        shift_x,
        shift_y,
        shift_x,
        shift_y
    ], axis=0)
    # 转置
    shifts = np.transpose(shifts)
    # 获取行数即锚点数量
    number_of_anchors = np.shape(anchors)[0]
    # 获取锚点所有可能位置的数量
    k = np.shape(shifts)[0]

    # 将anchors重塑为3维数组，和shifts相加，得到所有可能位置锚点
    shifted_anchors = np.reshape(anchors, [1,number_of_anchors,4])+np.array(np.reshape(shifts, [k,1,4]), dtype=K.floatx())
    # 重塑为二维数组，每一行表示一个锚点的所有可能位置
    shifted_anchors = np.reshape(shifted_anchors, [k*number_of_anchors, 4])
    return shifted_anchors

def get_anchors(shape, width, height):
    anchors = generate_anchors()  # 获取生成的锚框
    # 对锚框进行平移操作，将锚框的坐标相对于输入图像的尺寸进行调整
    network_anchors = shift(shape, anchors)
    # 将锚框的坐标值转换为相对于输入图像的比例值
    network_anchors[:, 0] = network_anchors[:, 0]/width
    network_anchors[:, 1] = network_anchors[:, 1]/height
    network_anchors[:, 2] = network_anchors[:, 2]/width
    network_anchors[:, 3] = network_anchors[:, 3]/height
    # 将network_anchors中的元素限制在0到1之间
    network_anchors = np.clip(network_anchors, 0, 1)
    return network_anchors
