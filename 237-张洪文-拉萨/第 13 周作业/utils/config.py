

class Config:
    def __init__(self):
        self.anchor_box_scales = [128,256,512]  # 锚框尺度
        self.anchor_box_ratios = [[1,1],[1,2],[2,1]]  # 锚框长宽比例
        self.rpn_stride = 16  # rpn网络接受的feature map 和 M*N 的大小比
        self.rpn_min_overlap = 0.3  # RPN网络中提议框与真实框的最小和最大重叠度阈值
        self.rpn_max_overlap = 0.7
        self.num_rois = 32  # 输出保留区域提议数量 32

        self.classifier_min_overlap = 0.1  # 分类器中提议框与真实框的最小和最大重叠度阈值
        self.classifier_max_overlap = 0.5
        self.classifier_regr_std = [8.0, 8.0, 4.0, 4.0]  # 分类器中的目标框回归的标准差，这通常用于确定回归损失的标准差

        self.verbose = True  # 是否打印模型训练详情
        self.model_path = "logs/model.h5"  # 模型路径
