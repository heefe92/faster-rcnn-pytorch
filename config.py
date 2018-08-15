
class Config:
    # data
    voc_data_dir = '/media/heecheol/새 볼륨/DataSet/VOC2007/'
    min_size = 608  # image resize
    max_size = 1024 # image resize
    num_workers = 8
    test_num_workers = 1

    # sigma for l1_smooth_loss
    rpn_sigma = 3.
    roi_sigma = 1.

    # param for optimizer
    # 0.0005 in origin paper but 0.0001 in tf-faster-rcnn
    weight_decay = 0.0005
    lr_decay = 0.1  # 1e-3 -> 1e-4
    lr = 1e-3

    # sigma for l1_smooth_loss
    rpn_sigma = 3.
    roi_sigma = 1.

    # training
    epoch = 14

    batch_size=1



opt = Config()
