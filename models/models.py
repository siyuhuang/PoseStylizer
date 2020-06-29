
def create_model(opt):
    model = None
    print(opt.model)

    if opt.model == 'PoseStyleNet':
        assert opt.dataset_mode in ['keypoint', 'keypoint_segmentation']
        from .PoseStyleNet import TransferModel
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model = TransferModel()
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
