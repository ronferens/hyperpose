from .hyperpose.HyperPose import HyperPose


def get_model(model_name, backbone_path, config):
    """
    Get the instance of the request model
    :param model_name: (str) model name
    :param backbone_path: (str) path to a .pth backbone
    :param config: (dict) config file
    :return: instance of the model (nn.Module)
    """
    if model_name == 'hyperpose':
        return HyperPose(config, backbone_path)
    else:
        raise "{} not supported".format(model_name)