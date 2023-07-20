from .PoseNet import PoseNet


def get_model(model_name, config):
    if model_name == 'posenet':
        return PoseNet(config)
    else:
        raise "{} not supported".format(model_name)