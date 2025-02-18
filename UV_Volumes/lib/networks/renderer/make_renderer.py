import imp
from .uv_volumes import Renderer

def make_renderer(cfg, network):
    # module = cfg.renderer_module
    # print("render module: " ,module)
    # path = cfg.renderer_path
    # print("render path: " ,path)
    # renderer = imp.load_source(module, path).Renderer(network)
    renderer = Renderer(network)
    return renderer
