def load_torch_points3d():
    import os
    import sys
    import importlib
    DIR = os.path.dirname(os.path.realpath(__file__))
    torch_points3d_path = os.path.join(
        DIR, "..", "..", "torch-points3d", "torch_points3d")
    assert os.path.exists(torch_points3d_path)

    MODULE_PATH = os.path.join(torch_points3d_path, "__init__.py")
    MODULE_NAME = "torch_points3d"
    spec = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
