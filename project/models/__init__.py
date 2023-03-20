import sys 

try:
    from make_model import *
    from optical_flow import *
    from pytorchvideo_models import *
except:
    sys.path.append('/workspace/Two_Stream_PyTorch/project/models')

    from make_model import *
    from optical_flow import *
    from pytorchvideo_models import *