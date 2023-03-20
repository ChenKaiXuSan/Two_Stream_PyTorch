import sys 

try:
    from data_loader import *
except:
    sys.path.append('/workspace/Two_Stream_PyTorch/project/dataloader')
    from data_loader import *