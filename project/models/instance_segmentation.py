import torch
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

class batch_instance_segmentation():
    def __init__(self) -> None:
        
        # set for instance segmentation, with detectron2.
        cfg = get_cfg()
        # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.90  # set threshold for this model
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

        self.predictor = DefaultPredictor(cfg)

    @torch.no_grad()
    def draw_mask_to_img(
        self,
        image: torch.Tensor,
        predictor
    ) -> torch.Tensor:

        if not isinstance(image, torch.Tensor):
            raise TypeError(f"The image must be a tensor, got {type(image)}")
        elif image.dim() != 3:
            raise ValueError("Pass individual images, not batches")

        _dtype = image.dtype
        
        # pred output instance
        outputs = predictor(image.permute(1, 2, 0).cpu().numpy()) # c, h, w to h, w, c

        masks = outputs['instances'].pred_masks # c, h, w
        pred_classes = outputs['instances'].pred_classes # pred class
        image = image.detach().clone().cuda() # h, w, c to c, h, w

        if masks.dtype != torch.bool:
            raise ValueError(f"The masks must be of dtype bool. Got {masks.dtype}")
        elif masks.shape[-2:] != image.shape[-2:]:
            raise ValueError("The image and the masks must have the same height and width")
        elif masks.size()[0] == 0:
            warnings.warn("mask doesn't contain any mask, dont clip image with mask.")
            return image
        
        for i, pred_class in enumerate(pred_classes):
            # only need the person mask.
            if pred_class == 0:
                # convert torch.Bool to torch.float32
                masks_to_draw = masks[i].detach().clone().to(_dtype).cuda() # c, h, w

                # draw mask to image.
                out = image * masks_to_draw
            else:
                out = image

        return out.to(_dtype)

    def handle_batch_video(self, video):

        b, c, t, h, w = video.size()

        batch_masked_video = []

        for batch_index in range(b):
                
            draw_img_list = []

            for frame_index in range(t):
                # print('the time frame is %d' % frame_index)
                drawed_img = self.draw_mask_to_img(video[batch_index, :, frame_index, :], self.predictor) # c, h, w
                draw_img_list.append(drawed_img)

            # check if lost frame when segmentation.
            if len(draw_img_list) != t:
                raise ValueError(f"Frame lost, lost {abs(len(draw_img_list) - t)}")

            stacked_img = torch.stack(draw_img_list, dim=1) # c, t, h, w
            batch_masked_video.append(stacked_img)

        return torch.stack(batch_masked_video, dim=0) # b, c, t, h, w
