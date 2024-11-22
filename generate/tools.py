import torch
import re

def pre_caption(caption, max_words = 70):
    caption = str(caption)
    
    if caption.endswith("</s>"):
        caption = caption[:-4]
    if caption.endswith("<|im_end|>"):
        caption = caption.replace("<|im_end|>", "")
    caption = caption.strip("\"")
    caption = caption.rstrip('\n')
    caption = caption.strip(' ')

    caption_words = caption.split(' ')
    if len(caption_words) > max_words:
        caption = ' '.join(caption_words[:max_words])

    # if not len(caption):
    #     raise ValueError(f"pre_caption yields invalid text (raw: {caption_raw})")

    return caption
