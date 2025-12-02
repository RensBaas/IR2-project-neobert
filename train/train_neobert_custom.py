import sys
import os
from transformers import AutoConfig, AutoTokenizer, AutoModel
from functools import wraps

# Save original functions
original_config = AutoConfig.from_pretrained
original_tokenizer = AutoTokenizer.from_pretrained
original_model = AutoModel.from_pretrained

# Create new versions that always trust remote code
@wraps(original_config)
def patched_config(*args, **kwargs):
    kwargs['trust_remote_code'] = True
    return original_config(*args, **kwargs)

@wraps(original_tokenizer)
def patched_tokenizer(*args, **kwargs):
    kwargs['trust_remote_code'] = True
    return original_tokenizer(*args, **kwargs)

@wraps(original_model)
def patched_model(*args, **kwargs):
    kwargs['trust_remote_code'] = True
    return original_model(*args, **kwargs)

# Replace the functions
AutoConfig.from_pretrained = patched_config
AutoTokenizer.from_pretrained = patched_tokenizer
AutoModel.from_pretrained = patched_model

# Now run the normal training
from tevatron.driver.train import main
main()


