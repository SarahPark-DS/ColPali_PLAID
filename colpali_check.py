#%%
import importlib.util
from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor

print("fast_plaid installed:", importlib.util.find_spec("fast_plaid") is not None)
print("has create_plaid_index:", hasattr(BaseVisualRetrieverProcessor, "create_plaid_index"))
print("has get_topk_plaid:", hasattr(BaseVisualRetrieverProcessor, "get_topk_plaid"))
#print(help(BaseVisualRetrieverProcessor))
# %%
print([name for name in dir(BaseVisualRetrieverProcessor) if not name.startswith("__")])
# %%
