# torch_patch.py
# This is a hack to prevent Streamlit's file watcher from trying to watch torch._classes
import sys
import types

class FakeModule(types.ModuleType):
    def __init__(self, name):
        super().__init__(name)
        self.__path__ = []

# Apply the patch before importing anything else
sys.modules['torch._classes'] = FakeModule('torch._classes')