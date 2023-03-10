import json


class SimConfig:
    def __init__(self, scene_file_path) -> None:
        self.config = None
        with open(scene_file_path, "r") as f:
            self.config = json.load(f)
    
    def get_cfg(self, name, enforce_exist=False):
        if enforce_exist:
            assert name in self.config["Configuration"]
        if name not in self.config["Configuration"]:
            if enforce_exist:
                assert name in self.config["Configuration"]
            else:
                return None
        return self.config["Configuration"][name]