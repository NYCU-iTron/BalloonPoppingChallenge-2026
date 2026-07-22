from stable_baselines3.common.callbacks import BaseCallback


class SaveVecNormalizeCallback(BaseCallback):
    def __init__(self, save_path):
        super().__init__()
        self.save_path = save_path
    def _on_step(self) -> bool:
        self.model.get_vec_normalize_env().save(self.save_path)
        return True
