import pandas as pd

class GradientBoostingTrainer:
    def _initialize_model(self) -> float:
        pass
    def _draw_subsample(self) -> pd.DataFrame:
        pass
    def _compute_pseudo_residuals(self) -> pd.Series:
        pass
    def _fit_weak_learner(self) -> None:
        pass
    def _update_model(self) -> None:
        pass
    def _check_early_stopping(self) -> bool:
        pass
