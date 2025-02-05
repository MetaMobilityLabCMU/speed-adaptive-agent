"""
This module defines the SpeedVAIL class, which extends the VAIL_TRPO class from the imitation_lib
to support training with speed-adaptive demonstrations.
"""
from imitation_lib.imitation import VAIL_TRPO

class SpeedVAIL(VAIL_TRPO):
    def __init__(self, **kwargs):
        self.root_speed = kwargs.pop('root_speed', 1.25)
        self._demonstration_collections = kwargs.pop('demonstration_collections')
        kwargs['demonstrations'] = self._demonstration_collections[self.root_speed]
        super().__init__(**kwargs)

    def fit(self, dataset, **info):
        target_speed  = info.pop('target_speed', 1.25)
        self._demonstrations = self._demonstration_collections[target_speed]
        # call base fit method
        super().fit(dataset)
