"""
This module defines the SpeedVAIL class, which extends the VAIL_TRPO class from the imitation_lib
to support training with speed-adaptive demonstrations.

Classes:
- SpeedVAIL: Extends the VAIL_TRPO class to support training with speed-adaptive demonstrations.

Usage:
Import the SpeedVAIL class and use it to create an instance for training agents with speed-adaptive
demonstrations in a specified environment.

Example:
    from speed_vail import SpeedVAIL

    demonstration_collections = {
        1.0: [demo1, demo2],
        1.25: [demo3, demo4],
        1.5: [demo5, demo6]
    }

    vail = SpeedVAIL(demonstration_collections=demonstration_collections, root_speed=1.25)
    vail.fit(dataset, target_speed=1.25)
"""
from imitation_lib.imitation import VAIL_TRPO

class SpeedVAIL(VAIL_TRPO):
    """
    Extends the VAIL_TRPO class to support training with speed-adaptive demonstrations.

    This class allows for training agents using demonstrations collected at different speeds.
    It adapts the training process to use the appropriate demonstrations based on the target speed.

    Attributes:
        root_speed (float): The root speed of the demonstrations provided in demonstration_collections.
        _demonstration_collections (dict): A dictionary with keys as floats and values as lists of
            demonstrations. The keys represent the speed of the demonstrations in the list.

    Methods:
        fit(dataset, **info):
            Fits the model to the provided dataset, adapting to the target speed specified in info.
    """
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
