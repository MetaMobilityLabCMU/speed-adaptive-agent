from imitation_lib.imitation import VAIL_TRPO

class SpeedVAIL(VAIL_TRPO):
    def __init__(self, **kwargs):
        """
        Args:
            root_speed (float, optional): Root speed of the demonstrations provided
                in demonstration_collections. Defaults to 1.25.
            demonstration_collections (dict): A dictionary with keys as floats and
                values as lists of demonstrations. The keys represent the speed of
                the demonstrations in the list.
        """
        self.root_speed = kwargs.pop('root_speed', 1.25)
        self._demonstration_collections = kwargs.pop('demonstration_collections')
        kwargs['demonstrations'] = self._demonstration_collections[self.root_speed]
        super(SpeedVAIL, self).__init__(**kwargs)

    def fit(self, dataset, **info):
        target_speed  = info.pop('target_speed', 1.25)
        self._demonstrations = self._demonstration_collections[target_speed]
        # call base fit method
        super(SpeedVAIL, self).fit(dataset)
