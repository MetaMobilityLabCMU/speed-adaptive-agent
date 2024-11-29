from imitation_lib.imitation import VAIL_TRPO

class SpeedVAIL(VAIL_TRPO):

    def __init__(self, **kwargs):
        self.root_speed = kwargs.pop('root_speed', 1.2)
        self._demonstration_collections = kwargs.pop('demonstration_collections')
        kwargs['demonstrations'] = self._demonstration_collections[self.root_speed]
        super(SpeedVAIL, self).__init__(**kwargs)

    def fit(self, dataset, **info):
        target_speed  = info.pop('target_speed', 1.2)
        self._demonstrations = self._demonstration_collections[target_speed]
        # call base fit method
        super(SpeedVAIL, self).fit(dataset)
