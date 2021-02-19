from .ModelWithLoss import PipelineV2, PipelineV3, PipelineFiltering, ModelandDispLoss


class SingleFrameMPI:
    def __init__(self):
        self.modelloss = ModelandDispLoss()

    def inferframes(self, refimgs):
        # todo: will need this or not?
        pass
