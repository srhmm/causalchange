from typing import Any

from causalchange._cc_types import DataMode, ScoreType, MixingType
from causalchange.discovery._mixins import (
    TabularDomainMixin,
    TabularScoreMixin,
    LINCMixin,
    TemporalDomainMixin,
    AutoRegressiveScoreMixin,
    SpaceTimeMixin,
)
from causalchange.discovery._search import TOPICSearch, GLOBESearch


class EstimatorBaseMixin:
    def __init__(
        self,
        *,
        data_mode: DataMode = DataMode.IID,
        score_type: ScoreType = ScoreType.GAM,
        mixing_type: MixingType = MixingType.SKIP,
        score_params: dict[str, Any] | None = None,
        lg=None,
        vb: int = 0,
        **kwargs,
    ):
        super().__init__()
        self.data_mode = data_mode
        self.score_type = score_type
        self.mixing_type = mixing_type
        self.score_params = {} if score_params is None else dict(score_params)
        self.lg = lg
        self.vb = int(vb)
        self._info = lambda st, min_vb=0: (lg.info(st) if lg is not None else print(st)) if self.vb > min_vb else None


class TOPIC(EstimatorBaseMixin, TabularDomainMixin, TabularScoreMixin, TOPICSearch):
    pass


class GLOBE(EstimatorBaseMixin, TabularDomainMixin, TabularScoreMixin, GLOBESearch):
    pass


class LINC(LINCMixin, TOPIC):
    pass


class LINC_GLOBE(LINCMixin, GLOBE):
    pass


class SpaceTime(EstimatorBaseMixin, TemporalDomainMixin, AutoRegressiveScoreMixin, SpaceTimeMixin, TOPICSearch):
    pass


class SpaceTime_GLOBE(EstimatorBaseMixin, TemporalDomainMixin, AutoRegressiveScoreMixin, SpaceTimeMixin, GLOBESearch):
    pass


class SpaceTime_C(LINCMixin, SpaceTime):
    pass


class SpaceTime_GLOBE_C(LINCMixin, SpaceTime_GLOBE):
    pass
