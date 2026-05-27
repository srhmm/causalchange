from causalchange.config.cc_config import CausalChangeConfig


class SpaceTimeEngine:
    @classmethod
    def from_config(cls, cfg: CausalChangeConfig) -> "SpaceTimeEngine":
        if cfg.spacetime is None:
            raise ValueError("SpaceTimeEngine requires cfg.spacetime.")

        domain = TimeDomain(tau_max=cfg.spacetime.tau_max)
        scorer = EdgeScoreTime(cfg=cfg)
        search = make_temporal_search(cfg, scorer)

        changepoints = SpaceTimeChangepointDetection(cfg.spacetime)
        partitioning = SpaceTimePartitioning(cfg.spacetime)

        return cls(
            cfg=cfg,
            domain=domain,
            scorer=scorer,
            search=search,
            changepoint_detection=changepoints,
            partitioning=partitioning,
        )
