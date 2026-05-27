from causalchange.discovery.search.topic import DAGSearchResult


class TemporalTopicSearch:
    def run(
        self,
        *,
        variables: list[str],
        tau_max: int,
        allowed_edge,
        score_fun,
    ) -> DAGSearchResult: ...
