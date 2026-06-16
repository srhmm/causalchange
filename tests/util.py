def has_rpy2_flexmix() -> bool:
    try:
        import rpy2.robjects  # noqa: F401
        from rpy2.robjects.packages import importr

        importr("flexmix")
        return True
    except Exception:
        return False
