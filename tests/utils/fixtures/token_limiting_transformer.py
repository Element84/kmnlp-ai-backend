from tests.experiment.util.token_limiting_transformer import TokenLimitingTransformer

_s = None


def shared_token_limiting_transformer() -> TokenLimitingTransformer:
    global _s  # noqa: PLW0603
    if _s is None:
        _s = TokenLimitingTransformer()

    return _s
