import os

GPU_HOURLY_USD = float(os.getenv("GPU_HOURLY_USD", "0.50"))
PRICE_PER_1K_TOKENS = float(os.getenv("PRICE_PER_1K_TOKENS", "0.004"))
OVERHEAD_PER_CALL = float(os.getenv("OVERHEAD_PER_CALL", "0.0002"))


def compute_cost(tokens_in: int, tokens_out: int, seconds: float) -> float:
    token_cost = (tokens_in + tokens_out) / 1000.0 * PRICE_PER_1K_TOKENS
    time_cost = (seconds / 3600.0) * GPU_HOURLY_USD
    return round(token_cost + time_cost + OVERHEAD_PER_CALL, 6)
