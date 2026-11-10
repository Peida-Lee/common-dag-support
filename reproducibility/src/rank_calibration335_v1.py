"""Inclusive outward rank decisions."""
from bisect import bisect_left
from fractions import Fraction


def rank_decision(profile_interval, lower_calibrators, upper_calibrators, alpha=Fraction(1,20)):
    lower,upper = map(Fraction,profile_interval)
    size = len(lower_calibrators)
    assert size == len(upper_calibrators) and 0 < alpha < 1
    count_lower = size - bisect_left(lower_calibrators,upper)
    count_upper = size - bisect_left(upper_calibrators,lower)
    p_lower,p_upper = Fraction(1+count_lower,size+1),Fraction(1+count_upper,size+1)
    assert p_lower <= p_upper
    status = "certified" if p_upper <= alpha else "not_certified" if p_lower > alpha else "numerically_unresolved"
    return dict(status=status, raw_P_interval=[str(p_lower),str(p_upper)],
                raw_P_display=[float(p_lower),float(p_upper)], threshold=str(alpha),
                tail_counts=[count_lower,count_upper], calibration_size=size)
