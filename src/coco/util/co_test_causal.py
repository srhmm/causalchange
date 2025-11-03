import scipy


def test_causal(h1, mi, stdev_cent_sampling, t=1.96):
    if stdev_cent_sampling == 0:
        z = h1 - mi
    else:
        z = (h1 - mi) / stdev_cent_sampling  # TODO emi or mean_emi_sampling?

    pval = scipy.stats.norm.sf(abs(z))

    causal = t > z > -t
    return causal, pval