import pandas as pd
import numpy as np
from typing import Tuple, List
import scipy.stats as stats


## Load data and convert to easier types:
def load_data() -> pd.DataFrame:
    """
    Loads the data and converts it to a pandas DataFrame.
    """
    
    df = (
    pd.read_excel("rotterdam_dataset_15.xlsx")
      .sort_values("Arrival_Date")
      .reset_index(drop=True)
)
    df['Vessel_Type'] = df['Vessel_Type'].astype('category')
    df['Arrival_Day'] = df['Arrival_Date'].dt.date
    df['Arrival_Month'] = df['Arrival_Date'].dt.month

    return df

## For calculating moments and fitting the distributions:
def moments(data: pd.Series | np.ndarray) -> Tuple[float, float]:
    """
    Calculates the first two moments
    """

    M1 = np.mean(data)
    M2 = np.mean(np.pow(data,2))

    return M1, M2

def fit_normal(mom: Tuple[float, float]):
    M1, M2 = mom
    return stats.norm(loc=M1, scale=np.sqrt(M2 - M1**2))

def fit_gamma(mom: Tuple[float, float]):
    M1, M2 = mom
    beta = M1 / (M2 - M1**2)
    alpha = M1 * beta
    return stats.gamma(a = alpha, scale = 1/beta)

def fit_lognormal(mom: Tuple[float, float]):
    M1, M2 = mom
    var = M2 - M1 ** 2
    sigma2 = np.log(1 + var / M1 ** 2)
    mu = np.log(M1) - sigma2 / 2
    return stats.lognorm(s=np.sqrt(sigma2), scale=np.exp(mu))


## For simulating the Poisson process(es):
def simPoissonProcess(lam, t):
    interArrivalTimeDist = stats.expon(scale=1/lam)
    Nt = 0
    time = interArrivalTimeDist.rvs()
    while time < t:
        # arrival will be counted
        Nt += 1
        time += interArrivalTimeDist.rvs()
    return Nt

def simCPP(
        T: int,
        lambdas: Tuple[float, float, float],
        dists: tuple) -> list:

    lambda_C, lambda_T, lambda_B = lambdas
    dist_C, dist_T, dist_B = dists

    N_C = simPoissonProcess(lambda_C, T)
    N_T = simPoissonProcess(lambda_T, T)
    N_B = simPoissonProcess(lambda_B, T)
    N_total = N_C + N_T + N_B

    S_C = S_T = S_B = 0
    if N_C > 0:
        S_C = np.sum(dist_C.rvs(N_C))
    if N_T > 0:
        S_T = np.sum(dist_T.rvs(N_T))
    if N_B > 0:
        S_B = np.sum(dist_B.rvs(N_B))
    S_total = S_C + S_T + S_B

    return [N_C, N_T, N_B, N_total, S_C, S_T, S_B, S_total]


def simPoissonProcessSamplePath(lam: float, T: int) -> Tuple[np.ndarray, int]:
    interArrivalTimeDist = stats.expon(scale=1/lam)

    Nt = 0
    time = interArrivalTimeDist.rvs()
    arrivalTimes = np.array([])

    while time < T:
        # arrival will be counted
        Nt += 1
        arrivalTimes = np.append(arrivalTimes, time)
        time += interArrivalTimeDist.rvs()
    return Nt, arrivalTimes

def simDailyCPP(
        monthly_lambdas: list,
        dists: tuple,
        month_lengths: list) -> np.ndarray:
    """
    Simulates one year day by day. monthly_lambdas[m] holds (lambda_C, lambda_T, lambda_B)
    for month m, so constant rates give the basic scenario and monthly rates the seasonal one.
    Returns an array (days, 8) with the same columns as simCPP, one row per day.
    """
    n_days = sum(month_lengths)
    daily = np.zeros((n_days, 8))

    start = 0
    for m, length in enumerate(month_lengths):
        for k, (lam, dist) in enumerate(zip(monthly_lambdas[m], dists)):
            Nt, arrivalTimes = simPoissonProcessSamplePath(lam, length)
            days = (start + np.floor(arrivalTimes)).astype(int)

            daily[:, k] += np.bincount(days, minlength=n_days)
            if Nt > 0:
                daily[:, 4 + k] += np.bincount(days, weights=dist.rvs(Nt), minlength=n_days)
        start += length

    daily[:, 3] = daily[:, 0:3].sum(axis=1)
    daily[:, 7] = daily[:, 4:7].sum(axis=1)
    return daily

def applyWeather(
        daily_S: np.ndarray,
        p_severe: float,
        p_mild: float,
        d_mild: float = 0.5) -> np.ndarray:
    """
    Delays cargo on days with bad weather. Each day is, independently,
    - a storm with probability p_severe: nothing is handled, all cargo moves to the next day,
    - a mild disruption with probability p_mild: a fraction d_mild moves to the next day,
    - a normal day otherwise: everything, including cargo carried over, is handled.
    Returns the cargo actually handled on each day.
    """
    handled = np.zeros(len(daily_S))
    carry = 0

    for day in range(len(daily_S)):
        u = np.random.uniform()
        if u < p_severe:
            d = 1
        elif u < p_severe + p_mild:
            d = d_mild
        else:
            d = 0

        load = daily_S[day] + carry
        handled[day] = (1 - d) * load
        carry = d * load

    return handled
