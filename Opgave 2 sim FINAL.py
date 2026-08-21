import numpy as np
from scipy.stats import norm

## 2.1 parameters

# demographics
N = 50_000 # number of individuals
age_start = 18 # starting age
age_retire = 65 # retirement age

n_years = age_retire - age_start # 47 years, ages 18-64
ages = np.arange(age_start, age_retire)

# education: short, medium, long
educations = ['short', 'medium', 'long']
p_e = np.array([0.40, 0.35, 0.25]) # shares
S_e = np.array([1, 3, 5])  # years of education
h_e0 = np.array([1.00, 1.20, 1.55]) # initial human capital
delta_e = np.array([0.010, 0.020, 0.030]) # growth of human capital

# human capital
delta = 0.06 # depreciation while unemployed
sigma_psi = 0.10 # std of psi

# labor market
lam = 0.60 # job-finding probability
sigma = 0.05 # job-separation probability

# income
y_SU = 0.45 # student grant while under education
rho = 0.60 # replacement rate while unemployed
y_floor = 0.35 # floor for those never employed

# simulation
seed = 2026


## 2.2 simulating the income distribution

def draw_shocks(rng, N=N, n_years=n_years, sigma_psi=sigma_psi, p_e=p_e,
                S_e=S_e):
    """ Draw human capital shocks, educations and uniforms

    Args:
        rng (Generator): random number generator
        N (int): number of individuals
        n_years (int): number of periods
        sigma_psi (float): std. of the log human capital shock
        p_e (ndarray): education probabilities
        S_e (ndarray): years of education by education level

    Returns:
        psi (ndarray): human capital shocks
        education_level_i (ndarray): draws education levels
        S_e_i (ndarray): years of education of each individual
        uniforms (ndarray): uniforms for the labor market transitions
    """

    # human capital shocks, mean one
    psi = rng.lognormal(-0.5 * sigma_psi**2, sigma_psi,size=(N, n_years))

    # education of each individual
    education_level_i = rng.choice(3, size=N, p=p_e)
    S_e_i = S_e[education_level_i]

    # uniforms driving the labor market transitions
    uniforms = rng.uniform(0.0, 1.0, size=(N, n_years))

    return psi, education_level_i, S_e_i, uniforms


def check_education_shares(education_level_i, p_e=p_e, educations=educations):
    """ compare the simulated education shares with the target p_e

    Args:
        education_level_i (ndarray): education level of each individual
        p_e (ndarray): target education probabilities
        educations (list): names of the education levels
    """

    for e in range(3):

        share = np.mean(education_level_i == e)

        print(f'Share {educations[e]:<7s}{share:.4f} (target {p_e[e]})')


def simulate_employment(S_e_i, uniforms, N=N, n_years=n_years, lam=lam,
                        sigma=sigma):
    """ Simulate employment over the 47 years
    Args:

        S_e_i (ndarray): years of education of each individual
        uniforms (ndarray): uniforms
        N (int): number of individuals
        n_years (int): number of periods
        lam (float): job-finding probability
        sigma (float): job-separation probability
    Returns:
        employed (ndarray): 1 if employed, 0 otherwise
    """

    employed = np.zeros((N, n_years))

    for i in range(N):
        for t in range(n_years):

            if t < S_e_i[i]:
                # Checking if they are under education
                employed[i, t] = 0

            elif t == S_e_i[i]: # just finished education
                employed[i, t] = 0

            else:
                # already in the labor market
                if employed[i, t - 1] == 0:
                    if uniforms[i, t] < lam:
                        employed[i, t] = 1
                    else:
                        employed[i, t] = 0
                else:
                    if uniforms[i, t] < sigma:
                        employed[i, t] = 0
                    else:
                        employed[i, t] = 1

    return employed


def check_unemployment_rate(employed, S_e_i, t_values, age_start=age_start,
                            lam=lam, sigma=sigma):
    """ Compare the simulated unemployment rate with the steady state

    Individuals still in education are not counted in the labor force.
    Args:
        employed (ndarray): employment
        S_e_i (ndarray): years of education of each individual
        t_values (list): periods to report
        age_start (int): starting age
        lam (float): job-finding probability
        sigma (float): job-separation probability
    """
    print('Comparing unemployment rate to theoretical:')

    for t in t_values:

        # Checking if in labor force
        in_labor_force = t >= S_e_i
        unemployment_rate = np.mean(employed[in_labor_force, t] == 0)

        print(f't = {t} (age {age_start + t}): '
              f'unemployment rate = {unemployment_rate:.4f}')

    theoretical = sigma / (sigma + lam)
    print(f'\nTheoretical steady-state unemployment rate:'
          f' sigma/(sigma+lam) = {theoretical:.4f}')


def simulate_income(S_e_i, employed, psi, education_level_i, N=N,
                    n_years=n_years, h_e0=h_e0, delta_e=delta_e,
                    delta=delta, y_SU=y_SU, rho=rho, y_floor=y_floor):
    """ simulate human capital and income

    Args:
        S_e_i (ndarray): years of education of each individual
        employed (ndarray): employment
        psi (ndarray): human capital shocks
        education_level_i (ndarray): education level of each individual
        N (int): number of individuals
        n_years (int): number of periods
        h_e0 (ndarray): initial human capital by education
        delta_e (ndarray): growth of human capital by education
        delta (float): depreciation while unemployed
        y_SU (float): student grant
        rho (float): replacement rate while unemployed
        y_floor (float): floor for those never employed
    Returns:
        income (ndarray)
        human_capital (ndarray)
    """

    income = np.zeros((N, n_years))
    human_capital = np.zeros((N, n_years))

    last_job_income = np.zeros(N) # income in the most recent job
    ever_employed = np.zeros(N, dtype=bool)  # has held a job yet?

    for i in range(N):
        for t in range(n_years):

            e = education_level_i[i]

            if t < S_e_i[i]:
                # still in education: receives the student grant
                income[i, t] = y_SU
                human_capital[i, t] = 0

            else:
                # human capital
                if t == S_e_i[i]:
                    # just finished school: education-specific level
                    human_capital[i, t] = h_e0[e]
                elif employed[i, t] == 1:
                    # employed: grows at the education-specific rate
                    human_capital[i, t] = (human_capital[i, t - 1]
                                           * (1 + delta_e[e]) * psi[i, t])
                else:
                    # unemployed: depreciates
                    human_capital[i, t] = (human_capital[i, t - 1]
                                           * (1 - delta) * psi[i, t])

                # income
                if employed[i, t] == 1:
                    income[i, t] = human_capital[i, t]
                    last_job_income[i] = income[i, t]
                    ever_employed[i] = True
                else:
                    if ever_employed[i]:
                        # unemployed income
                        income[i, t] = rho * last_job_income[i]
                    else:
                        # never had a job:
                        income[i, t] = y_floor

    return income, human_capital


## 2.3 the Gini coefficient

def gini_coefficient(y):
    """ Twice the area between the 45-degree line (perfect equality) and the
        cumulated income share (the Lorenz curve).

    Args:
        y (ndarray): incomes, assumed non-negative
    Returns:
        gini (float)
    """

    sorted_y = np.sort(y)
    n = len(sorted_y)

    #cumulated % of total income and cumulated % of population
    cumulative_income_share = np.cumsum(sorted_y) / np.sum(sorted_y)
    population_share = np.arange(1, n + 1) / n

    gini = 2 * np.sum(population_share - cumulative_income_share) / n

    return gini


def gini_lognormal(sigma):
    return 2 * norm.cdf(sigma / np.sqrt(2)) - 1


def lorenz_curve(y):
    """ cumulated shares for plotting a Lorenz curve

    Args:
        y (ndarray): incomes
    Returns:
        population_share (ndarray): cumulated share of the population
        cumulative_income_share (ndarray): cumulated share of income
    """

    sorted_y = np.sort(y)
    n = len(sorted_y)

    cumulative_income_share = np.cumsum(sorted_y) / np.sum(sorted_y)
    population_share = np.arange(1, n + 1) / n

    return population_share, cumulative_income_share


def gini_by_age(income, n_years=n_years):
    """ Gini coefficient within each age

    Args:
        income (ndarray): income
        n_years (int): number of periods
    Returns:
        gini (ndarray): Gini coefficient for each age
    """

    gini = np.zeros(n_years)

    for t in range(n_years):
        gini[t] = gini_coefficient(income[:, t])

    return gini


## 2.4 what drives inequality?

def average_education_parameters(p_e=p_e, S_e=S_e, h_e0=h_e0,
                                 delta_e=delta_e):
    """ Education parameters with "identical" individuals

    Everybody is given the population-weighted average years of
    education, initial human capital and growth rate.

    Args:
        p_e (ndarray): education probabilities
        S_e (ndarray): years of education by education level
        h_e0 (ndarray): initial human capital by education level
        delta_e (ndarray): growth of human capital by education level
    Returns:
        S_e_avg (int): average years of education
        h_e0_avg (ndarray): average initial human capital, same for all
        delta_e_avg (ndarray): average growth rate, same for all
    """

    S_e_avg = int(round(np.dot(p_e, S_e)))
    h_e0_avg = np.full(3, np.dot(p_e, h_e0))
    delta_e_avg = np.full(3, np.dot(p_e, delta_e))

    return S_e_avg, h_e0_avg, delta_e_avg


def employment_always_employed(S_e_i, N=N, n_years=n_years):
    """ Employment when there is no unemployment

    Everybody is employed in every period from the year they finish education.

    Args:
        S_e_i (ndarray): years of education of each individual
        N (int): number of individuals
        n_years (int): number of periods
    Returns:
        employed (ndarray): 1 if employed, 0 otherwise
    """

    employed = np.zeros((N, n_years))

    for i in range(N):
        for t in range(n_years):
            if t < S_e_i[i]:
                employed[i, t] = 0
            else:
                employed[i, t] = 1

    return employed


## 2.5 extension: risk of injury

def simulate_income_injury(S_e_i, uniforms, psi, education_level_i,
                           p_injury_e, p_recovery, delta_injury, N=N,
                           n_years=n_years, h_e0=h_e0, delta_e=delta_e,
                           delta=delta, lam=lam, sigma=sigma, y_SU=y_SU,
                           rho=rho, y_floor=y_floor):
    """ We choose to add risk to the model in order to simulate
        different working conditions for different levels of education.
        While employed, each period there is a risk of getting injured
        delta_injury. If an individual is injured, they get the same income
        but their human capital depreciates at a faster rate than 
        if they were unemployed.

    Args:
        S_e_i (ndarray): years of education of each individual
        uniforms (ndarray): uniforms
        psi (ndarray): human capital shocks
        education_level_i (ndarray): education level of each individual
        p_injury_e (ndarray): injury probability by education level
        p_recovery (float): probability of recovering each period
        delta_injury (float): depreciation while injured
        N (int): number of individuals
        n_years (int): number of periods
        h_e0 (ndarray): initial human capital by education
        delta_e (ndarray): growth of human capital by education
        delta (float): depreciation while unemployed
        lam (float): job-finding probability
        sigma (float): job-separation probability
        y_SU (float): student grant
        rho (float): replacement rate while unemployed
        y_floor (float): floor for those never employed
    Returns:
        income (ndarray)
        human_capital (ndarray)
        status (ndarray): 0=unemployed, 1=employed, 2=injured
    """

    status = np.zeros((N, n_years))
    human_capital = np.zeros((N, n_years))
    income = np.zeros((N, n_years))

    last_job_income = np.zeros(N)
    ever_employed = np.zeros(N, dtype=bool)

    for i in range(N):
        for t in range(n_years):

            e = education_level_i[i]

            if t < S_e_i[i]:
                #still in education
                income[i, t] = y_SU
                human_capital[i, t] = 0

            else:
                # status
                if t == S_e_i[i]:
                    # just finished education:
                    status[i, t] = 0
                else:
                    prev_status = status[i, t - 1]
                    draw = uniforms[i, t]

                    if prev_status == 1: # was employed:
                        p_inj = p_injury_e[e]
                        if draw < p_inj:
                            status[i, t] = 2 # injured
                        elif draw < p_inj + sigma:
                            status[i, t] = 0 # unemployed
                        else:
                            status[i, t] = 1 # employed
                    elif prev_status == 2: # was injured:
                        status[i, t] = 1 if draw < p_recovery else 2
                    else:
                        # was unemployed:
                        status[i, t] = 1 if draw < lam else 0

                # human capital
                if t == S_e_i[i]:
                    human_capital[i, t] = h_e0[e]
                elif status[i, t] == 1:
                    human_capital[i, t] = (human_capital[i, t - 1]
                                           * (1 + delta_e[e]) * psi[i, t])
                elif status[i, t] == 2:
                    # depreciates faster while injured
                    human_capital[i, t] = (human_capital[i, t - 1]
                                           * (1 - delta_injury)* psi[i, t])
                else:
                    human_capital[i, t] = (human_capital[i, t - 1]
                                           * (1 - delta) * psi[i, t])

                # income
                if status[i, t] == 1:
                    income[i, t] = human_capital[i, t]
                    last_job_income[i] = income[i, t]
                    ever_employed[i] = True
                elif status[i, t] == 2:
                    # injured: keeps the salary from before the injury
                    income[i, t] = last_job_income[i]
                else:
                    if ever_employed[i]:
                        income[i, t] = rho * last_job_income[i]
                    else:
                        income[i, t] = y_floor

    return income, human_capital

