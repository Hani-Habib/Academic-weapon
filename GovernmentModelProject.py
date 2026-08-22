from types import SimpleNamespace
from typing import Any

import numpy as np

from scipy import optimize

from ConsumerModelProject import ConsumerClass

class GovernmentClass(ConsumerClass):
    """ a government raising revenue from the consumer in Consumer.py

    Two kinds of instrument:

    1) a lump-sum tax T, which reduces income
    2) product taxes tau1, tau2, tau3, which raise the prices

    The consumer is the ConsumerClass, so everything from there is inherited.

    """

    def __init__(self,par=None):

        # a. default setup
        self.setup()
        self.setup_government()

        # b. update parameters
        if not par is None:
            for k,v in par.items():
                self.par.__dict__[k] = v

        # c. remember the situation without taxes
        #    this must happen *after* b., or a changed price would not be picked up
        self.sync_pre_tax()

    def setup_government(self):
        """ add the tax instruments to the parameters """

        par = self.par

        # a. lump-sum tax
        par.T = 0.0 # lump-sum tax (a transfer if negative)

        # b. product taxes
        par.tau1 = 0.0 # tax rate on food
        par.tau2 = 0.0 # tax rate on bus trips
        par.tau3 = 0.0 # tax rate on train trips

    def sync_pre_tax(self):
        """ store the current prices and income as the situation without taxes

        Revenue is collected at these prices, so they must be the ones *before*
        any taxes were added.

        """

        par = self.par

        par.p1_pre = par.p1
        par.p2_pre = par.p2
        par.p3_pre = par.p3
        par.I_pre = par.I

    ##############################
    # 1. what the consumer faces #
    ##############################

    def set_taxes(self,T=0.0,tau1=0.0,tau2=0.0,tau3=0.0):
        """ set the taxes, and update the prices and the income the consumer faces

        The price the consumer pays for good j is (1+tau_j) times the price the
        seller receives, and income is reduced by the lump-sum tax. After this
        call every method inherited from ConsumerClass -- .solve(), .shares(),
        .value_of_choice(), .solve_grid() -- automatically refers to the
        situation *with* taxes.

        Args:

            T (float): lump-sum tax
            tau1 (float): tax rate on food
            tau2 (float): tax rate on bus trips
            tau3 (float): tax rate on train trips

        """

        par = self.par

        # a. remember the taxes
        par.T = T
        par.tau1 = tau1
        par.tau2 = tau2
        par.tau3 = tau3

        # b. the prices the consumer pays
        par.p1 = (1+tau1)*par.p1_pre
        par.p2 = (1+tau2)*par.p2_pre
        par.p3 = (1+tau3)*par.p3_pre

        # c. income after the lump-sum tax
        par.I = par.I_pre - T

    #########################################
    # 2. revenue, and what the consumer gets #
    #########################################

    def tax_revenue(self, opt=None):

        par = self.par

        # solve consumer problem if needed
        if opt is None:
            opt = self.solve(do_print=False)

        # quantities bought
        x1, x2, x3 = self.quantities(opt.s1, opt.w)

        # total tax revenue
        R = (
            par.T
            + par.tau1 * par.p1_pre * x1
            + par.tau2 * par.p2_pre * x2
            + par.tau3 * par.p3_pre * x3
        )

        return R


    def revenue_and_utility(self, tau, goods=(2,)):

        # choose which goods are taxed
        tau1 = tau if 1 in goods else 0.0
        tau2 = tau if 2 in goods else 0.0
        tau3 = tau if 3 in goods else 0.0

        # set taxes
        self.set_taxes(
            tau1=tau1,
            tau2=tau2,
            tau3=tau3
        )

        # solve consumer problem
        opt = self.solve(do_print=False)

        # revenue and utility
        R = self.tax_revenue(opt)
        u = opt.u

        return R, u


    def revenue_and_utility_lump_sum(self, T):

        # only lump-sum tax
        self.set_taxes(T=T)

        # solve consumer problem
        opt = self.solve(do_print=False)

        # revenue and utility
        R = self.tax_revenue(opt)
        u = opt.u

        return R, u


    ##########################################
    # 3. hitting a given revenue requirement #
    ##########################################

def max_revenue(self,goods=(2,),tau_max=10.0,N=1001) -> tuple[Any, Any]:

    tau_vec = np.linspace(0,tau_max,N)
    R_vec = np.empty(N)

    for i,tau_i in enumerate(tau_vec):
        R_vec[i] = self.revenue_and_utility(tau_i,goods)[0]

    i = np.argmax(R_vec)

    tau = tau_vec[i]
    R = R_vec[i]

    return tau,R


def find_tax_rate(self,R_target,goods=(2,),bracket=(1e-10,1.0)) -> float:

    def f(tau):
        R = self.revenue_and_utility(tau,goods)[0]
        return R - R_target

    try:
        res = optimize.root_scalar(
            f,
            bracket=bracket,
            method='brentq'
        )

        tau = res.root

    except ValueError:
        tau = np.nan

    return tau