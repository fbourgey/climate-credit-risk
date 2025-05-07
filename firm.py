"""
Firm class for climate credit risk analysis.

This module defines the `Firm` class, which models a single firm's carbon
emission optimization with respect to a benchmark emission. It includes methods
for:
- Calculating optimal emission strategies.
- Computing production statistics (mean, variance).
- Evaluating default probabilities and boundaries.
- Simulating production paths using Monte Carlo methods.

The module also integrates climate scenarios and sector-specific data for
analysis.
"""

import numpy as np
import pandas as pd
from scipy import interpolate, optimize, stats

import utils


class Firm:
    """
    Single firm optimizing its carbon emission with respect to a benchmark
    emission `e_bench`.
    """

    def __init__(
        self,
        prod_0,
        r,
        sig,
        a,
        b,
        c,
        w_1,
        w_2,
        n_units,
        t_final,
        alpha,
        beta,
        theta,
        scenario,
        sector,
    ):
        """
        Initialize the set of parameters for the portfolio loss.

        Parameters
        ----------
        prod_0 : float
            Initial production of the firm, must be positive.
        r : float
            Interest rate, must be positive.
        sig : float
            Volatility modeling uncertainty in demand, must be positive.
        a : float
            Average production level. Constant term in the drift function.
        b : float
            Mean-reverting parameter.
        c : np.ndarray
            Firm's dependence to CO2 emission.
        w_1 : float
            Penalty coefficient.
        w_2 : float
            Reward coefficient.
        n_units : float
            Number of units sold.
        t_final : float
            Final maturity.
        sector : str
            Sector name. Must of one SECTORS.
        scenario : str
            Climate scenario (SSP) name. Must of one SCENARIOS.
        alpha : np.ndarray, optional
            Linear coefficient(s) for cost function, by default np.zeros(1).
        beta : np.ndarray, optional
            Quadratic coefficient(s) for cost function, by default
            0.5*np.ones(1).
        theta : np.ndarray, optional
            Inverse of emission factor "energy / CO2e ratio" for each energy e,
            by default np.ones(1).
        """

        if prod_0 <= 0:
            raise ValueError("Initial production `prod_0` must be positive.")

        if r <= 0:
            raise ValueError("Interest rate `r` must be positive.")

        if sig <= 0:
            raise ValueError("Volatility `sig` must be positive.")

        if alpha.shape[0] != c.shape[0]:
            raise ValueError("`alpha` and `c` must have the same shape.")

        if beta.shape[0] != c.shape[0]:
            raise ValueError("`beta` and `c` must have the same shape.")

        if theta.shape[0] != c.shape[0]:
            raise ValueError("`theta` and `c` must have the same shape.")

        if scenario not in utils.SCENARIOS:
            raise ValueError(f"`scenario` not in {utils.SCENARIOS}.")

        if sector not in utils.SECTORS:
            raise ValueError(f"`sector` not in {utils.SECTORS}.")

        self.n_energy = c.shape[0]
        self.prod_0 = prod_0
        self.r = r
        self.sig = sig
        self.a = a
        self.b = b
        self.c = c
        self.w_1 = w_1
        self.w_2 = w_2
        self.n_units = n_units
        self.t_final = t_final
        self.theta = theta
        self.alpha = alpha
        self.beta = beta
        self.gamma_zero_w = self.gamma_unpenalized()
        self.sector = sector
        self.scenario = scenario

        if self.w_1 == 0.0 and self.w_2 == 0.0:
            # unpenalized scenario
            def f_scenario(t):
                return self.gamma_zero_w * np.ones_like(t)

            self.f_scenario = f_scenario

        # Load SSP data
        df_ssp = pd.read_csv("data/SSP_CMIP6_201811.csv")
        # Filter data for the given scenario and the world region
        df0 = df_ssp[
            (df_ssp["REGION"] == "World")
            & (df_ssp["VARIABLE"] == f"CMIP6 Emissions|CO2|{self.sector}")
        ].reset_index(drop=True)
        df = df0[df0["SCENARIO"].isin(utils.SCENARIOS)][
            [
                "SCENARIO",
                "2015",
                "2020",
                "2030",
                "2040",
                "2050",
                "2060",
                "2070",
                "2080",
                "2090",
                "2100",
            ]
        ]
        ssp_values = df[df["SCENARIO"] == self.scenario].iloc[0, 1:]
        self.ssp_values = ssp_values.values.astype(float)
        self.ssp_values *= self.gamma_zero_w / float(self.ssp_values[0])
        # Initial year 2015 corresponds to t=0
        self.ssp_time = np.array(
            [0.0, 5.0, 15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0, 85.0]
        )
        # Interpolatation function for the scenario
        self.f_scenario = interpolate.PchipInterpolator(
            x=self.ssp_time,
            y=self.ssp_values,
        )

    def gamma_unpenalized(self) -> float:
        """
        Return the optimal constant emission for the unpenalized case
        (w1 = w2 = 0).

        Returns
        -------
        float
            The unpenalized optimal constant emission.
        """
        gamma_zero_w_e = (
            self.c * self.theta / (self.r + self.b) - self.alpha * self.theta
        ) / (2.0 * self.beta * self.theta)

        return float(np.sum(gamma_zero_w_e))

    def gamma_optimal_energy(self, t: np.ndarray) -> np.ndarray:
        """
        Return the optimal emission strategy per energy.

        Parameters
        ----------
        t : np.ndarray
            Time.

        Returns
        -------
        gamma_opt: np.ndarray, shape is (number of energy, shape of 't')
            Optimal emission per energy e.
        """

        c_theta = self.c * self.theta
        alpha_theta = self.alpha * self.theta
        beta_theta = self.beta * self.theta
        xi_1 = self.w_1 / np.sum(beta_theta)
        xi_2 = self.w_2 / np.sum(beta_theta)

        # Compute excess and lack of emissions
        gamma_excess = (2.0 * self.w_1 / (1 + xi_1)) * np.maximum(
            self.gamma_unpenalized() - self.f_scenario(t), 0.0
        )
        gamma_lack = (2.0 * self.w_2 / (1 - xi_2)) * np.maximum(
            self.f_scenario(t) - self.gamma_unpenalized(), 0.0
        )

        # Compute optimal emission per energy
        gamma_opt_per_e = (
            c_theta[:, None] / (self.r + self.b)
            - alpha_theta[:, None]
            - gamma_excess[None, :]
            - gamma_lack[None, :]
        )

        if not np.all(gamma_opt_per_e >= 0):
            raise ValueError("The optimal emission gamma has to be positive.")

        gamma_opt_per_e /= 2.0 * beta_theta[:, None]

        return gamma_opt_per_e

    def gamma_optimal(self, t: np.ndarray) -> np.ndarray:
        """
        Return the firm's optimal emission strategy, which is the sum of
        all optimal emissions per energy.

        Parameters
        ----------
        t : np.ndarray
            Time.

        Returns
        -------
        np.ndarray, shape of 't'
            The firm's optimal emission strategy.
        """
        return self.gamma_optimal_energy(t=t).sum(axis=0)

    def mean_production(self, u: float, t: float, n_leg=20) -> float:
        """
        Return the mean of the firm's production at time u conditionally on
        time t.

        Parameters
        ----------
        u : np.ndarray
            Time points at which to calculate the mean.

        Returns
        -------
        np.ndarray
            Mean of the firm's production at time u.
        """
        if not u >= t:
            raise ValueError("u must be greater or equal than t.")

        knots, weights = utils.gauss_legendre(a=t, b=u, n=n_leg)
        c_theta = self.c * self.theta
        c_theta_gam = np.sum(
            c_theta[:, None] * self.gamma_optimal_energy(t=knots),
            axis=0,
        )
        mean_p = (self.a / self.b) * (1.0 - np.exp(-self.b * (u - t)))
        mean_p += np.sum(weights * np.exp(-self.b * (u - knots)) * c_theta_gam)
        return mean_p

    def variance_production(self, u: np.ndarray) -> np.ndarray:
        """
        Return the variance of the firm's production at time u.

        Parameters
        ----------
        u : np.ndarray
            Time points at which to calculate the variance.

        Returns
        -------
        np.ndarray
            Variance of the firm's production at time u.
        """
        return self.sig**2 * (1.0 - np.exp(-2.0 * self.b * u)) / (2.0 * self.b)

    def optimal_log_prod(self, tab_t, n_mc: int = 1, seed=None):
        """
        Compute the optimal log production.

        Parameters
        ----------
        tab_t: array-like with shape (n_time, 1)
            Time points.
        n_mc : int, optional
            Number of Monte Carlo simulations (default is 100).
        seed : int, optional
            Seed for the random number generator (default is None).

        Returns
        -------
        log_prod : array-like
            Optimal log production at each time point.
        """
        n_disc = tab_t.shape[0]

        if seed is not None:
            np.random.seed(seed)

        z = np.random.randn(n_disc, n_mc)
        log_prod = np.zeros_like(z)
        log_prod[0, :] = np.log(self.prod_0)

        for i in range(n_disc - 1):
            dt_i = tab_t[i + 1] - tab_t[i]
            mean_i = (
                self.mean_production(u=tab_t[i + 1], t=tab_t[i])
                + np.exp(-self.b * dt_i) * log_prod[i, :]
            )
            std_i = np.sqrt(self.variance_production(dt_i))
            log_prod[i + 1, :] = mean_i + std_i * z[i, :]

        return log_prod

    def _compute_integrand_components(self, knots, x, t, n_leg: int = 20):
        """
        Helper function to compute the components of the integrand used in the
        firm's value calculation.

        This method calculates various components required for the integration
        in the `h` method, including discount factors, mean production,
        penalization, and reward terms.

        Parameters
        ----------
        knots : np.ndarray
            Time points (knots) for Gauss-Legendre quadrature.
        x : np.ndarray
            Log production values at the current time `t`.
        t : float
            Current time.
        n_leg : int, optional
            Number of nodes for Gauss-Legendre quadrature (default is 20).

        Returns
        -------
        tuple
            A tuple containing the following components:
            - discount_knots : np.ndarray
                Discount factors for each knot.
            - integrand_mean_cov : np.ndarray
                Mean and covariance term of the integrand.
            - integrand_alpha_beta : np.ndarray
                Alpha and beta term of the integrand.
            - integrand_penal : np.ndarray
                Penalization term of the integrand.
            - integrand_reward : np.ndarray
                Reward term of the integrand.
        """
        gamma_opt_knots = self.gamma_optimal(knots)
        gamma_opt_e_knots = self.gamma_optimal_energy(knots)
        f_scen_knots = self.f_scenario(knots)

        discount_knots = np.exp(-self.r * (knots - t))
        mean_knots = np.array(
            [self.mean_production(u=knot, t=t, n_leg=n_leg) for knot in knots]
        )
        integrand_mean_cov = self.n_units * np.exp(
            np.exp(-self.b * (knots - t))[:, None] * x[None, :]
            + mean_knots[:, None]
            + 0.5 * self.variance_production(knots - t)[:, None]
        )
        integrand_alpha_beta = np.sum(
            (self.alpha * self.theta)[:, None] * gamma_opt_e_knots
            + (self.beta * self.theta)[:, None] * gamma_opt_e_knots**2,
            axis=0,
        )
        integrand_penal = np.maximum(gamma_opt_knots - f_scen_knots, 0) ** 2
        integrand_penal *= self.w_1
        integrand_reward = np.maximum(f_scen_knots - gamma_opt_knots, 0) ** 2
        integrand_reward *= self.w_2

        return (
            discount_knots,
            integrand_mean_cov,
            integrand_alpha_beta,
            integrand_penal,
            integrand_reward,
        )

    def h(self, t, x, n_leg: int = 20, return_integrand: bool = False):
        """
        Compute the firm's optimal value using the function `h`.

        The function `h` is defined as:
        h(t, x) = V_t with x = log production at time `t`.

        Parameters
        ----------
        t: float
            Current time.
        x: float or array-like
            Log production.
        n_leg : int, optional
            Number of nodes for Gauss-Legendre quadrature (default is 20).
        return_integrand: bool, optional
            If True, return the integrand components instead of the optimal
            value.

        Returns
        -------
        opt_value : float or array-like
            Optimal firm's value if `return_integrand` is False.
        integrand_components : dict
            Components of the integrand if `return_integrand` is True.
        """
        x = np.atleast_1d(x)
        knots, weights = utils.gauss_legendre(a=t, b=self.t_final, n=n_leg)

        (
            discount_knots,
            integrand_mean_cov,
            integrand_alpha_beta,
            integrand_penal,
            integrand_reward,
        ) = self._compute_integrand_components(knots, x, t, n_leg)

        if return_integrand:
            return {
                "knots": knots,
                "weights": weights,
                "integrand_mean_cov": integrand_mean_cov,
                "integrand_penal": integrand_penal,
                "integrand_reward": integrand_reward,
                "integrand_alpha_beta": integrand_alpha_beta,
                "discount_knots": discount_knots,
            }

        integrand = (
            integrand_mean_cov
            - integrand_alpha_beta[:, None]
            - integrand_penal[:, None]
            + integrand_reward[:, None]
        )
        weights_disc = (weights * discount_knots)[:, None]
        optimal_value = np.sum(weights_disc * integrand, axis=0)

        return optimal_value

    def proba_default_merton(self, t, barrier_t, n_leg=20):
        """
        Compute the probability of default P(V_t <= barrier_t) using Merton's
        model.

        V_t denotes the optimal's firm value.

        Parameters
        ----------
        t : float
            Time point at which to compute the probability of default.
        barrier_t : float
            Default barrier at time t.
        n_leg : int, optional
            Number of nodes for Gauss-Legendre quadrature (default is 20).

        Returns
        -------
        float
            Probability of default at time t.
        """

        (
            knots,
            weights,
            integrand_mean_cov,
            integrand_penal,
            integrand_reward,
            integrand_alpha_beta,
            discount_knots,
        ) = (
            self.h(t=t, x=0.0, n_leg=n_leg, return_integrand=True)[k]
            for k in [
                "knots",
                "weights",
                "integrand_mean_cov",
                "integrand_penal",
                "integrand_reward",
                "integrand_alpha_beta",
                "discount_knots",
            ]
        )
        integrand_mean_cov = integrand_mean_cov.ravel()

        integrand_no_x = -integrand_alpha_beta - integrand_penal
        integrand_no_x += integrand_reward
        integral_no_x = np.sum(weights * discount_knots * integrand_no_x)

        def fun(x):
            # minimize function to find the inverse
            integrand_x = np.exp(np.exp(-self.b * (knots - t)) * x)
            integrand_x *= integrand_mean_cov
            h_x = np.sum(weights * discount_knots * integrand_x)
            h_x += integral_no_x
            return (h_x - barrier_t) ** 2

        tmp = (
            optimize.minimize_scalar(fun=fun).x
            - (
                self.mean_production(u=t, t=0.0, n_leg=n_leg)
                + np.exp(-self.b * t) * np.log(self.prod_0)
            )
        ) / np.sqrt(self.variance_production(t))

        return stats.norm.cdf(tmp)

    @staticmethod
    def default_boundary_unpenalized(firm_no_pen, t, lbd_ref, n_leg=20):
        """
        Compute the default boundary L_T for an unpenalized firm.

        The default boundary is calculated using the formula:
        `P(V_T <= L) = 1 - exp(-lbd_ref * T)`

        Parameters
        ----------
        firm_no_pen : Firm
            Firm instance with zero emission constraint (w1 = w2 = 0).
        t : float
            Time point at which to calculate the default boundary.
        lbd_ref : float
            Reference intensity.
        n_leg : int, optional
            Number of quadrature points to use for the integration
            (default is 20).

        Returns
        -------
        float
            Default boundary L_T.

        Raises
        ------
        ValueError
            If the firm's penalization coefficient w_1 or reward coefficient
            w_2 is not zero.
        """
        if firm_no_pen.w_1 != 0.0:
            raise ValueError("Penalization coefficient w_1 must be >= 0.")

        if firm_no_pen.w_2 != 0.0:
            raise ValueError("Reward coefficient w_2 must be >= 0.")

        x = (
            stats.norm.ppf(1.0 - np.exp(-lbd_ref * t))
            * np.sqrt(firm_no_pen.variance_production(t))
            + firm_no_pen.mean_production(u=t, t=0.0, n_leg=n_leg)
            + np.exp(-firm_no_pen.b * t) * np.log(firm_no_pen.prod_0)
        )
        boundary = firm_no_pen.h(t=t, x=x, n_leg=n_leg)

        return boundary

    @staticmethod
    def default_intensity(
        tab_t: np.ndarray,
        prob_default: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the intensity for each probability of default.

        The intensity is calculated using the formula:
        lambda_t_{i-1} = log((1-pd_t_{i-1})/(1-pd_t_{i})) / (t_{i} - t_{i-1})

        Parameters
        ----------
        tab_t : np.ndarray
            Time points.
        prob_default : np.ndarray
            Probabilities of default at each time point.

        Returns
        -------
        np.ndarray
            Associated intensity for each probability of default.
        """
        return np.log((1.0 - prob_default[:-1]) / (1.0 - prob_default[1:])) / (
            tab_t[1:] - tab_t[:-1]
        )
