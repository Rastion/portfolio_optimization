import math
import random
import sys
from qubots.base_problem import BaseProblem
import os

class PortfolioOptimizationProblem(BaseProblem):
    """
    Portfolio Optimization Problem.

    Given a set of stocks with known covariance (risk) matrix and expected returns,
    decide what proportion of a unit portfolio to invest in each stock so that:
      - The entire portfolio is invested (i.e., the proportions sum to 1),
      - The portfolio profit is at least the given expected profit,
      - And the portfolio risk (a quadratic function of the proportions) is minimized.
    """
    
    def __init__(self, instance_file: str, **kwargs):
        # Read the instance file and store the data.
        self.expected_profit, self.nb_stocks, self.sigma, self.delta = self._read_instance(instance_file)
    
    def _read_instance(self, filename: str):
        """
        Reads an instance from a text file.

        The file format is:
          - First line: expected profit (a float, in percentage of the portfolio)
          - Second line: (possibly empty or a header)
          - Third line: number of stocks (an integer)
          - Next nb_stocks lines: each line has nb_stocks floats (the covariance matrix rows)
          - Last line: nb_stocks floats representing the expected return (variation) of each stock
        """

        # Resolve relative path with respect to this module’s directory.
        if not os.path.isabs(filename):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            filename = os.path.join(base_dir, filename)

        with open(filename, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
        
        # First line: expected profit.
        expected_profit = float(lines[0].split()[0])
        
        # Assume the number of stocks is on the second nonempty line.
        nb_stocks = int(lines[1].split()[0])
        
        # Read covariance matrix (rows start at line index 4, for nb_stocks lines).
        sigma = []
        for s in range(nb_stocks):
            row = [float(val) for val in lines[3+s].split()]
            sigma.append(row)
        
        # Read expected returns from the line immediately following the covariance matrix.
        delta = [float(val) for val in lines[nb_stocks+3].split()]
        
        return expected_profit, nb_stocks, sigma, delta
    
    def evaluate_solution(self, solution) -> float:
        """
        Evaluates a candidate solution.
        
        Expects:
          solution: a dictionary with key "portfolio" mapping to a list of floats of length nb_stocks.
                    These represent the proportion of the portfolio invested in each stock.
                    
        Returns:
          The portfolio risk (a scalar value) if the solution is feasible; otherwise, a high penalty.
        """
        # Check that the solution is well-formed.
        if not isinstance(solution, dict) or "portfolio" not in solution:
            return sys.maxsize
        portfolio = solution["portfolio"]
        if len(portfolio) != self.nb_stocks:
            return sys.maxsize
        
        # Constraint: All proportions must be between 0 and 1.
        for x in portfolio:
            if x < 0 or x > 1:
                return sys.maxsize
        
        # Constraint: The proportions must sum to 1.
        total = sum(portfolio)
        if abs(total - 1.0) > 1e-6:
            return sys.maxsize
        
        # Compute the portfolio profit.
        profit = sum(portfolio[s] * self.delta[s] for s in range(self.nb_stocks))
        # Constraint: The profit must be at least the expected profit.
        if profit < self.expected_profit - 1e-6:
            return sys.maxsize
        
        # Compute the portfolio risk: sum_{s,t} x_s * x_t * sigma[s][t]
        risk = 0.0
        for s in range(self.nb_stocks):
            for t in range(self.nb_stocks):
                risk += portfolio[s] * portfolio[t] * self.sigma[s][t]
        
        return risk
    
    def random_solution(self):
        """
        Generates a random candidate solution.
        
        Returns a dictionary with key "portfolio" mapping to a random probability vector of length nb_stocks.
        """
        raw = [random.random() for _ in range(self.nb_stocks)]
        total = sum(raw)
        portfolio = [x/total for x in raw]
        return {"portfolio": portfolio}
