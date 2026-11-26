"""Exception raised by a measurement wrapper when the evaluation budget is used up."""


class BudgetReached(RuntimeError):
    pass
